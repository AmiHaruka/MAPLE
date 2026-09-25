"""Small graph-batching helpers for MAPLE's scripted MACE wrappers.

The wrappers in this package already build a no-PBC radius graph from ASE
atoms for single-structure inference.  These helpers preserve that exact graph
semantics while concatenating independent structures into one disconnected
graph batch, matching the PyG/MACE batching pattern without depending on MACE's
training-time dataloader at runtime.
"""
from __future__ import annotations

from typing import Sequence

import numpy as np
import torch

from ._common import one_hot_node_attrs as _one_hot_node_attrs
from ._common import radius_graph_no_pbc


def one_hot_node_attrs(
    Z: torch.Tensor,
    atomic_number_table: Sequence[int],
    *,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Keep the batch helper's call/error contract over the shared core."""
    return _one_hot_node_attrs(Z, list(atomic_number_table), dtype=dtype)


def concatenate_positions(
    atoms_list: Sequence,
    *,
    device: torch.device,
    dtype: torch.dtype,
    requires_grad: bool,
) -> tuple[torch.Tensor, list[int]]:
    counts = [len(at) for at in atoms_list]
    if counts:
        positions_np = np.concatenate([at.get_positions() for at in atoms_list], axis=0)
    else:
        positions_np = np.zeros((0, 3), dtype=np.float64)
    positions = torch.tensor(
        positions_np,
        dtype=dtype,
        device=device,
        requires_grad=requires_grad,
    )
    return positions, counts


def batch_index_and_ptr(counts: Sequence[int], *, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    batch_parts = [
        torch.full((int(n),), i, dtype=torch.long, device=device)
        for i, n in enumerate(counts)
    ]
    batch = torch.cat(batch_parts, dim=0) if batch_parts else torch.zeros(0, dtype=torch.long, device=device)
    ptr = [0]
    for n in counts:
        ptr.append(ptr[-1] + int(n))
    return batch, torch.tensor(ptr, dtype=torch.long, device=device)


def batched_radius_graph_no_pbc(
    positions: torch.Tensor,
    counts: Sequence[int],
    r_max: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Build a disconnected-graph batch by stacking per-structure graphs.

    Each structure's graph is built independently (no cross-structure edges)
    and its edge indices are offset into the concatenated node range.  The
    dense ``(n, n)`` all-pairs distance tensors in ``radius_graph_no_pbc`` are
    allocated per structure inside the loop and freed each iteration, so peak
    memory is ``O(max_i N_i^2)`` for the largest structure in the chunk, not
    ``O((sum_i N_i)^2)`` over the whole concatenated stack.  Chunk size on the
    MACE path is bounded by the auto batch sizer's image cap
    (``_AutoBatchSizer._apply_math_cap``).
    """
    edge_parts = []
    shift_parts = []
    offset = 0
    for n in counts:
        n = int(n)
        local_pos = positions[offset:offset + n]
        edge_index, shifts = radius_graph_no_pbc(local_pos, r_max)
        if edge_index.numel() > 0:
            edge_parts.append(edge_index + offset)
            shift_parts.append(shifts)
        offset += n

    if edge_parts:
        return torch.cat(edge_parts, dim=1), torch.cat(shift_parts, dim=0)
    return (
        torch.zeros((2, 0), dtype=torch.long, device=positions.device),
        torch.zeros((0, 3), dtype=positions.dtype, device=positions.device),
    )


def batched_atomic_numbers(atoms_list: Sequence, *, device: torch.device) -> torch.Tensor:
    arrays = [at.get_atomic_numbers() for at in atoms_list]
    if arrays:
        numbers = np.concatenate(arrays, axis=0)
    else:
        numbers = np.zeros(0, dtype=np.int64)
    return torch.tensor(numbers, dtype=torch.long, device=device)


def energy_vector_from_output(
    energy_out: torch.Tensor,
    *,
    batch_size: int,
    n_atoms_total: int,
    batch: torch.Tensor,
) -> torch.Tensor:
    energy_vec = energy_out.reshape(-1)
    if energy_vec.numel() == batch_size:
        return energy_vec
    if energy_vec.numel() == n_atoms_total:
        return torch.zeros(
            batch_size,
            dtype=energy_vec.dtype,
            device=energy_vec.device,
        ).scatter_add(0, batch, energy_vec)
    if energy_vec.numel() == 1 and batch_size == 1:
        return energy_vec
    raise RuntimeError(f"Unexpected MACE energy shape {tuple(energy_out.shape)}")


def build_mace_data_dict_batch(
    atoms_list: Sequence,
    *,
    atomic_numbers: Sequence[int],
    r_max: float,
    device: torch.device,
    dtype: torch.dtype,
    requires_grad: bool,
):
    positions, counts = concatenate_positions(
        atoms_list,
        device=device,
        dtype=dtype,
        requires_grad=requires_grad,
    )
    Z = batched_atomic_numbers(atoms_list, device=device)
    node_attrs = one_hot_node_attrs(Z, atomic_numbers, dtype=dtype)
    edge_index, shifts = batched_radius_graph_no_pbc(positions, counts, r_max)
    batch, ptr = batch_index_and_ptr(counts, device=device)
    B = len(atoms_list)
    N = positions.size(0)

    data_dict = {
        "batch": batch,
        "cell": torch.zeros(3, 3, dtype=dtype, device=device),
        "charges": torch.zeros(N, dtype=dtype, device=device),
        "dipole": torch.zeros(B, 3, dtype=dtype, device=device),
        "edge_index": edge_index,
        "energy": torch.zeros(B, dtype=dtype, device=device),
        "energy_weight": torch.zeros(B, dtype=dtype, device=device),
        "forces": torch.zeros(N, 3, dtype=dtype, device=device),
        "forces_weight": torch.zeros(B, dtype=dtype, device=device),
        "node_attrs": node_attrs,
        "positions": positions,
        "ptr": ptr,
        "shifts": shifts,
        "stress": torch.zeros(B, 3, 3, dtype=dtype, device=device),
        "stress_weight": torch.zeros(B, dtype=dtype, device=device),
        "unit_shifts": torch.zeros(edge_index.size(1), 3, dtype=dtype, device=device),
        "virials": torch.zeros(B, 3, 3, dtype=dtype, device=device),
        "virials_weight": torch.zeros(B, dtype=dtype, device=device),
        "weight": torch.ones(B, dtype=dtype, device=device),
    }
    local_or_ghost = torch.ones(N, dtype=dtype, device=device)
    return data_dict, local_or_ghost, counts


def build_mace_tuple_batch(
    atoms_list: Sequence,
    *,
    atomic_numbers: Sequence[int],
    r_max: float,
    device: torch.device,
    dtype: torch.dtype,
    requires_grad: bool,
):
    positions, counts = concatenate_positions(
        atoms_list,
        device=device,
        dtype=dtype,
        requires_grad=requires_grad,
    )
    Z = batched_atomic_numbers(atoms_list, device=device)
    node_attrs = one_hot_node_attrs(Z, atomic_numbers, dtype=dtype)
    edge_index, shifts = batched_radius_graph_no_pbc(positions, counts, r_max)
    batch, ptr = batch_index_and_ptr(counts, device=device)
    return (positions, node_attrs, edge_index, shifts, batch, ptr), counts
