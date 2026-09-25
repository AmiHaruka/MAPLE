"""Exact, order-sensitive locks for AIMNet2's original neighbor-list contract."""

import numpy as np
import pytest
import torch

from maple.function.calculator.aimnet._aimnet2_calculator import (
    _molecule_local_indices,
    build_aimnet2_neighbor_matrices,
    nblist_all_pairs_padded_multi,
    nblist_dense_padded_multi,
)


DEVICES = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])


def _original_neighbors(coord, mol_idx, cutoff):
    """Pre-optimization row loop, including exact distance arithmetic."""
    n_atoms = len(coord)
    if n_atoms == 0:
        return torch.full((1, 1), 0, dtype=torch.int32, device=coord.device)
    mask = (mol_idx[:, None] == mol_idx[None, :]) & ~torch.eye(
        n_atoms, dtype=torch.bool, device=coord.device
    )
    if np.isfinite(cutoff):
        diff = coord[:, None, :] - coord[None, :, :]
        mask &= torch.sum(diff ** 2, dim=-1) <= cutoff ** 2
    width = max(int(mask.sum(dim=1).max().item()), 1)
    result = torch.full(
        (n_atoms + 1, width), n_atoms, dtype=torch.int32, device=coord.device
    )
    for i in range(n_atoms):
        neighbors = torch.nonzero(mask[i], as_tuple=False).flatten()
        result[i, :neighbors.numel()] = neighbors.to(torch.int32)
    return result


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("labels", [[], [7], [3, 1, 3, 7, 1, 3], [0] * 12])
@pytest.mark.parametrize("cutoff", [0.01, 1.0, 5.0, float("inf")])
def test_neighbor_order_and_padding_exactly_match_original(device, dtype, labels, cutoff):
    rng = np.random.default_rng(290922)
    # Strided inputs ensure packing does not rely on contiguous storage.
    coordinates = torch.tensor(
        rng.normal(size=(len(labels), 6)), dtype=dtype, device=device
    )[:, ::2]
    molecules = torch.tensor(labels, dtype=torch.int32, device=device)
    expected = _original_neighbors(coordinates, molecules, cutoff)
    actual = (
        nblist_all_pairs_padded_multi(molecules)
        if np.isinf(cutoff)
        else nblist_dense_padded_multi(coordinates, molecules, cutoff)
    )
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    assert actual.dtype == torch.int32
    assert actual.device == coordinates.device


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_cutoff_boundary_coincident_atoms_and_cross_molecule_exclusion(device, dtype):
    cutoff = torch.tensor(5.0, dtype=dtype, device=device)
    below = torch.nextafter(cutoff, torch.zeros_like(cutoff))
    above = torch.nextafter(cutoff, torch.full_like(cutoff, float("inf")))
    coordinates = torch.zeros((6, 3), dtype=dtype, device=device)
    coordinates[1:4, 0] = torch.stack([below, cutoff, above])
    molecules = torch.tensor([0, 0, 0, 0, 0, 1], dtype=torch.int32, device=device)
    short, long_range = build_aimnet2_neighbor_matrices(
        coordinates, molecules, cutoff=5.0, cutoff_lr=float("inf")
    )
    torch.testing.assert_close(
        short, _original_neighbors(coordinates, molecules, 5.0), rtol=0, atol=0
    )
    torch.testing.assert_close(
        long_range, _original_neighbors(coordinates, molecules, float("inf")),
        rtol=0, atol=0,
    )
    assert short[0].tolist() == [1, 2, 4, 6]
    assert long_range[0].tolist() == [1, 2, 3, 4]


@pytest.mark.parametrize("device", DEVICES)
def test_dsf_keeps_independent_short_and_long_range_masks(device):
    coord = torch.tensor([[0.0, 0.0, 0.0], [10.0, 0.0, 0.0]], device=device)
    molecules = torch.zeros(2, dtype=torch.int32, device=device)
    short, long_range = build_aimnet2_neighbor_matrices(
        coord, molecules, cutoff=5.0, cutoff_lr=15.0
    )
    assert short.tolist() == [[2], [2], [2]]
    assert long_range.tolist() == [[1], [0], [2]]


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("cutoff_lr", [float("inf"), 7.0])
def test_mixed_molecule_order_preserves_exact_global_neighbor_indices(
    device, dtype, cutoff_lr
):
    rng = np.random.default_rng(230923)
    labels = torch.tensor(
        [4, 4, 1, 7, 1, 4, 2, 2, 7, 1, 9, 9, 9, 2, 7],
        dtype=torch.int32, device=device,
    )
    coordinates = torch.tensor(
        rng.normal(size=(len(labels), 3)), dtype=dtype, device=device
    )

    short, long_range = build_aimnet2_neighbor_matrices(
        coordinates, labels, cutoff=2.0, cutoff_lr=cutoff_lr
    )

    torch.testing.assert_close(
        short, _original_neighbors(coordinates, labels, 2.0), rtol=0, atol=0
    )
    torch.testing.assert_close(
        long_range, _original_neighbors(coordinates, labels, cutoff_lr),
        rtol=0, atol=0,
    )


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("cutoff_lr", [float("inf"), 3.0])
def test_bucketed_large_interleaved_molecules_match_original(device, cutoff_lr):
    # This crosses the CPU local-bucket boundary, and CUDA also crosses its
    # launch-bound dense boundary. Labels deliberately interleave globally.
    n_per_molecule = 129 if device == "cuda" else 12
    n_molecules = 16 if device == "cuda" else 8
    rng = np.random.default_rng(230924)
    n_atoms = n_per_molecule * n_molecules
    coordinates = torch.tensor(
        rng.normal(size=(n_atoms, 3)), dtype=torch.float32, device=device
    )
    cutoff = torch.tensor(1.7, dtype=coordinates.dtype, device=device)
    coordinates[0] = 0.0
    coordinates[n_molecules:n_molecules * 4:n_molecules, :] = 0.0
    coordinates[n_molecules, 0] = torch.nextafter(cutoff, torch.zeros_like(cutoff))
    coordinates[n_molecules * 2, 0] = cutoff
    coordinates[n_molecules * 3, 0] = torch.nextafter(
        cutoff, torch.full_like(cutoff, float("inf"))
    )
    labels = torch.arange(n_molecules, dtype=torch.int32, device=device).repeat(
        n_per_molecule
    ) * 3 + 7
    short, long_range = build_aimnet2_neighbor_matrices(
        coordinates, labels, cutoff=1.7, cutoff_lr=cutoff_lr
    )
    expected_short = _original_neighbors(coordinates.cpu(), labels.cpu(), 1.7)
    expected_long = _original_neighbors(coordinates.cpu(), labels.cpu(), cutoff_lr)
    torch.testing.assert_close(short.cpu(), expected_short, rtol=0, atol=0)
    torch.testing.assert_close(long_range.cpu(), expected_long, rtol=0, atol=0)


@pytest.mark.parametrize("device", DEVICES)
def test_large_single_molecule_retains_exact_original_neighbor_order(device):
    rng = np.random.default_rng(230925)
    coord = torch.tensor(rng.normal(size=(120, 3)), dtype=torch.float32, device=device)
    mol_idx = torch.full((120,), -7, dtype=torch.int32, device=device)
    short, long_range = build_aimnet2_neighbor_matrices(
        coord, mol_idx, cutoff=2.0, cutoff_lr=float("inf")
    )
    torch.testing.assert_close(
        short.cpu(), _original_neighbors(coord.cpu(), mol_idx.cpu(), 2.0), rtol=0, atol=0
    )
    torch.testing.assert_close(
        long_range.cpu(), _original_neighbors(coord.cpu(), mol_idx.cpu(), float("inf")),
        rtol=0, atol=0,
    )


def test_extremely_ragged_groups_fall_back_before_allocating_padded_masks():
    labels = torch.tensor([0] * 70 + list(range(1, 71)), dtype=torch.int32)
    assert _molecule_local_indices(labels) is None
    coord = torch.arange(len(labels), dtype=torch.float32)[:, None].expand(-1, 3)
    actual = nblist_dense_padded_multi(coord, labels, cutoff=1.0)
    torch.testing.assert_close(actual, _original_neighbors(coord, labels, 1.0), rtol=0, atol=0)
