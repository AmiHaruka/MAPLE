"""Deterministic EFH contracts; no checkpoint assets or model downloads."""

import numpy as np
import pytest
import torch
from ase import Atoms

from maple.function.calculator.aimnet._aimnet2_batch_calculator import AIMNet2BatchCalc
from maple.function.calculator.aimnet._aimnet2_calculator import (
    AIMNET2_PADDED_PER_MOLECULE_LAYOUT,
)
from maple.function.calculator.calculator_base import EV2HARTREE


class MolecularEnergy(torch.nn.Module):
    num_charge_channels = 1

    def __init__(self, *, dtype=torch.float64, max_batch=None, failure=None, nonlinear=False):
        super().__init__()
        self.scale = torch.nn.Parameter(torch.ones((), dtype=dtype), requires_grad=False)
        self.max_batch = max_batch
        self.failure = failure
        self.nonlinear = nonlinear
        self.calls = []

    def forward(self, data):
        size = len(data["charge"]) - 1
        self.calls.append((size, str(data["coord"].device)))
        if self.failure:
            raise RuntimeError(self.failure)
        if self.max_batch is not None and size > self.max_batch:
            raise RuntimeError("CUDA out of memory: synthetic bounded-EFH test")
        coord = data["coord"]
        if self.nonlinear:
            values = coord ** 4 + 0.3 * coord ** 3 + torch.sin(coord) * torch.exp(0.1 * coord)
        else:
            values = 0.5 * coord ** 2
        atomic = values.sum(-1) * data["numbers"] * self.scale
        energies = torch.zeros(len(data["charge"]), dtype=torch.float64, device=coord.device)
        return {"energy": energies.index_add(0, data["mol_idx"].long(), atomic.double())}


def adapter(model, *, batch_size=None, cap=8):
    return AIMNet2BatchCalc(
        model=model,
        device="cpu",
        batch_size=batch_size,
        auto_batch_hard_cap=cap,
        batch_energy_layout=AIMNET2_PADDED_PER_MOLECULE_LAYOUT,
    )


def structures():
    return [
        Atoms("H", positions=[[0.2, -0.3, 0.4]]),
        Atoms("OH2", positions=[[0.0, 0.0, 0.1], [0.9, 0.0, 0.0], [-0.2, 0.9, 0.0]]),
        Atoms("H2", positions=[[0.0, 0.0, 0.0], [0.75, 0.0, 0.0]]),
    ]


def test_efh_honors_explicit_chunks_and_keeps_ef_outputs():
    model = MolecularEnergy()
    calc = adapter(model, batch_size=2)
    atoms = structures()
    calc.prepare(atoms)
    e_ref, f_ref = calc.get_ef_gpu()
    model.calls.clear()
    energy, force, hessian, _ = calc.get_efh_gpu()
    assert [size for size, _ in model.calls] == [2, 1]
    torch.testing.assert_close(energy, e_ref, rtol=0, atol=0)
    torch.testing.assert_close(force, f_ref, rtol=0, atol=0)
    for index, at in enumerate(atoms):
        dof = 3 * len(at)
        expected = np.diag(np.repeat(at.numbers, 3) * EV2HARTREE)
        np.testing.assert_allclose(hessian[index, :dof, :dof].numpy(), expected, rtol=0, atol=1e-15)
        assert torch.count_nonzero(hessian[index, dof:]) == 0
        assert torch.count_nonzero(hessian[index, :, dof:]) == 0


def test_efh_honors_auto_cap():
    model = MolecularEnergy()
    calc = adapter(model, cap=2)
    calc.prepare(structures())
    calc.get_efh_gpu()
    assert [size for size, _ in model.calls] == [2, 1]


def test_efh_padding_reports_the_fixed_prepared_width_after_shrink():
    model = MolecularEnergy()
    calc = adapter(model)
    atoms = structures()[:2]
    calc.prepare(atoms, fixed_nmax=18)
    energy, force, hessian, padding = calc.get_efh_gpu()
    assert energy.shape == (2,)
    assert force.shape == (2, 18)
    assert hessian.shape == (2, 18, 18)
    assert padding.tolist() == [5, 3]


def test_efh_oom_backoff_preserves_order_device_and_coordinates():
    model = MolecularEnergy(max_batch=2)
    calc = adapter(model)
    atoms = structures() + [structures()[0].copy()]
    calc.prepare(atoms)
    initial = calc.coord.clone()
    energy, _, _, _ = calc.get_efh_gpu()
    assert [size for size, _ in model.calls] == [4, 2, 2]
    assert {device for _, device in model.calls} == {"cpu"}
    expected = [0.5 * np.sum(at.numbers[:, None] * at.positions ** 2) * EV2HARTREE for at in atoms]
    np.testing.assert_allclose(energy.numpy(), expected, rtol=0, atol=1e-15)
    torch.testing.assert_close(calc.coord, initial, rtol=0, atol=0)


def test_efh_unrecoverable_oom_stops_at_one_and_preserves_coordinates():
    model = MolecularEnergy(max_batch=0)
    calc = adapter(model)
    calc.prepare(structures())
    initial = calc.coord.clone()
    with pytest.raises(RuntimeError, match="out of memory"):
        calc.get_efh_gpu()
    assert [size for size, _ in model.calls] == [3, 1]
    torch.testing.assert_close(calc.coord, initial, rtol=0, atol=0)


def test_efh_does_not_retry_unrelated_backend_errors():
    model = MolecularEnergy(failure="synthetic model schema failure")
    calc = adapter(model)
    calc.prepare(structures())
    with pytest.raises(RuntimeError, match="model schema failure"):
        calc.get_efh_gpu()
    assert len(model.calls) == 1


def test_efh_hartree_scaling_precedes_second_derivative_rounding():
    model = MolecularEnergy(dtype=torch.float32, nonlinear=True)
    calc = adapter(model)
    atom = Atoms("O", positions=[[0.29, -0.81, 0.67]])
    calc.prepare([atom])
    # Differentiate the checkpoint's required FP32 input, regardless of
    # the optimizer coordinate storage used by the adapter.
    coord = calc.coord.detach().to(calc.dtype).requires_grad_(True)
    values = coord ** 4 + 0.3 * coord ** 3 + torch.sin(coord) * torch.exp(0.1 * coord)
    energy_ha = (values.sum(-1) * 8).double().sum() * EV2HARTREE
    grad = torch.autograd.grad(energy_ha, coord, create_graph=True)[0]
    expected = torch.stack([
        torch.autograd.grad(value, coord, retain_graph=True)[0].reshape(-1)
        for value in grad.reshape(-1)
    ]).double()
    _, _, hessian, _ = calc.get_efh_gpu()
    torch.testing.assert_close(hessian[0], expected, rtol=0, atol=0)
