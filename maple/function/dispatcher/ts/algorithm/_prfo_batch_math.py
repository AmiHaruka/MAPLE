"""Device-native Torch mathematics for the experimental batched P-RFO path."""

from __future__ import annotations

import math
import torch


@torch.no_grad()
def prfo_step_batched(
    w: torch.Tensor,
    V: torch.Tensor,
    gp: torch.Tensor,
    target_mode: torch.Tensor,
    trust_radius: torch.Tensor,
    *,
    evals_eps: float = 1.0e-10,
    max_bisect_it: int = 60,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Solve independent shared-alpha RS-P-RFO systems as a Torch batch.

    Invalid and unbracketed rows receive a zero step and a false success bit.
    The bracket and bisection state is per-row and remains on the input device.
    """
    if w.ndim != 2 or gp.shape != w.shape:
        raise ValueError("w and gp must have matching (B, N) shapes")
    batch, width = w.shape
    if V.shape != (batch, width, width):
        raise ValueError("V must have shape (B, N, N)")
    if target_mode.shape != (batch,) or trust_radius.shape != (batch,):
        raise ValueError("target_mode and trust_radius must have shape (B,)")
    if not w.is_floating_point() or V.dtype != w.dtype or gp.dtype != w.dtype:
        raise TypeError("w, V, and gp must use the same floating dtype")
    if any(t.device != w.device for t in (V, gp, target_mode, trust_radius)):
        raise ValueError("all inputs must be on the same device")
    if trust_radius.dtype != w.dtype:
        raise TypeError("trust_radius must match the eigensystem dtype")
    if not math.isfinite(float(evals_eps)) or evals_eps <= 0.0:
        raise ValueError("evals_eps must be finite and positive")
    if (isinstance(max_bisect_it, bool)
            or not isinstance(max_bisect_it, int) or max_bisect_it < 1):
        raise ValueError("max_bisect_it must be a positive integer")

    steps = torch.zeros_like(gp)
    success = torch.zeros(batch, dtype=torch.bool, device=w.device)
    valid = (
        torch.isfinite(w).all(1) & torch.isfinite(V).all((1, 2))
        & torch.isfinite(gp).all(1) & torch.isfinite(trust_radius)
        & (trust_radius > 0.0) & (target_mode >= 0) & (target_mode < width)
    )
    rows = valid.nonzero(as_tuple=False).flatten()
    if rows.numel() == 0:
        return steps, success

    wr, Vr, gr = w[rows], V[rows], gp[rows]
    target = target_mode[rows]
    radius = trust_radius[rows]
    count = rows.numel()
    row_index = torch.arange(count, device=w.device)
    target_curvature = wr[row_index, target]
    target_gradient = gr[row_index, target]
    all_indices = torch.arange(width, device=w.device).expand(count, width)
    complement = all_indices[all_indices != target[:, None]].reshape(count, width - 1)
    comp_curvature = wr.gather(1, complement) if width > 1 else wr[:, :0]
    comp_gradient = gr.gather(1, complement) if width > 1 else gr[:, :0]

    def step_for_alpha(alpha: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        root_alpha = torch.sqrt(alpha)
        target_aug = torch.zeros((count, 2, 2), dtype=w.dtype, device=w.device)
        coupling = target_gradient / root_alpha
        target_aug[:, 0, 1] = coupling
        target_aug[:, 1, 0] = coupling
        target_aug[:, 1, 1] = target_curvature / alpha
        target_root = torch.linalg.eigvalsh(target_aug)[:, -1]
        target_denom = target_curvature - alpha * target_root
        target_denom = torch.where(
            target_denom.abs() < evals_eps,
            torch.where(target_denom < 0.0, -evals_eps, evals_eps),
            target_denom,
        )
        projected = torch.zeros((count, width), dtype=w.dtype, device=w.device)
        projected[row_index, target] = -target_gradient / target_denom

        if width > 1:
            comp_aug = torch.zeros(
                (count, width, width), dtype=w.dtype, device=w.device
            )
            comp_coupling = comp_gradient / root_alpha[:, None]
            comp_aug[:, 0, 1:] = comp_coupling
            comp_aug[:, 1:, 0] = comp_coupling
            comp_aug[:, 1:, 1:] = torch.diag_embed(comp_curvature / alpha[:, None])
            comp_root = torch.linalg.eigvalsh(comp_aug)[:, 0]
            comp_denom = comp_curvature - alpha[:, None] * comp_root[:, None]
            comp_denom = torch.where(
                comp_denom.abs() < evals_eps,
                torch.where(comp_denom < 0.0, -evals_eps, evals_eps),
                comp_denom,
            )
            projected.scatter_(1, complement, -comp_gradient / comp_denom)
        step = torch.bmm(Vr, projected.unsqueeze(-1)).squeeze(-1)
        finite = torch.isfinite(step).all(1)
        return step, finite

    one = torch.ones(count, dtype=w.dtype, device=w.device)
    unrestricted, finite = step_for_alpha(one)
    unrestricted_norm = torch.linalg.norm(unrestricted, dim=1)
    tolerance = 1.0e-10 * torch.maximum(one, radius)
    direct = finite & (unrestricted_norm <= radius)
    solved_steps = torch.zeros_like(unrestricted)
    solved_steps[direct] = unrestricted[direct]

    needs = finite & ~direct
    alpha_lo = one.clone()
    alpha_hi = torch.full_like(one, 2.0)
    restricted, restricted_finite = step_for_alpha(alpha_hi)
    restricted_norm = torch.linalg.norm(restricted, dim=1)
    needs &= restricted_finite
    for _ in range(max_bisect_it):
        expand = needs & (restricted_norm > radius + tolerance)
        if not bool(expand.any()):
            break
        alpha_lo = torch.where(expand, alpha_hi, alpha_lo)
        alpha_hi = torch.where(expand, 2.0 * alpha_hi, alpha_hi)
        candidate, candidate_finite = step_for_alpha(alpha_hi)
        restricted = torch.where(expand[:, None], candidate, restricted)
        restricted_norm = torch.linalg.norm(restricted, dim=1)
        needs &= (~expand) | candidate_finite

    bracketed = needs & (restricted_norm <= radius + tolerance)
    bisect_lo, bisect_hi = alpha_lo.clone(), alpha_hi.clone()
    converged = torch.zeros_like(bracketed)
    for _ in range(max_bisect_it):
        active = bracketed & ~converged
        if not bool(active.any()):
            break
        midpoint = 0.5 * (bisect_lo + bisect_hi)
        candidate, candidate_finite = step_for_alpha(midpoint)
        norm = torch.linalg.norm(candidate, dim=1)
        outside = active & candidate_finite & (norm > radius)
        inside = active & candidate_finite & ~outside
        bisect_lo = torch.where(outside, midpoint, bisect_lo)
        bisect_hi = torch.where(inside, midpoint, bisect_hi)
        restricted = torch.where(inside[:, None], candidate, restricted)
        close = (norm - radius).abs() <= tolerance
        restricted = torch.where((active & candidate_finite & close)[:, None], candidate, restricted)
        converged |= active & candidate_finite & close
        bracketed &= candidate_finite

    solved_steps[bracketed] = restricted[bracketed]
    row_success = direct | bracketed
    steps[rows] = solved_steps
    success[rows] = row_success
    return steps, success
