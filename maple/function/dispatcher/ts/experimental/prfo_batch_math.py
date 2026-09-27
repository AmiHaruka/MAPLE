"""Device-native Torch mathematics for the experimental batched P-RFO path."""

from __future__ import annotations

import math
import torch


def _safe_row_norm(values: torch.Tensor) -> torch.Tensor:
    """Vector 2-norm without avoidable intermediate square overflow."""
    maximum = values.abs().amax(dim=1)
    safe_maximum = torch.where(maximum > 0.0, maximum, torch.ones_like(maximum))
    scaled = values / safe_maximum[:, None]
    return maximum * torch.sqrt(scaled.square().sum(dim=1))


def _smallest_arrowhead_root_batched(
    diagonal: torch.Tensor,
    coupling: torch.Tensor,
    *,
    max_iterations: int = 96,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Torch-native smallest roots of ``[[0,b.T],[b,diag(d)]]``.

    Each row is solved only in the extremal interval below its first active
    pole.  Exactly decoupled entries are deflated and compared afterward.
    """
    if diagonal.shape != coupling.shape or diagonal.ndim != 2:
        raise ValueError("diagonal and coupling must have matching (B, N) shapes")
    count, width = diagonal.shape
    roots = torch.zeros(count, dtype=diagonal.dtype, device=diagonal.device)
    success = torch.isfinite(diagonal).all(1) & torch.isfinite(coupling).all(1)
    if width == 0:
        return roots, success

    active = coupling != 0.0
    has_active = active.any(1)
    zeros = torch.zeros_like(diagonal)
    scale = torch.maximum(
        torch.where(active, diagonal.abs(), zeros).amax(1),
        torch.where(active, coupling.abs(), zeros).amax(1),
    )
    nonzero_scale = scale > 0.0
    safe_scale = torch.where(nonzero_scale, scale, torch.ones_like(scale))
    d = torch.where(active, diagonal / safe_scale[:, None], zeros)
    b = torch.where(active, coupling / safe_scale[:, None], zeros)
    lost_coupling = (
        active & ((b == 0.0) | ((b != 0.0) & (b * b == 0.0)))
    ).any(1)
    success &= ~lost_coupling

    inf = torch.full_like(d, float("inf"))
    active_pole = torch.minimum(
        torch.zeros_like(scale), torch.where(active, d, inf).amin(1)
    )
    original_active_pole = torch.minimum(
        torch.zeros_like(scale),
        torch.where(active, diagonal, inf).amin(1),
    )
    inactive_min = torch.where(~active, diagonal, inf).amin(1)
    decoupled = torch.minimum(
        torch.zeros_like(scale),
        torch.where(torch.isfinite(inactive_min), inactive_min, torch.zeros_like(scale)),
    )
    roots = decoupled.clone()

    eligible = success & has_active & nonzero_scale
    active_count = active.sum(1)
    single_rows = (eligible & (active_count == 1)).nonzero(
        as_tuple=False
    ).flatten()
    if single_rows.numel() > 0:
        active_index = active[single_rows].to(torch.int64).argmax(1)
        local_rows = torch.arange(single_rows.numel(), device=diagonal.device)
        single_d = d[single_rows][local_rows, active_index]
        single_b = b[single_rows][local_rows, active_index]
        length = torch.hypot(single_d, 2.0 * single_b)
        root_scaled = torch.where(
            single_d <= 0.0,
            0.5 * (single_d - length),
            -2.0 * single_b * (single_b / (length + single_d)),
        )
        root_original = root_scaled * scale[single_rows]
        single_ok = (
            torch.isfinite(root_scaled) & torch.isfinite(root_original)
            & (root_scaled <= active_pole[single_rows])
            & (root_original <= original_active_pole[single_rows])
            & ~((active_pole[single_rows] == 0.0) & (root_scaled == 0.0))
            & ~((root_scaled != 0.0) & (root_original == 0.0))
        )
        roots[single_rows] = torch.where(
            single_ok,
            torch.minimum(root_original, decoupled[single_rows]),
            roots[single_rows],
        )
        success[single_rows] &= single_ok

    rows = (eligible & (active_count > 1)).nonzero(as_tuple=False).flatten()
    if rows.numel() == 0:
        return roots, success
    dr, br = d[rows], b[rows]
    ar = active[rows]
    pole = active_pole[rows]
    coupling_norm = _safe_row_norm(br)
    lower = torch.nextafter(
        pole - coupling_norm,
        torch.full_like(pole, float("-inf")),
    )
    upper = torch.nextafter(pole, torch.full_like(pole, float("-inf")))

    def secular(x):
        denominator = dr - x[:, None]
        terms = torch.where(
            ar, br * (br / denominator), torch.zeros_like(br)
        )
        return -x - terms.sum(1)

    f_lower = secular(lower)
    f_upper = secular(upper)
    valid_bracket = (
        torch.isfinite(f_lower) & ~torch.isnan(f_upper) & (f_lower >= 0.0)
    )
    rounded_to_pole = valid_bracket & (f_upper > 0.0)
    solve = valid_bracket & ~rounded_to_pole
    lo, hi = lower.clone(), upper.clone()
    bisect_valid = torch.ones_like(solve)
    stagnated = torch.zeros_like(solve)
    for _ in range(max_iterations):
        midpoint = lo + 0.5 * (hi - lo)
        value = secular(midpoint)
        bad_value = solve & torch.isnan(value)
        bisect_valid &= ~bad_value
        stalled = (
            solve & bisect_valid & ((midpoint == lo) | (midpoint == hi))
        )
        stagnated |= stalled
        movable = solve & bisect_valid & ~stalled
        go_right = value > 0.0
        lo = torch.where(movable & go_right, midpoint, lo)
        hi = torch.where(movable & ~go_right, midpoint, hi)
    adjacent = torch.nextafter(lo, torch.full_like(lo, float("inf"))) >= hi
    resolved = rounded_to_pole | (
        solve & bisect_valid & (stagnated | adjacent)
    )
    active_root_scaled = torch.where(
        rounded_to_pole, pole, lo + 0.5 * (hi - lo)
    )
    row_ok = (
        valid_bracket & resolved & torch.isfinite(active_root_scaled)
        & (active_root_scaled <= pole)
        & ~((pole == 0.0) & (active_root_scaled == 0.0))
    )
    rescaled_root = active_root_scaled * scale[rows]
    row_ok &= (
        torch.isfinite(rescaled_root)
        & (rescaled_root <= original_active_pole[rows])
        & ~((active_root_scaled != 0.0) & (rescaled_root == 0.0))
    )
    active_root = torch.minimum(
        rescaled_root, decoupled[rows]
    )
    roots[rows] = torch.where(row_ok, active_root, roots[rows])
    success[rows] &= row_ok
    return roots, success


def _extremal_arrowhead_root_batched(
    diagonal: torch.Tensor, coupling: torch.Tensor, *, maximum: bool
) -> tuple[torch.Tensor, torch.Tensor]:
    if maximum:
        root, success = _smallest_arrowhead_root_batched(-diagonal, coupling)
        return -root, success
    return _smallest_arrowhead_root_batched(diagonal, coupling)


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
        alpha_valid = torch.isfinite(alpha) & (alpha > 0.0)
        root_alpha = torch.sqrt(alpha)
        scaled_target_gradient = root_alpha * target_gradient
        target_length = torch.hypot(
            target_curvature, 2.0 * scaled_target_gradient
        )
        target_denom = torch.where(
            target_curvature <= 0.0,
            0.5 * (target_curvature - target_length),
            -2.0 * scaled_target_gradient * (
                scaled_target_gradient / (target_length + target_curvature)
            ),
        )
        target_decoupled = target_gradient == 0.0
        target_ok = alpha_valid & (
            target_decoupled
            | (torch.isfinite(target_denom) & (target_denom < 0.0))
        )
        projected = torch.zeros((count, width), dtype=w.dtype, device=w.device)
        projected[row_index, target] = torch.where(
            target_decoupled,
            torch.zeros_like(target_gradient),
            -target_gradient / target_denom,
        )

        if width > 1:
            comp_coupling = comp_gradient / root_alpha[:, None]
            lost_comp_coupling = (
                (comp_gradient != 0.0) & (comp_coupling == 0.0)
            ).any(1)
            comp_root, comp_root_ok = _extremal_arrowhead_root_batched(
                comp_curvature / alpha[:, None], comp_coupling, maximum=False
            )
            comp_denom = comp_curvature - alpha[:, None] * comp_root[:, None]
            comp_decoupled = comp_gradient == 0.0
            comp_denom_ok = comp_decoupled | (
                torch.isfinite(comp_denom) & (comp_denom > 0.0)
            )
            comp_step = torch.where(
                comp_decoupled,
                torch.zeros_like(comp_gradient),
                -comp_gradient / comp_denom,
            )
            projected.scatter_(1, complement, comp_step)
        step = torch.bmm(Vr, projected.unsqueeze(-1)).squeeze(-1)
        finite = target_ok & torch.isfinite(step).all(1)
        if width > 1:
            finite &= (
                comp_root_ok & ~lost_comp_coupling & comp_denom_ok.all(1)
            )
        return step, finite

    one = torch.ones(count, dtype=w.dtype, device=w.device)
    unrestricted, finite = step_for_alpha(one)
    unrestricted_norm = _safe_row_norm(unrestricted)
    tolerance = torch.minimum(
        1.0e-10 * torch.maximum(one, radius),
        0.5 * radius,
    )
    direct = finite & (unrestricted_norm <= radius)
    solved_steps = torch.zeros_like(unrestricted)
    solved_steps[direct] = unrestricted[direct]

    needs = finite & ~direct
    alpha_lo = one.clone()
    alpha_hi = torch.full_like(one, 2.0)
    restricted, restricted_finite = step_for_alpha(alpha_hi)
    restricted_norm = _safe_row_norm(restricted)
    needs &= restricted_finite
    for _ in range(max_bisect_it):
        expand = needs & (restricted_norm > radius + tolerance)
        if not bool(expand.any()):
            break
        alpha_lo = torch.where(expand, alpha_hi, alpha_lo)
        alpha_hi = torch.where(expand, 2.0 * alpha_hi, alpha_hi)
        candidate, candidate_finite = step_for_alpha(alpha_hi)
        restricted = torch.where(expand[:, None], candidate, restricted)
        restricted_norm = _safe_row_norm(restricted)
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
        norm = _safe_row_norm(candidate)
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
