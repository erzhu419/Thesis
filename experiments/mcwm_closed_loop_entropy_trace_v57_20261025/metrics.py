"""Compact factual-root diagnostics for frozen future-trace entropy removal.

Different roots are different observed states, never a paired action contrast.
Beta removes an already observed entropy term and supplies no new information.
"""
from __future__ import annotations

import numpy as np


def _get(row, name):
    return row[name] if isinstance(row, dict) else getattr(row, name)


def _stats(values):
    a = np.asarray(values, dtype=np.float64)
    if not a.size:
        return dict(n=0, mean=None, std=None, min=None, max=None)
    return dict(n=int(a.size), mean=float(a.mean()), std=float(a.std()),
                min=float(a.min()), max=float(a.max()))


def _mass_fit(shift, mass):
    """Remove exposure/chain-position variation before describing residuals."""
    if not len(shift):
        return dict(rank=0, intercept=None, mass_slope=None, residual_std=None,
                    variance_explained=None)
    x = np.column_stack((np.ones(len(mass)), mass))
    coefficients, _, rank, _ = np.linalg.lstsq(x, shift, rcond=None)
    residual = shift - x @ coefficients
    variance = float(np.var(shift))
    return dict(rank=int(rank), intercept=float(coefficients[0]) if rank == 2 else float(shift.mean()),
                mass_slope=float(coefficients[1]) if rank == 2 else None,
                residual_std=float(residual.std()),
                variance_explained=(float(1.-np.var(residual)/variance) if variance > 0. else None))


def _group(native, beta0, beta1, mass, lines, indices):
    selected = np.asarray(indices, dtype=int)
    shift = beta1[:, selected]-beta0[:, selected]
    mean_shift = shift.mean(0)
    n0, t0, t1 = (a[:, selected].mean(0) for a in (native, beta0, beta1))
    byline_centered = mean_shift.copy()
    for line in set(lines[selected]):
        mask = lines[selected] == line
        byline_centered[mask] -= mean_shift[mask].mean()
    pairs = []
    for line in sorted(set(lines[selected])):
        local = np.flatnonzero(lines[selected] == line)
        pairs.extend(zip(local[:-1], local[1:]))
    left = np.asarray([p[0] for p in pairs], dtype=int)
    right = np.asarray([p[1] for p in pairs], dtype=int)
    d0, d1 = t0[right]-t0[left], t1[right]-t1[left]
    changes = d1-d0
    return dict(rows=len(indices), shift_across_factual_roots=_stats(mean_shift),
        block_sampling_sd_rms=(float(np.sqrt(np.mean(np.var(shift, axis=0)))) if len(indices) else None),
        shift_after_global_centering_rms=(float(np.sqrt(np.mean((mean_shift-mean_shift.mean())**2))) if len(indices) else None),
        shift_after_line_centering_rms=(float(np.sqrt(np.mean(byline_centered**2))) if len(indices) else None),
        targets_across_factual_roots=dict(native=_stats(n0),beta0=_stats(t0),beta1=_stats(t1)),
        trace_minus_root=dict(beta0=_stats(t0-n0), beta1=_stats(t1-n0)),
        future_mass=_stats(mass[selected]), exposure_fit=_mass_fit(mean_shift, mass[selected]),
        adjacent_same_line_factual_roots=dict(pairs=len(pairs),
            beta0_difference=_stats(d0), beta1_difference=_stats(d1),
            difference_change=_stats(changes),
            sign_changes=int(np.count_nonzero(np.sign(d0) != np.sign(d1))),
            beta0_zero_differences=int(np.count_nonzero(d0 == 0.)),
            beta1_zero_differences=int(np.count_nonzero(d1 == 0.))))


def summarize(native_root, beta0_target, beta1_target, chainrows, future_mass,
              reach_seconds=None, *, focus_lines=('102X', '705X', '7X')):
    """Summarize blocks x 271 targets using the actual 270 supported roots.

    ``future_mass`` is eta times the sum of legal future continuation weights.
    ``reach_seconds`` describes the available real-chain span, not a fabricated
    fixed-300-second state. If omitted it is last completion minus root issue.
    All shifts and same-line root differences are descriptive, not action
    advantages, policy correctness or a test against held-out native outcomes.
    """
    native, beta0, beta1 = (np.asarray(a, dtype=np.float64)
                            for a in (native_root, beta0_target, beta1_target))
    if (native.ndim != 2 or native.shape != beta0.shape or native.shape != beta1.shape
            or native.shape[1] != len(chainrows) or len(chainrows) != 271):
        raise ValueError('Expected matching draw-block x 271 factual-root targets')
    mass = np.asarray(future_mass, dtype=np.float64)
    if mass.shape == native.shape:
        mass = mass.mean(0)
    if mass.shape != (271,):
        raise ValueError('Future mass must align with the 271 real roots')
    lines = np.asarray([str(_get(row, 'line_id')) for row in chainrows])
    durations = np.asarray([float(_get(row, 'dt')) for row in chainrows])
    if reach_seconds is None:
        end = float(_get(chainrows[-1], 'completed_at'))
        reach = np.asarray([end-float(_get(row, 'started_at')) for row in chainrows])
    else:
        reach = np.asarray(reach_seconds, dtype=np.float64)
    if reach.shape != (271,):
        raise ValueError('Real remaining spans must align with the 271 roots')
    supported = np.arange(270)
    groups = {'all': supported.tolist()}
    groups.update({line: supported[lines[:270] == line].tolist() for line in focus_lines})
    shifts = (beta1-beta0).mean(0)

    def small(indices):
        selected = np.asarray(indices, dtype=int)
        return dict(rows=len(indices), shift=_stats(shifts[selected]),
                    remaining_real_span_seconds=_stats(reach[selected]),
                    future_mass=_stats(mass[selected]))

    duration_groups = {name: small(supported[mask].tolist()) for name, mask in
        [('zero_duration', durations[:270] == 0.), ('positive_duration', durations[:270] > 0.)]}
    span_groups = {name: small(supported[mask].tolist()) for name, mask in
        [('zero_span', reach[:270] == 0.),
         ('positive_below300', (reach[:270] > 0.) & (reach[:270] < 300.)),
         ('at_least300', reach[:270] >= 300.)]}
    mass_groups = {name: small(supported[mask].tolist()) for name, mask in
        [('zero_future_mass', mass[:270] == 0.), ('positive_future_mass', mass[:270] > 0.)]}
    return dict(scope=('Actual factual-root variation only: adjacent same-line roots are '
                       'not same-state counterfactual actions. Beta removes known entropy; '
                       'nonconstant shifts are not new information or proven control improvement.'),
        draw_blocks=int(native.shape[0]), chain_rows=271, supported_roots=270,
        excluded_no_successor=dict(index=270, line=str(lines[-1]),
                                  shift_max_abs=float(np.abs(beta1[:, -1]-beta0[:, -1]).max())),
        coverage={line:dict(chain_rows=int(np.count_nonzero(lines == line)),
                            supported_roots=int(np.count_nonzero(lines[:270] == line)))
                  for line in sorted(set(lines))},
        groups={name:_group(native, beta0, beta1, mass, lines, indices)
                for name, indices in groups.items()},
        duration_groups=duration_groups, remaining_span_groups=span_groups,
        future_mass_groups=mass_groups)
