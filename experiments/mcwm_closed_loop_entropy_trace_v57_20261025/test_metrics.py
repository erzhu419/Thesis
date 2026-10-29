"""Known level and exposure effects must not masquerade as action evidence."""
import json

import numpy as np

from experiments.mcwm_closed_loop_entropy_trace_v57_20261025.metrics import summarize


def fixture():
    rows = [dict(line_id=('102X', '705X', '7X')[i % 3], dt=0. if i % 4 == 0 else 2.,
                 started_at=float(i*2), completed_at=float(i*2+2)) for i in range(271)]
    native = np.stack((np.arange(271), np.arange(271)), axis=0).astype(float)
    mass = np.linspace(1., 0., 271)
    return rows, native, mass


def test_constant_level_shift_does_not_change_factual_state_differences():
    rows, native, mass = fixture()
    beta0 = native+2.
    beta1 = beta0-3.
    beta1[:, -1] = beta0[:, -1]
    result = summarize(native, beta0, beta1, rows, mass)
    allrows = result['groups']['all']
    assert allrows['shift_across_factual_roots']['mean'] == -3.
    assert allrows['shift_after_global_centering_rms'] == 0.
    assert allrows['shift_after_line_centering_rms'] == 0.
    assert allrows['adjacent_same_line_factual_roots']['difference_change']['max'] == 0.
    assert allrows['adjacent_same_line_factual_roots']['sign_changes'] == 0
    assert allrows['targets_across_factual_roots']['beta0']['std'] == allrows['targets_across_factual_roots']['beta1']['std']
    assert result['supported_roots'] == 270
    assert result['excluded_no_successor']['shift_max_abs'] == 0.


def test_line_offsets_disappear_under_line_centering():
    rows, native, mass = fixture()
    offsets = np.asarray([-float(i % 3 + 1) for i in range(271)])
    result = summarize(native, native, native+offsets[None, :], rows, mass)
    stats = result['groups']['all']
    assert stats['shift_after_global_centering_rms'] > 0.
    assert stats['shift_after_line_centering_rms'] == 0.
    assert stats['adjacent_same_line_factual_roots']['difference_change']['max'] == 0.


def test_exposure_effect_separated_from_unexplained_state_variation():
    rows, native, mass = fixture()
    shift = -4.*mass
    result = summarize(native, native, native+shift[None, :], rows, mass)
    fit = result['groups']['all']['exposure_fit']
    assert fit['rank'] == 2
    assert abs(fit['mass_slope']+4.) < 1e-12
    assert fit['residual_std'] < 1e-12
    assert abs(fit['variance_explained']-1.) < 1e-12
    assert result['groups']['all']['shift_after_global_centering_rms'] > 0.


def test_blocks_separate_sampling_variance_and_emit_small_finite_summary():
    rows, native, mass = fixture()
    shift = np.stack((-mass-1., -mass+1.))
    result = summarize(native, native, native+shift, rows, mass)
    assert abs(result['groups']['all']['block_sampling_sd_rms']-1.) < 1e-12
    assert sum(v['rows'] for v in result['duration_groups'].values()) == 270
    assert sum(v['rows'] for v in result['remaining_span_groups'].values()) == 270
    assert sum(v['rows'] for v in result['future_mass_groups'].values()) == 270
    assert len(json.dumps(result, allow_nan=False, separators=(',', ':'))) < 12000


def test_missing_focus_line_is_explicit_not_zero_effect():
    rows, native, mass = fixture()
    result = summarize(native, native, native, rows, mass, focus_lines=('unknown',))
    assert result['groups']['unknown']['rows'] == 0
    assert result['groups']['unknown']['shift_across_factual_roots']['mean'] is None
