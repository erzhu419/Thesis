"""Execute every parity branch with untrained small native networks; no steps."""
import pytest

from experiments.mcwm_closed_loop_context_wiring_v80_20261027 import probe
from experiments.mcwm_closed_loop_context_wiring_v80_20261027.test_learning import initial
from experiments.mcwm_closed_loop_update_attribution_v27_20261023.shadow import _equal, _rng


@pytest.mark.parametrize('arm', ['fresh', 'no_fresh'])
def test_archived_fixture_probe_reaches_all_zero_update_paths_and_preserves_outer_rng(arm, monkeypatch):
    def load(seed, *, arm):
        learner = initial(arm)
        return learner, learner.replay_buffer, None, None, 'untrained_local_unit_fixture'
    monkeypatch.setattr(probe.learning.native, 'load_learner', load)
    before = _rng()
    result = probe.run_parity(6208, arm)
    # This fixture has0 initialization updates; it must never masquerade as50000.
    assert [key for key, value in result['checks'].items() if not value] == ['initializer_50000']
    assert _equal(before, _rng())
    assert result['paid_training_updates'] == result['native_simulated_seconds'] == 0
    assert result['fixture']['new_target_feedback'] == 0
    assert result['fixture']['v79_diagnostic_data_used'] is False
