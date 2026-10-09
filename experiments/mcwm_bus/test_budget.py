from types import SimpleNamespace

import h5py
import numpy as np

from run_budget import block_view, target_splits
from source_mismatch import travel_steps


def test_virtual_budget_contains_only_complete_selected_blocks(tmp_path):
    source = tmp_path / "source.h5"
    view = tmp_path / "view.h5"
    with h5py.File(source, "w") as handle:
        handle.create_dataset("demand_realization_seed", data=[1, 1, 2, 2, 3, 3])
        handle.create_dataset("observations", data=np.arange(12).reshape(6, 2))
        handle.attrs["schema_version"] = "test"
    assert block_view(source, view, [1, 3]) == 4
    with h5py.File(view, "r") as handle:
        assert handle["observations"].is_virtual
        np.testing.assert_array_equal(handle["observations"][:], [[0, 1], [2, 3], [8, 9], [10, 11]])
        np.testing.assert_array_equal(handle["demand_realization_seed"][:], [1, 1, 3, 3])


def test_model_splits_stay_inside_the_offline_budget():
    replay = SimpleNamespace(
        delta_t_seconds=np.ones(12), terminals=np.tile([0., 1.], 6),
        demand_realization_seeds=np.repeat(np.arange(1201, 1207), 2),
        observations=np.zeros((12, 229)), actions=np.zeros((12, 1)),
        next_observations=np.zeros((12, 229)), rewards=-np.ones(12))
    p = {"model_max_fit_rows": 20, "model_max_validation_rows": 20}
    low = target_splits(replay, [1201, 1202], p)
    assert set(low["fit"]["block"]) == {1201}
    assert set(low["stop"]["block"]) == set(low["calibrate"]["block"]) == {1202}
    high = target_splits(replay, list(range(1201, 1207)), p)
    assert set(high["fit"]["block"]) == {1201, 1202, 1203, 1204}
    assert set(high["stop"]["block"]) == {1205}
    assert set(high["calibrate"]["block"]) == {1206}


def test_mean_speed_bias_changes_physical_travel_time():
    assert travel_steps(104, 4, 1., 0, .4, 90) == 100
    assert travel_steps(104, 4, .5, 0, .4, 90) == 200
