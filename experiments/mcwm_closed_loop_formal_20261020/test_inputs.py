"""Tests for real tails, sampling mass, and bounded state reads."""

import json

import h5py
import numpy as np
import pytest

from inputs import load_history


@pytest.fixture
def history_path(tmp_path):
    path = tmp_path / "history.h5"
    count = 120
    with h5py.File(path, "w") as handle:
        handle.attrs["schema_version"] = "transit-smdp-v7"
        handle.attrs["station_max_by_line"] = json.dumps([24.0])
        handle.attrs["config"] = json.dumps(dict(horizon_seconds=24000))
        states = np.arange(count * 3, dtype=np.float32).reshape(count, 3)
        terminal = np.zeros(count, np.uint8)
        terminal[[29, 59, 89, 119]] = 1
        rewards = -np.arange(count, dtype=np.float64)
        rewards[terminal.astype(bool)] = -900000.0
        rewards[3] = np.nan  # Missing values are retained, never replaced with zero.
        arrays = dict(observations=states, next_observations=states + 1,
                      actions=np.zeros((count, 1), np.float32), rewards=rewards,
                      delta_t_seconds=np.where(np.arange(count) % 3, 1.0, 0.0),
                      terminals=terminal, demand_realization_seed=np.repeat([1201, 1202], 60),
                      trajectory_id=np.repeat(np.arange(4), 30))
        for key, array in arrays.items():
            handle.create_dataset(key, data=array, chunks=(20, *array.shape[1:]))
    return path


def test_terminal_tails_and_original_probability_mass(history_path):
    table, metadata = load_history(history_path, max_rows=24, seed=55)
    tail = table["terminal"].astype(bool)
    assert tail.sum() == 4
    assert np.all(table["r"][tail] == -900000.0)
    assert table["sample_weight"][tail].sum() == pytest.approx(4 / 120)
    assert table["sample_weight"].sum() == pytest.approx(1.0)
    assert metadata["strata"]["ordinary"]["inclusion_probability"] == pytest.approx(20 / 116)
    assert metadata["selected_row_indices"] == sorted(metadata["selected_row_indices"])


def test_block_restriction_and_determinism(history_path):
    first, metadata = load_history(history_path, max_rows=20, seed=77, blocks=[1202])
    second, repeated = load_history(history_path, max_rows=20, seed=77, blocks=[1202])
    _, different = load_history(history_path, max_rows=20, seed=78, blocks=[1202])
    assert metadata == repeated
    np.testing.assert_array_equal(first["s"], second["s"])
    assert min(metadata["selected_row_indices"]) >= 60
    assert metadata["blocks"] == [1202]
    assert set(metadata["by_trajectory"]) == {"2", "3"}
    assert first["terminal"].sum() == 2
    assert metadata["selected_row_indices"] != different["selected_row_indices"]
    with pytest.raises(ValueError, match="absent"):
        load_history(history_path, max_rows=20, blocks=[1203])


def test_state_reads_are_selected_bounded_rows(history_path, monkeypatch):
    original = h5py.Dataset.__getitem__
    selected_sizes = []

    def tracked(dataset, selection):
        if dataset.name in ("/observations", "/next_observations"):
            assert isinstance(selection, np.ndarray)
            assert len(selection) <= 18
            selected_sizes.append(len(selection))
        return original(dataset, selection)

    monkeypatch.setattr(h5py.Dataset, "__getitem__", tracked)
    table, _ = load_history(history_path, max_rows=18, seed=2)
    assert len(table["s"]) == 18
    assert sum(selected_sizes) == 36


def test_full_small_table_preserves_zero_duration_and_missing_value(history_path):
    table, metadata = load_history(history_path, max_rows=120)
    assert np.isnan(table["r"][3])
    assert metadata["zero_duration_rows"] == 40
    assert (table["dt"] == 0).sum() == 40
    np.testing.assert_allclose(table["sample_weight"], np.full(120, 1 / 120))
