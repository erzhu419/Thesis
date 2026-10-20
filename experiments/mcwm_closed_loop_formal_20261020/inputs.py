"""Bounded canonical-v7 history, retaining real horizon tails and their mass."""

import json
from pathlib import Path

import h5py
import numpy as np


H5_KEYS = dict(s="observations", a="actions", ns="next_observations",
               r="rewards", dt="delta_t_seconds", terminal="terminals")
SCAN_ROWS = 65536


def _read_rows(dataset, rows):
    """Select states by storage chunk; never materialize the full state table."""
    output = np.empty((len(rows), *dataset.shape[1:]), dtype=dataset.dtype)
    chunk = dataset.chunks[0] if dataset.chunks else SCAN_ROWS
    boundaries = np.r_[0, np.flatnonzero(np.diff(rows // chunk)) + 1, len(rows)]
    for start, stop in zip(boundaries[:-1], boundaries[1:]):
        output[start:stop] = dataset[rows[start:stop]]
    return output


def load_history(path, max_rows=32768, seed=6201, blocks=None):
    """Read at most ``max_rows`` canonical rows from a local/server-shared HDF5.

    All permitted terminal rows are retained. Ordinary rows are a uniform
    without-replacement sample, selected by streaming random priorities.
    ``sample_weight`` gives each stratum its original probability mass, so a
    weighted replay draw does not inflate the frequency of rare terminal tails.
    Rows retain source order (trajectory order, then time within trajectory),
    rather than claiming one global timeline across independent trajectories.
    The caller must keep the weights separate before concatenating new rows.
    """
    path = Path(path).expanduser().resolve()
    if max_rows < 2:
        raise ValueError("At least two historical rows are required")
    allowed = None if blocks is None else np.asarray(sorted(set(blocks)), dtype=np.int64)
    rng = np.random.default_rng(seed)
    candidates = np.empty(0, dtype=np.int64)
    priorities = np.empty(0, dtype=np.float64)
    tails = []
    by_block, by_trajectory = {}, {}
    population = ordinary_count = 0

    with h5py.File(path, "r") as handle:
        if handle.attrs.get("schema_version") != "transit-smdp-v7":
            raise ValueError("History must use the canonical transit-smdp-v7 schema")
        total_rows = len(handle["terminals"])
        for start in range(0, total_rows, SCAN_ROWS):
            stop = min(start + SCAN_ROWS, total_rows)
            block = handle["demand_realization_seed"][start:stop].reshape(-1)
            permitted = np.ones(len(block), dtype=bool) if allowed is None else np.isin(block, allowed)
            if not permitted.any():
                continue
            terminal = handle["terminals"][start:stop].reshape(-1).astype(bool)
            trajectory = handle["trajectory_id"][start:stop].reshape(-1)
            rows = np.arange(start, stop, dtype=np.int64)
            tails.extend(rows[permitted & terminal].tolist())
            population += int(permitted.sum())
            for values, registry in ((block, by_block), (trajectory, by_trajectory)):
                identities, counts = np.unique(values[permitted], return_counts=True)
                for identity, count in zip(identities, counts):
                    record = registry.setdefault(str(int(identity)), dict(source_rows=0, source_terminal_rows=0))
                    record["source_rows"] += int(count)
                identities, counts = np.unique(values[permitted & terminal], return_counts=True)
                for identity, count in zip(identities, counts):
                    registry[str(int(identity))]["source_terminal_rows"] += int(count)
            ordinary = rows[permitted & ~terminal]
            ordinary_count += len(ordinary)
            candidates = np.concatenate((candidates, ordinary))
            priorities = np.concatenate((priorities, rng.random(len(ordinary))))
            if len(candidates) > max_rows:
                keep = np.argpartition(priorities, max_rows - 1)[:max_rows]
                candidates, priorities = candidates[keep], priorities[keep]

        missing = [] if allowed is None else sorted(set(map(int, allowed)) - set(map(int, by_block)))
        if missing:
            raise ValueError(f"Requested history blocks are absent: {missing}")
        if population < 2:
            raise ValueError("The permitted blocks contain fewer than two history rows")
        terminal_count = len(tails)
        limit = min(int(max_rows), population)
        if terminal_count > limit or (ordinary_count and terminal_count == limit):
            raise ValueError("History budget must retain every terminal and at least one ordinary row")
        ordinary_limit = limit - terminal_count
        if len(candidates) > ordinary_limit:
            keep = np.argpartition(priorities, ordinary_limit - 1)[:ordinary_limit]
            candidates = candidates[keep]
        rows = np.sort(np.concatenate((np.asarray(tails, dtype=np.int64), candidates)))
        table = {key: _read_rows(handle[name], rows) for key, name in H5_KEYS.items()}
        selected_terminal = table["terminal"].reshape(-1).astype(bool)
        weight = np.full(len(rows), ordinary_count / population / max(ordinary_limit, 1), dtype=np.float64)
        weight[selected_terminal] = 1.0 / population
        table["sample_weight"] = weight
        selected_blocks = _read_rows(handle["demand_realization_seed"], rows).reshape(-1)
        selected_trajectories = _read_rows(handle["trajectory_id"], rows).reshape(-1)
        for values, registry in ((selected_blocks, by_block), (selected_trajectories, by_trajectory)):
            for identity, record in registry.items():
                mask = values == int(identity)
                record["selected_rows"] = int(mask.sum())
                record["selected_terminal_rows"] = int((mask & selected_terminal).sum())
        metadata = dict(
            path=str(path), schema="transit-smdp-v7", source_rows=total_rows,
            permitted_rows=population, selected_rows=len(rows), max_rows=int(max_rows), seed=int(seed),
            blocks=sorted(map(int, by_block)), by_block=by_block, by_trajectory=by_trajectory,
            selected_row_indices=rows.tolist(),
            selected_trajectory_ids=selected_trajectories.tolist(),
            row_order="original HDF5 row order; independent trajectories, time ordered within trajectory",
            selection="all permitted terminals plus uniform ordinary rows without replacement",
            strata=dict(
                terminal=dict(population=terminal_count, selected=terminal_count,
                              inclusion_probability=1.0 if terminal_count else None,
                              replay_mass=terminal_count / population),
                ordinary=dict(population=ordinary_count, selected=ordinary_limit,
                              inclusion_probability=ordinary_limit / ordinary_count if ordinary_count else None,
                              replay_mass=ordinary_count / population)),
            terminal_rows=int(selected_terminal.sum()),
            zero_duration_rows=int((table["dt"].reshape(-1) == 0).sum()),
            retained_array_bytes=int(sum(value.nbytes for value in table.values())),
            station_max_by_line=json.loads(handle.attrs["station_max_by_line"]),
            contract=json.loads(handle.attrs["config"]),
            split_policy="caller splits retained learned rows in source order; no trajectory-disjoint split is implied",
        )
    return table, metadata
