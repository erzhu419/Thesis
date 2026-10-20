"""Collect one bounded source-only same-trip physical interval calibration day."""

import json
import os
import time

from bootstrap import HERE, formal, load_learner, write
from stream import EventStream, create_source_env
from experiments.mcwm_closed_loop_formal_20261020.scenarios import (
    AFFECTED_LINES, is_direct_interval)
from experiments.mcwm_closed_loop_diagnostic_v2_20261020.interval import serialize


def main():
    p = json.loads((HERE/"protocol.json").read_text())
    output = HERE/"inputs/source_intervals.json"
    if output.exists() or (HERE/"inputs/source_summary.json").exists():
        raise FileExistsError("Source calibration already exists")
    started = time.monotonic()
    env = None
    try:
        learner, _, _, contract, stations, initializer = load_learner(
            p["source_calibration_policy_seed"])
        env = create_source_env(os.environ["THESIS_SOURCE_ENV"],
                                seed=p["source_calibration_seed"])
        stream = EventStream(env, stations, contract, domain="source")
        stream.reset()
        rows = {line: [] for line in AFFECTED_LINES}
        last = started
        while not stream.done:
            stream.step(lambda s: learner.act(s[None], deterministic=True)[0])
            for row in stream.release_intervals():
                if (is_direct_interval(row.line_id, row.station_id, row.next_station_id)
                        and len(rows[row.line_id]) < p["source_intervals_per_line_limit"]):
                    rows[row.line_id].append(serialize(row))
            if time.monotonic()-last >= 30:
                print("SCHEDULEURM_PROGRESS "+json.dumps(dict(current=stream.now,
                    total=24000, unit="sim_seconds", phase="source_intervals",
                    counts={k: len(v) for k, v in rows.items()})), flush=True)
                last = time.monotonic()
        counts = {key: len(value) for key, value in rows.items()}
        if min(counts.values()) < p["source_intervals_per_line_minimum"]:
            raise RuntimeError(f"Insufficient actual source direct intervals: {counts}")
        write(output, [row for line in AFFECTED_LINES for row in rows[line]])
        summary = dict(status="complete", counts_by_line=counts,
            source_seed=p["source_calibration_seed"], initializer=str(initializer),
            policy_updates=0, target_rows=0, source_time_seconds=stream.now,
            rows=sum(counts.values()), wall_seconds=time.monotonic()-started,
            semantics="same-trip decision-ready to next-stop decision-ready; holding plus motion/signals plus service")
        write(HERE/"inputs/source_summary.json", summary)
        print(json.dumps(summary), flush=True)
    except Exception as error:
        write(HERE/"inputs/source_failure.json", dict(status="failed",
              type=type(error).__name__, message=str(error)))
        raise
if __name__ == "__main__":
    main()
