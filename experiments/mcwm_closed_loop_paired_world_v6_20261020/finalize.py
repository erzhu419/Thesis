"""Aggregate the four diagnostic days on server; collect no trajectories."""

import json
from pathlib import Path
import time

from analysis import analyze, result_name, run_cells

HERE = Path(__file__).resolve().parent


def main():
    p = json.loads((HERE / "protocol.json").read_text())
    expected = [HERE / "runs/control" / row["scenario"] / row["arm"] / f"seed{row['seed']}"
                for row in run_cells(p)]
    started = time.monotonic()
    while True:
        complete = sum((path / "summary.json").exists() for path in expected)
        failed = sum((path / "failure.json").exists() for path in expected)
        terminal = sum((path / "summary.json").exists() or (path / "failure.json").exists() for path in expected)
        print("SCHEDULEURM_PROGRESS " + json.dumps(dict(current=terminal, total=4, unit="runs",
              elapsed_seconds=time.monotonic() - started, complete=complete, failed=failed)), flush=True)
        if terminal == 4 or time.monotonic() - started >= 48 * 3600:
            value = analyze(HERE / "runs", p)
            path = HERE / result_name()
            path.write_text(json.dumps(value, ensure_ascii=False, separators=(",", ":"), allow_nan=False) + "\n")
            print(json.dumps(dict(stage="summary_written", status=value["status"],
                                  bytes=path.stat().st_size)), flush=True)
            return
        time.sleep(60)


if __name__ == "__main__":
    main()
