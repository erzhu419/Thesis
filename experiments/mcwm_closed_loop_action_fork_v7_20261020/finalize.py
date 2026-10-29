"""Aggregate 24 fixed branch records on server and retain explicit failure states."""

import json
from pathlib import Path
import time

from analysis import analyze, branch_cells, cell_path, result_name

HERE = Path(__file__).resolve().parent


def main():
    p = json.loads((HERE / "protocol.json").read_text())
    expected = [HERE / "runs" / cell_path(row) for row in branch_cells(p)]
    started = time.monotonic()
    while True:
        complete = sum((path / "summary.json").exists() for path in expected)
        failed = sum((path / "failure.json").exists() for path in expected)
        terminal = sum((path / "summary.json").exists() or (path / "failure.json").exists() for path in expected)
        print("SCHEDULEURM_PROGRESS " + json.dumps(dict(current=terminal, total=24, unit="branches",
              elapsed_seconds=time.monotonic() - started, complete=complete, failed=failed)), flush=True)
        if terminal == 24 or time.monotonic() - started >= 72 * 3600:
            value = analyze(HERE / "runs", p)
            path = HERE / result_name()
            path.write_text(json.dumps(value, ensure_ascii=False, separators=(",", ":"), allow_nan=False) + "\n")
            print(json.dumps(dict(stage="summary_written", status=value["status"], bytes=path.stat().st_size)), flush=True)
            return
        time.sleep(60)


if __name__ == "__main__":
    main()
