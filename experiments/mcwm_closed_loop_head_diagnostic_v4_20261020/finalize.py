"""Aggregate resident diagnostics once, including the fixed post-update window."""

import json
from pathlib import Path
import time

from analysis import analyze, run_cells

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
        print("SCHEDULEURM_PROGRESS " + json.dumps(dict(current=terminal, total=len(expected), unit="runs",
              elapsed_seconds=time.monotonic() - started, complete=complete, failed=failed)), flush=True)
        if terminal == len(expected) or time.monotonic() - started >= 48 * 3600:
            result = analyze(HERE / "runs", p)
            (HERE / "head_diagnostic_summary.json").write_text(json.dumps(result, ensure_ascii=False,
                    separators=(",", ":"), allow_nan=False) + "\n")
            print(json.dumps(dict(stage="head_diagnostic_summary_written", status=result.get("status"),
                  complete=complete, failed=failed)), flush=True)
            return
        time.sleep(60)


if __name__ == "__main__":
    main()
