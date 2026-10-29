"""Aggregate the declared v5 stage on server after all four runs terminate."""

import argparse
import json
from pathlib import Path
import time

from analysis import analyze, run_cells, result_name

HERE = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("stage_a", "stage_b"), default="stage_a")
    args = parser.parse_args()
    p = json.loads((HERE / "protocol.json").read_text())
    root = HERE / ("runs" if args.stage == "stage_a" else "frozen_runs")
    expected = [root / "control" / row["scenario"] / row["arm"] / f"seed{row['seed']}"
                for row in run_cells(p, args.stage)]
    started = time.monotonic()
    while True:
        complete = sum((path / "summary.json").exists() for path in expected)
        failed = sum((path / "failure.json").exists() for path in expected)
        terminal = sum((path / "summary.json").exists() or (path / "failure.json").exists() for path in expected)
        print("SCHEDULEURM_PROGRESS " + json.dumps(dict(current=terminal, total=len(expected), unit="runs",
              elapsed_seconds=time.monotonic() - started, complete=complete, failed=failed)), flush=True)
        if terminal == len(expected) or time.monotonic() - started >= 48 * 3600:
            result = analyze(root, p, stage=args.stage)
            path = HERE / result_name(args.stage)
            path.write_text(json.dumps(result, ensure_ascii=False, separators=(",", ":"), allow_nan=False) + "\n")
            print(json.dumps(dict(stage="summary_written", status=result.get("status"),
                  complete=complete, failed=failed, bytes=path.stat().st_size)), flush=True)
            return
        time.sleep(60)


if __name__ == "__main__":
    main()
