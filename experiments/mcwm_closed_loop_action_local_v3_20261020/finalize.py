"""Aggregate resident pilot logs; only the compact JSON is collected locally."""

import json
from pathlib import Path
import time

from analysis import analyze

HERE = Path(__file__).resolve().parent


def main():
    p = json.loads((HERE / "protocol.json").read_text())
    expected = [HERE / "runs/control" / scenario / arm / f"seed{seed}"
                for seed in p["training_seeds"] for scenario in p["scenarios"] for arm in p["arms"]]
    started = time.monotonic()
    while True:
        complete = sum((path / "summary.json").exists() for path in expected)
        failed = sum((path / "failure.json").exists() for path in expected)
        terminal = sum((path / "summary.json").exists() or (path / "failure.json").exists() for path in expected)
        print("SCHEDULEURM_PROGRESS " + json.dumps(dict(current=terminal, total=len(expected), unit="runs",
              elapsed_seconds=time.monotonic() - started, complete=complete, failed=failed)), flush=True)
        if terminal == len(expected) or time.monotonic() - started >= 48 * 3600:
            result = analyze(HERE / "runs")
            (HERE / "pilot_summary.json").write_text(json.dumps(result, ensure_ascii=False,
                    separators=(",", ":"), allow_nan=False) + "\n")
            print(json.dumps(dict(stage="pilot_summary_written", status=result.get("status"),
                  complete=complete, failed=failed)), flush=True)
            return
        time.sleep(60)


if __name__ == "__main__":
    main()
