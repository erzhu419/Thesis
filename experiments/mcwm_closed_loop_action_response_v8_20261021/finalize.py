"""Aggregate the 12 resident prefix records and preserve failed/missing states."""

import json
from itertools import product
from pathlib import Path
import time

from analysis import analyze

HERE = Path(__file__).resolve().parent


def main():
    p = json.loads((HERE / "protocol.json").read_text())
    expected = [HERE / "runs" / scenario / line / f"seed{seed}"
                for seed, scenario, line in product(p["training_seeds"], p["scenarios"], p["target_lines"])]
    started = time.monotonic()
    while True:
        complete = sum((path / "summary.json").exists() for path in expected)
        failed = sum((path / "failure.json").exists() for path in expected)
        terminal = sum((path / "summary.json").exists() or (path / "failure.json").exists() for path in expected)
        print("SCHEDULEURM_PROGRESS " + json.dumps(dict(current=terminal, total=len(expected), unit="states",
              elapsed_seconds=time.monotonic() - started, complete=complete, failed=failed)), flush=True)
        if terminal == len(expected) or time.monotonic() - started >= 72 * 3600:
            value = analyze(HERE / "runs")
            path = HERE / "response_summary.json"
            path.write_text(json.dumps(value, ensure_ascii=False, separators=(",", ":"), allow_nan=False) + "\n")
            print(json.dumps(dict(stage="summary_written", status=value["status"], bytes=path.stat().st_size)), flush=True)
            return
        time.sleep(60)


if __name__ == "__main__":
    main()
