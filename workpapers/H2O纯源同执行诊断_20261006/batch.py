"""Run the fixed paired cases, retaining each result and its small process log."""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
from pathlib import Path
import subprocess
import time


def run_case(command):
    output = Path(command[command.index("--output") + 1])
    if output.exists():
        return {"case": output.name, "status": "existing"}
    output.parent.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    with output.with_suffix(".log").open("w") as log:
        result = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT)
    return {"case": output.name, "returncode": result.returncode,
            "seconds": round(time.monotonic() - started, 2)}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cases", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=12)
    args = parser.parse_args()
    cases = json.loads(args.cases.read_text())
    failed = 0
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        pending = [pool.submit(run_case, command) for command in cases]
        for future in as_completed(pending):
            row = future.result()
            failed += row.get("returncode", 0) != 0
            print(json.dumps(row), flush=True)
    raise SystemExit(1 if failed else 0)
