"""One-off handoff of the already-running 300-episode jobs at checkpoint 99."""

import argparse
import json
from pathlib import Path
import shlex
import subprocess
import sys
import time

from deploy import HERE, PYTHON, SCHEDULER, spec, submit

sys.path.insert(0, str(Path(SCHEDULER).parent))
import scheduler


def inspect(node, jobs):
    identifiers = [job["id"] for job in jobs]
    code = (
        "import json,re; from pathlib import Path; result={}; "
        f"ids={identifiers!r}\n"
        "for task in ids:\n"
        " p=Path('/tmp/sched_'+task+'.log')\n"
        " text=p.read_text(errors='replace') if p.exists() else ''\n"
        " counts=re.findall(r'phase=train progress=(\\d+)/300',text)\n"
        " result[task]={'ready':'[Checkpoint ep 99]' in text,"
        "'episodes':int(counts[-1]) if counts else 0}\n"
        "print(json.dumps(result))"
    )
    status, output, error = scheduler.run_on(node, PYTHON + " -c " + shlex.quote(code),
                                            timeout=45, check=False)
    if status:
        raise RuntimeError(f"{node}: {error[-500:]}")
    return json.loads(output)


def main(check_only):
    jobs = json.loads((HERE / "production_submission.json").read_text())["submitted"]
    deadline = time.monotonic() + 3 * 3600
    while time.monotonic() < deadline:
        pending = [job for job in jobs if not
                   (HERE / f"cap100_eval_{job['id']}_submission.json").exists()]
        if not pending:
            print("All 20 endpoint-99 evaluation jobs submitted", flush=True)
            return
        for node in sorted({job["require_node"] for job in pending}):
            group = [job for job in pending if job["require_node"] == node]
            progress = inspect(node, group)
            print(node, json.dumps(progress), flush=True)
            if check_only:
                continue
            ready = [job for job in group if progress[job["id"]]["ready"]]
            if not ready:
                continue
            # The marker is printed only after both endpoint weights are saved.
            subprocess.run(["python3", SCHEDULER, "cancel-batch", "--force", "--confirm",
                            "--force-lock-wait",
                            *[job["id"] for job in ready]], check=True)
            for job in ready:
                label = job["signature"].split("/")[-1]
                arm, seed = label.split("_seed")
                task = spec(label, f"--evaluate-only --arm {arm} --seed {seed} --workers 32",
                            job["require_node"])
                task["signature"] = "thesis-feedback-100/eval/" + label
                task["description"] = "TransitDuet endpoint 100 parallel evaluation " + label
                submit([task], f"cap100_eval_{job['id']}", dispatch=False)
        if check_only:
            return
        time.sleep(15)
    raise TimeoutError("Endpoint handoff incomplete after three hours; inspect training logs")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--check-only", action="store_true")
    main(parser.parse_args().check_only)
