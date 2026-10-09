"""Correct this campaign's zero-exit jobs misclassified for silent logs."""

from review_common import HERE, P, write
import json
import shlex
import sys
import time

sys.path.insert(0, str(HERE.parents[2] / "scheduleurm/skill"))
import scheduler


def main():
    changes = []
    with scheduler.state_lock():
        state = scheduler.load_state()
        for task in state["tasks"]:
            if (not task.get("signature", "").startswith(P["name"])
                    or task["status"] != "failed" or task.get("exit_code") != 0):
                continue
            tokens = shlex.split(task["cmd"])
            if "--out" not in tokens:
                continue
            remote = tokens[tokens.index("--out")+1]
            folder = HERE / remote.split("/review_round3/", 1)[1]
            if "--case" in tokens and ("legacy.py" in task["cmd"] or "mcwm.py" in task["cmd"]):
                case = int(tokens[tokens.index("--case")+1])
                name = f"eval_{case:02d}.json" if "mcwm.py" in task["cmd"] else f"eval{case:02d}.json"
                result = folder / "results" / name
                if not result.exists() or json.loads(result.read_text())["horizon_seconds"] != 24000:
                    continue
            else:
                result = folder / "results/ready.json"
                if not result.exists():
                    continue
                row = json.loads(result.read_text())
                if "/guard/" in remote or "/response/" in remote:
                    if row["horizon"] != 24000:
                        continue
                if "/mcwm/" in remote and row["continuation_updates"] != 20000:
                    continue
            record = dict(id=task["id"], exit_code=0, result=str(result),
                          previous_diagnosis=task.get("last_oom_forensics", {}).get("reason"),
                          reconciled_at=time.time())
            task["status"] = "done"
            task["completion_reconciliation"] = record
            changes.append(record)
        scheduler.save_state(state)
    prior = HERE / "completion_reconciliation.json"
    records = json.loads(prior.read_text()) if prior.exists() else []
    write(prior, records+changes)
    print(json.dumps(dict(corrected=len(changes), ids=[r["id"] for r in changes])))


if __name__ == "__main__":
    main()
