"""Recover retained execution and target-visit records without rerunning history."""

from review_common import HERE, EXPERIMENTS, write
from collections import Counter, defaultdict
import csv
import json
from pathlib import Path
import sys

ROOT = EXPERIMENTS.parent
H2O = ROOT.parent / "sumo-rl/H2Oplus"
sys.path.insert(0, str(ROOT.parent / "scheduleurm/skill"))
import scheduler


def interested(task):
    words = " ".join(str(task.get(k, "")) for k in ("project", "cwd", "signature", "cmd"))
    return ("Thesis" in words or ("H2Oplus" in words and ("transit" in words or "h2oplus_v" in words))) and "thesis-review-round3" not in words


def phase(task):
    words = (str(task.get("signature", ""))+" "+str(task.get("cmd", ""))).lower()
    if any(x in words for x in ("preflight", "smoke", "watch", "deploy", "verify", "profile")):
        return "运行检查与调度"
    if "collect" in words:
        return "日志或轨迹采集"
    if "calibrat" in words and "hold_calibration" not in words and "eval" not in words:
        return "源仿真标定"
    if "eval" in words or "formal_confirmation" in words or "attribution" in words:
        return "目标评价或混合评价任务"
    if "train" in words or "prefix" in words or "campaign" in words:
        return "学习训练或训练评价合并任务"
    return "其他已保留任务"


def recover_tasks():
    rows = {}
    archive = Path.home() / ".claude/scheduler/queue_archive.jsonl"
    if archive.exists():
        with archive.open() as stream:
            for line in stream:
                if "Thesis" not in line and "H2Oplus" not in line:
                    continue
                task = json.loads(line)
                if interested(task):
                    rows[task["id"]] = task
    for task in scheduler.load_state()["tasks"]:
        if interested(task):
            rows[task["id"]] = task
    output = []
    for task in rows.values():
        start, end = task.get("started_at"), task.get("finished_at")
        seconds = end-start if start and end and end >= start else None
        output.append(dict(task_id=task["id"], phase=phase(task), status=task["status"],
             node=task.get("node"), started_at=start, finished_at=end,
             recorded_task_elapsed_seconds=seconds, retry_count=task.get("retry_count", 0),
             signature=task.get("signature"), result_dir=task.get("result_dir"),
             log_path=task.get("log_path")))
    with (HERE / "historical_tasks.csv").open("w") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(output[0]))
        writer.writeheader()
        writer.writerows(output)
    groups = {}
    for key in sorted(set(r["phase"] for r in output)):
        selected = [r for r in output if r["phase"] == key]
        timed = [r["recorded_task_elapsed_seconds"] for r in selected if r["recorded_task_elapsed_seconds"] is not None]
        groups[key] = dict(tasks=len(selected), statuses=dict(Counter(r["status"] for r in selected)),
                           timed_attempts=len(timed), recorded_task_hours=sum(timed)/3600)
    return groups


def recover_target_visits():
    seen, records = set(), []
    roots = sorted((H2O / "reproducibility").glob("h2oplus*_results"))
    for root in roots:
        for path in sorted(root.rglob("sumo_seed*.json")):
            row = json.loads(path.read_text())
            if not isinstance(row, dict) or "episode" not in row or "method" not in row:
                continue
            episode = row["episode"]
            scenario = row.get("scenario", {})
            key = (row.get("created_utc", str(path)), str(row["method"]), row.get("sumo_seed"),
                   scenario.get("selection_seed"), scenario.get("demand_scale"))
            if key in seen:
                continue
            seen.add(key)
            records.append(dict(study=root.name, source=str(path.relative_to(H2O)),
                                method=row["method"], sumo_seed=row.get("sumo_seed"),
                                end_seconds=episode.get("observed_end_time_seconds"),
                                wall_seconds=row.get("runtime", {}).get("wall_time_seconds"),
                                limited=bool(episode.get("stopped_by_event_limit") or episode.get("stopped_by_time_limit"))))
    with (HERE / "historical_target_visits.csv").open("w") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)
    result = {}
    for key in sorted(set(r["study"] for r in records)):
        selected = [r for r in records if r["study"] == key]
        result[key] = dict(retained_episodes=len(selected), limited=sum(r["limited"] for r in selected),
            measured_simulated_hours=sum(r["end_seconds"] or 0 for r in selected)/3600,
            timed_episodes=sum(r["wall_seconds"] is not None for r in selected),
            measured_evaluation_hours=sum(r["wall_seconds"] or 0 for r in selected)/3600)
    return result


def main():
    result = dict(tasks=recover_tasks(), target_visits=recover_target_visits(),
       scope="Retained original bus H2O development/evaluation and thesis scheduler records before round3. Successful evaluation artifacts and execution attempts are separate ledgers, not additive costs.",
       missing=["Unscheduled and deleted early runs", "Per-episode progress inside failed jobs without retained result artifacts",
                "Exact CPU/GPU compute time: scheduler wall duration is not device utilization time",
                "Original field survey and data preparation time", "Unretained log-collection and source-calibration execution times"])
    write(HERE / "historical_cost_summary.json", result)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
