"""Summarise server-side training logs without transferring replay or weights."""

import argparse
import json
from pathlib import Path
import shlex
from statistics import mean
import subprocess
import sys


def historical_summary():
    root = Path("/home/erzhu419/mine_code/TransitDuet/transit_duet/logs")
    seeds = [42, 123, 456, 789, 1001, 1002, 1003, 1004, 1005, 1006]
    windows = ((80, 100), (130, 150), (180, 200), (230, 250), (280, 300))
    output = {}
    for seed in seeds:
        rows = json.loads((root / f"H_timetable_v5_stable_300_seed{seed}/diagnostics.json").read_text())
        assert [r["ep"] for r in rows] == list(range(300))
        for row in rows:
            row["composite"] = (row["avg_wait_min"] / 10
                                + row["fleet_overshoot"] ** 2 / row["N_fleet"]
                                + row["headway_cv"])
        output[seed] = {f"{a+1}-{b}": {
            key: mean(r[key] for r in rows[a:b]) for key in
            ("composite", "avg_wait_min", "lower_action_mean", "upper_delta_mean", "upper_reward_mean")}
            for a, b in windows}
    return dict(source=str(root), interpretation="historical stochastic training, not a matched endpoint comparison",
                seed_windows=output)


def diagnose(root):
    import numpy as np

    output = {}
    for path in sorted((root / "runs").glob("*/training.jsonl")):
        rows = [json.loads(line) for line in path.read_text().splitlines()][:100]
        assert [row["ep"] for row in rows] == list(range(100))
        for row in rows:
            row["composite"] = (row["avg_wait_min"] / 10
                                + row["fleet_overshoot"] ** 2 / row["N_fleet"]
                                + row["headway_cv"])
        keys = ("composite", "avg_wait_min", "avg_total_passenger_time_min",
                "headway_cv", "upper_delta_mean", "lower_action_mean",
                "upper_reward_mean", "upper_q_loss", "lower_q_loss",
                "upper_alpha", "lower_alpha", "N_fleet")
        windows = [{key: mean(row[key] for row in rows[start:start + 20]) for key in keys}
                   for start in range(0, 100, 20)]
        # Descriptive slopes only: stochastic training returns are not held-out evaluations.
        tail = rows[60:100]
        fleets = sorted({r["N_fleet"] for r in tail})
        design = np.array([[1, (r["ep"] - 60) / 20,
                            *[int(r["N_fleet"] == fleet) for fleet in fleets[1:]]]
                           for r in tail])
        slopes = {key: float(np.linalg.lstsq(design, [r[key] for r in tail], rcond=None)[0][1])
                  for key in ("composite", "avg_wait_min", "upper_reward_mean")}
        output[path.parent.name] = dict(
            windows_20_episodes=windows, tail_fleet_adjusted_change_per_20_episodes=slopes,
            last20_seconds_per_episode=mean(r["wall_env_s"] + r["wall_train_s"] for r in rows[80:]),
            checkpoint_files=sorted(p.name for p in (path.parent / "checkpoints").glob("*.pt")),
            last_replay_sizes={key: rows[-1][key] for key in ("upper_replay_size", "lower_replay_size")})
    assert len(output) == 20
    return output


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--server", action="store_true")
    args = parser.parse_args()
    if args.server:
        print(json.dumps(diagnose(Path(__file__).resolve().parent)))
    else:
        from deploy import HERE, REMOTE, PYTHON, SCHEDULER

        sys.path.insert(0, str(Path(SCHEDULER).parent))
        import scheduler

        node = "node001"
        subprocess.run(["rsync", "-a", "-e", scheduler._ssh_rsync_shell_for_node(node),
                        str(Path(__file__).resolve()),
                        f"{scheduler._ssh_target_for_node(node)}:{REMOTE}/"], check=True)
        code, stdout, stderr = scheduler.run_on(
            node, f"{PYTHON} {shlex.quote(REMOTE + '/training_diagnostics.py')} --server",
            timeout=60, check=False)
        if code:
            raise RuntimeError(stderr[-2000:])
        result = json.loads(stdout)
        (HERE / "training_diagnostics.json").write_text(json.dumps(result, indent=2) + "\n")
        (HERE / "historical_training_diagnostics.json").write_text(
            json.dumps(historical_summary(), indent=2) + "\n")
        for arm in ("full", "masked"):
            runs = [v for k, v in result.items() if k.startswith(arm + "_")]
            print(arm)
            for key in ("composite", "avg_wait_min", "upper_reward_mean", "upper_delta_mean",
                        "lower_action_mean", "upper_q_loss", "lower_q_loss"):
                print(key, [round(mean(r["windows_20_episodes"][i][key] for r in runs), 5)
                            for i in range(5)])
            for key in ("composite", "avg_wait_min", "upper_reward_mean"):
                values = [r["tail_fleet_adjusted_change_per_20_episodes"][key] for r in runs]
                print("tail_adjusted", key, "mean", round(mean(values), 5),
                      "negative_seeds", sum(v < 0 for v in values), "of", len(values))
