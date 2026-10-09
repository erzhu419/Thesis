"""Read completed server-side records; no new rollouts or checkpoint transfer."""

from collections import defaultdict
import json
from pathlib import Path
from statistics import mean

HERE = Path(__file__).resolve().parent
P = json.loads((HERE / "protocol.json").read_text())


def inspect():
    grouped = defaultdict(list)
    roots = {"no_norm": HERE, "norm_001": HERE.parent / P["reference_campaign"]}
    training = {}
    for label, root in roots.items():
        rows = [json.loads(p.read_text()) for p in (root / "results/global_event").glob("seed*/day*.json")]
        expected = {(seed, day, scene) for seed in P["training_seeds"]
                    for day in P["evaluation_seeds"] for scene in P["scenes"]}
        assert len(rows) == len(expected)
        assert {(r["seed"], r["evaluation_seed"], r["scene"]) for r in rows} == expected
        for row in rows:
            assert row["timestamp_cost_identity_passed"]
            grouped[label, row["seed"], row["scene"]].append(row)
        for seed in P["training_seeds"]:
            folder = root / "runs/global_event" / f"seed{seed}"
            history = json.loads((folder / "train_summary.json").read_text())["history"]
            config = json.loads((folder / "settings.json").read_text())["config"]
            assert len(history) == 100 and all(r["credit"]["terminal_transitions"] == 1 for r in history)
            assert all(r["updates"] == 400 and r["credit"]["cost_identity_passed"] for r in history)
            # Each episode appends exactly one terminal transition, last in the episode.
            offset, terminal_count = 0, 0
            for row in reversed(history):
                if offset >= config["buffer_size"]:
                    break
                terminal_count += 1
                offset += row["transitions"]
            replay_size = min(config["buffer_size"], sum(r["transitions"] for r in history))
            terminal_fraction = terminal_count / replay_size
            last = history[-10:]
            training[f"{label}/seed{seed}"] = dict(
                final_replay_size=replay_size, final_replay_terminal_transitions=terminal_count,
                terminal_fraction=terminal_fraction,
                approximate_batch_without_terminal_probability=(1-terminal_fraction)**config["batch_size"],
                last_ten_transitions=mean(r["transitions"] for r in last),
                last_ten_zero_time_fraction=mean(r["credit"]["zero_time_transitions"]/r["transitions"] for r in last),
                last_ten_reward_mean=mean(r["replay_reward_mean"] for r in last),
                last_ten_terminal_accounting_passed=True)
    groups = {}
    for (label, seed, scene), cases in grouped.items():
        groups[f"{label}/seed{seed}/{scene}"] = dict(
            cost_seconds=mean(r["cost"] for r in cases),
            mean_wait_seconds=mean(r["groups"]["all"]["mean_wait_seconds"] for r in cases),
            mean_onboard_seconds=mean(r["groups"]["all"]["mean_onboard_seconds"] for r in cases),
            holding_seconds=mean(r["mean_holding_seconds"] for r in cases),
            original_window_cost_seconds=mean(r["groups"]["baseline_during"]["mean_generalized_seconds"] for r in cases))
        value = groups[f"{label}/seed{seed}/{scene}"]
        assert abs(value["cost_seconds"]-2*value["mean_wait_seconds"]-value["mean_onboard_seconds"]) < 1e-7
    output = dict(complete=True, analysis_location="server", new_rollouts=0,
                  training=training, per_seed=groups)
    (HERE / "mechanism_summary.json").write_text(json.dumps(output, indent=2, allow_nan=False) + "\n")
    print(json.dumps(dict(complete=True, new_rollouts=0, training=training,
                         burst50={k:v for k,v in groups.items() if k.endswith("/burst50")})))


if __name__ == "__main__":
    inspect()
