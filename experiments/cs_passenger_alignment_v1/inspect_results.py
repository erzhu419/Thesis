"""Small paired comparisons and training diagnostics, computed on the server."""

from collections import defaultdict
import json
from pathlib import Path
from statistics import mean

HERE = Path(__file__).resolve().parent
P = json.loads((HERE / "protocol.json").read_text())


def group_metrics(rows):
    return dict(cost=mean(r["cost"] for r in rows),
        holding=mean(r["mean_holding_seconds"] for r in rows),
        headway_penalty=mean(r["native_headway_penalty"] for r in rows),
        completion=mean(r["groups"]["all"]["completion_rate"] for r in rows),
        mean_wait=mean(r["groups"]["all"]["mean_wait_seconds"] for r in rows),
        mean_onboard=mean(r["groups"]["all"]["mean_onboard_seconds"] for r in rows))


def inspect():
    rows = [json.loads(p.read_text()) for p in (HERE / "results").glob("*/seed*/day*.json")]
    assert len(rows) == 234
    cases = {(r["arm"], r["seed"], r["scene"], r["evaluation_seed"]): r for r in rows}
    assert len(cases) == len(rows)
    per_seed = defaultdict(list)
    for r in rows:
        per_seed[f'{r["arm"]}/seed{r["seed"]}/{r["scene"]}'].append(r)
    comparisons = {}
    for candidate, reference in [("observed_overlap", "local_overlap"),
            ("observed_overlap", "frozen_passenger"),
            ("observed_overlap", "fixed0"),
            ("global_event", "observed_overlap"),
            ("global_shaped", "global_event"),
            ("global_shaped", "observed_overlap")]:
        for scene in P["scenes"]:
            pairs = []
            for seed in P["training_seeds"]:
                for day in P["evaluation_seeds"]:
                    a = cases[candidate, seed, scene, day]
                    b = cases[reference, 0 if reference == "fixed0" else seed, scene, day]
                    pairs.append((b["cost"] - a["cost"], b["cost"]))
            reduction, reference_mean = mean(p[0] for p in pairs), mean(p[1] for p in pairs)
            comparisons[f"{candidate}-vs-{reference}/{scene}"] = dict(
                reduction_seconds=reduction, reduction_percent=100 * reduction / reference_mean,
                improved_cases=sum(p[0] > 0 for p in pairs), cases=len(pairs))
    training = {}
    for path in (HERE / "runs").glob("*/seed*/train_summary.json"):
        record = json.loads(path.read_text())
        history = record["history"]
        assert len(history) == 100 and record["training_steps"] == 40000
        assert all(r["updates"] == 400 and r["horizon_seconds"] == 18000 and
                   r["timestamp_cost_identity_passed"] for r in history)
        settings = json.loads((path.parent / "settings.json").read_text())
        config = settings["config"]
        arm = settings["arm"]
        if arm.startswith("global"):
            assert all(r["credit"]["cost_identity_passed"] and
                       r["credit"]["terminal_transitions"] == 1 for r in history)
        blocks = {}
        for start, end in [(1, 10), (41, 50), (91, 100)]:
            block = history[start-1:end]
            blocks[f"{start}-{end}"] = dict(cost=mean(r["cost"] for r in block),
                holding=mean(r["mean_holding_seconds"] for r in block),
                transitions=mean(r["transitions"] for r in block),
                replay_reward_mean=mean(r["replay_reward_mean"] for r in block),
                replay_reward_std=mean(r["replay_reward_std"] for r in block),
                reward_ema_std=mean(r["reward_ema_std"] for r in block),
                alpha=mean(r["update_mean"]["alpha"] for r in block),
                q_mean=mean(r["update_mean"]["q_mean"] for r in block),
                q_std=mean(r["update_mean"]["q_std"] for r in block),
                q_loss=mean(r["update_mean"]["q_loss"] for r in block),
                zero_time_fraction=mean(r["credit"].get("zero_time_transitions", 0) /
                                        r["transitions"] for r in block))
        training[str(path.parent.relative_to(HERE / "runs"))] = dict(
            gamma=config["gamma"], enable_reward_ema=config["enable_reward_ema"],
            weight_reg=config["weight_reg"], beta_ood=config["beta_ood"],
            use_nau_actor=config["use_nau_actor"], jac_weight=config["jac_weight"],
            weight_sym=config["weight_sym"], blocks=blocks,
            all_episode_cost_identities_passed=True,
            terminal_transition_fraction=mean(r["credit"].get("terminal_transitions", 0) /
                                             r["transitions"] for r in history))
    window = {}
    for arm in ["fixed0", "frozen_passenger", "frozen_headway"] + P["arms"]:
        values = {}
        for scene in ["normal", "burst50"]:
            subset = [r for r in rows if r["arm"] == arm and r["scene"] == scene]
            values[scene] = mean(r["groups"]["baseline_during"]["mean_generalized_seconds"] for r in subset)
        window[arm] = dict(**values, increase_seconds=values["burst50"]-values["normal"],
                          increase_percent=100*(values["burst50"]/values["normal"]-1))
    result = dict(complete=True, evaluations=234, training_runs=8, analysis_location="server",
        comparisons=comparisons, per_seed={k: group_metrics(v) for k, v in sorted(per_seed.items())},
        training=training, original_window_passenger_cost=window)
    (HERE / "mechanism_summary.json").write_text(json.dumps(result, indent=2, allow_nan=False)+"\n")
    print(json.dumps({k: result[k] for k in ["complete", "evaluations", "training_runs", "analysis_location"]}))


if __name__ == "__main__":
    inspect()
