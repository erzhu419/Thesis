"""Paired-day comparisons without treating reused rule days as new replicates."""

import json
from pathlib import Path
from statistics import mean, stdev
import sys

HERE = Path(__file__).resolve().parent
USAGE = HERE.parent / "mcwm_usage_v3"
sys.path.insert(0, str(USAGE))
from usage_common import P as USAGE_P, write

P = json.loads((HERE / "protocol.json").read_text())


def read(path):
    return json.loads(path.read_text())


def validate(row, case):
    block = P["evaluation_blocks"][case % 5]
    expected = dict(horizon_seconds=P["evaluation_horizon_seconds"],
                    deployment_hold_scale=P["deployment_hold_scale"],
                    demand_scale=P["evaluation_demand_scales"][case // 5], **block)
    for key, value in expected.items():
        assert row[key] == value, (case, key, row[key], value)
    assert row["sumo_version"][1] == "SUMO 1.25.0"
    assert row["passengers"]["active_state_count_balance_error"] == 0


def summarize(rows):
    costs = [r["generalized_seconds_per_departed_passenger"] for r in rows]
    return dict(cost=mean(costs), day_cost_sd=stdev(costs),
        completion_percent=100 * mean(r["passengers"]["completion_rate"] for r in rows),
        wait_seconds=mean(r["passengers"]["accrued_wait_seconds_per_departed_passenger"] for r in rows),
        ride_seconds=mean(r["passengers"]["accrued_in_vehicle_seconds_per_departed_passenger"] for r in rows),
        headway_error=mean(r["headway"]["normalized_absolute_error"]["mean"] for r in rows),
        bunching_percent=100 * mean(r["headway"]["bunching_rate"] for r in rows),
        mean_hold_seconds=mean(r["mean_hold_seconds"] for r in rows))


def analyze():
    rules = {}
    references = [(m, layer) for m in P["methods"] for layer in P["execution_layers"]]
    for method, layer in references:
        for case in range(10):
            row = read(HERE / f"runs/{method}/{layer}/case{case:02d}/results/case{case:02d}.json")
            validate(row, case)
            assert row["method"] == method and row["execution_layer"] == layer
            assert row["comparison_protocol"] == P["name"]
            assert row["rule_diagnostic"]["decisions"] == row["decision_events"]
            rules[method, layer, case] = row
    for method in ("headway", "zero"):
        references.append((method, "early60"))
        for case in range(10):
            row = read(USAGE / f"rules/{method}/case{case:02d}/results/case{case:02d}.json")
            validate(row, case)
            assert row["execution_layer"] == "early60"
            rules[method, "early60", case] = row
    baselines, comparisons, actors = [], [], []
    for demand_index, demand in enumerate(P["evaluation_demand_scales"]):
        cases = range(demand_index * 5, demand_index * 5 + 5)
        for method, layer in references:
            rows = [rules[method, layer, case] for case in cases]
            entry = dict(method=method, layer=layer, demand_scale=demand, **summarize(rows))
            if method in P["methods"]:
                decisions = sum(r["rule_diagnostic"]["decisions"] for r in rows)
                entry["rule_diagnostic"] = {
                    key: sum(r["rule_diagnostic"][key] for r in rows) / decisions
                    for key in ("different_from_adjusted_headway", "absolute_hold_difference_seconds",
                                "unequal_scheduled_targets", "missing_neighbor")}
            baselines.append(entry)
        for scene in (s["name"] for s in USAGE_P["scenarios"]):
            for arm in ("h2o_matched", "normalized"):
                for seed in USAGE_P["training_seeds"]:
                    rows = [read(USAGE / f"runs/{scene}/{arm}_seed{seed}/results/eval_{c:02d}.json") for c in cases]
                    for case, row in zip(cases, rows):
                        validate(row, case)
                        assert (row["arm"], row["scenario"], row["training_seed"]) == (arm, scene, seed)
                    actors.append(dict(scenario=scene, arm=arm, seed=seed, demand_scale=demand, **summarize(rows)))
                    for method, layer in references:
                        days = []
                        for case, row in zip(cases, rows):
                            ref = rules[method, layer, case]
                            assert row["passengers"]["departed_passengers"] == ref["passengers"]["departed_passengers"]
                            days.append(dict(case=case,
                                reduction_seconds=ref["generalized_seconds_per_departed_passenger"] - row["generalized_seconds_per_departed_passenger"],
                                completion_gain_pp=100 * (row["passengers"]["completion_rate"] - ref["passengers"]["completion_rate"])))
                        baseline_cost = mean(rules[method, layer, c]["generalized_seconds_per_departed_passenger"] for c in cases)
                        reduction = mean(d["reduction_seconds"] for d in days)
                        comparisons.append(dict(method=method, layer=layer, scenario=scene, arm=arm,
                            seed=seed, demand_scale=demand, reduction_seconds=reduction,
                            reduction_percent=100 * reduction / baseline_cost,
                            completion_gain_pp=mean(d["completion_gain_pp"] for d in days),
                            better_days=sum(d["reduction_seconds"] > 0 for d in days), days=days))
    result = dict(complete=True, new_evaluations=40, reused_rule_evaluations=20,
                  reused_actor_evaluations=120, baselines=baselines, actors=actors,
                  comparisons=comparisons, scope=P["scope"], native_reference=P["native_reference"])
    write(HERE / "analysis.json", result)
    return result


def report(result):
    labels = {"daganzo_duet": "TransitDuet双向时距代理", "forward_proportional": "前向比例规则（0.6）",
              "headway": "原计划偏差时距规则", "zero": "零停站建议"}
    lines = ["# 固定规则补充比较", "",
        "40次新增完整评价全部完成；每项规则在两档需求下各评价5个24000秒运营日。表中时间为广义秒/已出发乘客，越低越好。", "",
        "| 固定规则 | 执行方式 | 正常需求时间 | 高需求时间 | 正常完成率/% | 高需求完成率/% |",
        "| --- | --- | ---: | ---: | ---: | ---: |"]
    pairs = list(dict.fromkeys((r["method"], r["layer"]) for r in result["baselines"]))
    for method, layer in pairs:
        rows = [r for r in result["baselines"] if (r["method"], r["layer"]) == (method, layer)]
        rows.sort(key=lambda r: r["demand_scale"])
        a, b = rows
        execution = "独立执行" if layer == "native" else "共同保障"
        lines.append(f"| {labels[method]} | {execution} | {a['cost']:.2f} | {b['cost']:.2f} | {a['completion_percent']:.3f} | {b['completion_percent']:.3f} |")
    lines += ["", "## 相同保障下的学习增量", "",
        "| 规则参照 | 学习方法 | 时间改善的种子条件/12 | 时间改善的逐日配对/60 | 六条件平均降幅范围/% | 完成率变化范围/百分点 |",
        "| --- | --- | ---: | ---: | ---: | ---: |"]
    for method in P["methods"]:
        for arm in ("h2o_matched", "normalized"):
            rows = [r for r in result["comparisons"] if (r["method"], r["layer"], r["arm"]) == (method, "early60", arm)]
            cells = [(s["name"], d) for s in USAGE_P["scenarios"] for d in P["evaluation_demand_scales"]]
            reductions, completions = [], []
            for scene, demand in cells:
                pair = [r for r in rows if r["scenario"] == scene and r["demand_scale"] == demand]
                reductions.append(mean(r["reduction_percent"] for r in pair))
                completions.append(mean(r["completion_gain_pp"] for r in pair))
            label = "H2O+" if arm == "h2o_matched" else "归一化MC-WM"
            lines.append(f"| {labels[method]} | {label} | {sum(r['reduction_seconds'] > 0 for r in rows)} | {sum(r['better_days'] for r in rows)} | {min(reductions):+.2f} 至 {max(reductions):+.2f} | {min(completions):+.3f} 至 {max(completions):+.3f} |")
    lines += ["", "正降幅表示学习方法的时间较低。条件为三种源标定与两档目标需求的组合；每条件两次训练分别保留，规则不重复计为新的训练样本。候车、车内时间、时距误差及逐日差值见同目录analysis.json。", "",
        "## 适用范围", "",
        "双向代理沿用TransitDuet的固定0.5系数及0至60秒停站范围；前向规则沿用第五章的0.6系数。当前时刻表、车队和观测接口保持不变，缺失邻车按当前计划时距占位。两者不是Daganzo原控制器的完整复现。独立执行结果另列，主结论只依据共同保障下的比较。", "",
        "本轮补充既有开发评价日的参照，不代表独立确认或真实道路验证。后续将规则定义、同保障比较及独立执行结果一起纳入第六章，保留全部正负结果。"]
    (HERE / "结果小结.md").write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    result = analyze()
    report(result)
    for row in result["baselines"]:
        print(row["method"], row["layer"], row["demand_scale"],
              round(row["cost"], 3), round(row["completion_percent"], 3))
    for method in P["methods"]:
        for arm in ("h2o_matched", "normalized"):
            rows = [r for r in result["comparisons"] if r["method"] == method and r["layer"] == "early60" and r["arm"] == arm]
            print(method, arm, "better_seed_conditions", sum(r["reduction_seconds"] > 0 for r in rows), "/", len(rows),
                  "better_paired_days", sum(r["better_days"] for r in rows), "/", sum(len(r["days"]) for r in rows))
