"""All paired outcomes, frozen rule references, and intervention diagnostics."""

from collections import defaultdict
import json
from statistics import mean

from collect_results import HERE, P, summarize, write


LABELS = dict(normalized="归一化对照", decision_confidence="决策置信", source_weighted="源经验加权",
    shared_whole="共享编码整模型选择", shared_gated="共享编码逐机制选择")


def read(path):
    return json.loads(path.read_text())


def analyze():
    summary = summarize()
    if not summary["complete"]:
        raise ValueError("Full training and evaluation outcomes are required")
    groups = defaultdict(list)
    for row in summary["outcomes"]:
        groups[row["scenario"], row["demand_scale"], row["arm"]].append(row)
    cells = [dict(scenario=s, demand_scale=d, arm=a, cost=mean(r["cost"] for r in rows),
        delta_seconds=mean(r["cost_minus_normalized"] for r in rows),
        completion_delta_pp=mean(r["completion_pp_minus_normalized"] for r in rows),
        seed_deltas=[r["cost_minus_normalized"] for r in rows]) for (s, d, a), rows in groups.items()]
    effects = []
    for arm in P["arms"]:
        rows = [r for r in summary["outcomes"] if r["arm"] == arm]
        effects.append(dict(arm=arm, mean_delta_seconds=mean(r["cost_minus_normalized"] for r in rows),
            better_seed_conditions=sum(r["cost_minus_normalized"] < 0 for r in rows),
            better_paired_days=sum(r["paired_days_better"] for r in rows),
            completion_delta_pp_range=[min(r["completion_pp_minus_normalized"] for r in rows),
                max(r["completion_pp_minus_normalized"] for r in rows)]))
    rules = {}
    for method in ("daganzo_duet", "forward_proportional"):
        for case in range(10):
            rules[method, case] = read(HERE.parent/"daganzo_baseline_v1/runs"/method/"early60"/
                f"case{case:02d}/results/case{case:02d}.json")
    references, service, diagnostics = [], [], []
    for train in summary["training"]:
        arm, scene, seed = train["arm"], train["scenario"], train["seed"]
        folder = HERE/"runs"/scene/f"{arm}_seed{seed}"
        last = read(folder/"results/diagnostics/update070000.json")
        model = last["model"]["calibrate"]
        row = dict(arm=arm, scenario=scene, seed=seed,
            final_model_weight=model["actual_weights"]["mean"],
            prediction_loss=model["unweighted_prediction_loss"]["mean"],
            slot_identity_accuracy=model["components"]["slot_identity_accuracy"],
            scaled_gradient_ratio=last["scaled_model_gradient_norm"]/last["scaled_real_gradient_norm"],
            training_wall_seconds=train["training_wall_seconds"],
            model_history=train["model_history"])
        if arm == "source_weighted":
            intervals = [read(p)["source_reliability"] for p in sorted((folder/"results/diagnostics").glob("update*.json"))]
            physical = sum(r["physical_rows"] for r in intervals)
            row.update(source_physical_weight=mean(r["mean_physical_weight"] for r in intervals),
                covered_source_physical_fraction=sum(r["covered_rows"] for r in intervals)/physical)
        if arm == "decision_confidence":
            final = train["model_history"][-1]
            chosen = final["decision_components"][final["selected_family"]]
            row["normalized_calibration_components"] = dict(
                prediction=chosen["proper_error"]/final["prediction_scale"],
                decision=chosen["backup_absolute_error"]/final["decision_scale"])
        if arm.startswith("shared_"):
            row["mixed_selection_refreshes"] = sum(h["heterogeneous_selection"] for h in train["model_history"])
        diagnostics.append(row)
        for demand_index, demand in enumerate(P["evaluation_demand_scales"]):
            comparisons = defaultdict(list)
            components = defaultdict(list)
            for case in range(5*demand_index, 5*demand_index+5):
                actor = read(folder/"results"/f"eval_{case:02d}.json")
                control = read(HERE/"runs"/scene/f"normalized_seed{seed}/results"/f"eval_{case:02d}.json")
                h2o = read(HERE.parent/"mcwm_usage_v3/runs"/scene/f"h2o_matched_seed{seed}/results"/f"eval_{case:02d}.json")
                for name, reference in dict(h2o_matched=h2o,
                        **{m: rules[m, case] for m in ("daganzo_duet", "forward_proportional")}).items():
                    block = P["evaluation_blocks"][case % 5]
                    for key, expected in dict(horizon_seconds=24000., demand_scale=demand,
                            deployment_hold_scale=1., **block).items():
                        assert actor[key] == reference[key] == expected, (name, case, key)
                    assert reference["sumo_version"][1] == actor["sumo_version"][1] == "SUMO 1.25.0"
                    assert actor["passengers"]["departed_passengers"] == reference["passengers"]["departed_passengers"]
                    comparisons[name].append(dict(case=case,
                        delta_seconds=actor["generalized_seconds_per_departed_passenger"]-reference["generalized_seconds_per_departed_passenger"],
                        completion_delta_pp=100*(actor["passengers"]["completion_rate"]-reference["passengers"]["completion_rate"])))
                for key in ("accrued_wait_seconds_per_departed_passenger", "accrued_in_vehicle_seconds_per_departed_passenger"):
                    components[key].append(actor["passengers"][key]-control["passengers"][key])
                components["headway_error"].append(actor["headway"]["normalized_absolute_error"]["mean"]-
                    control["headway"]["normalized_absolute_error"]["mean"])
                components["mean_hold_seconds"].append(actor["mean_hold_seconds"]-control["mean_hold_seconds"])
            for name, days in comparisons.items():
                references.append(dict(arm=arm, scenario=scene, seed=seed, demand_scale=demand,
                    reference=name, delta_seconds=mean(r["delta_seconds"] for r in days),
                    completion_delta_pp=mean(r["completion_delta_pp"] for r in days),
                    better_days=sum(r["delta_seconds"] < 0 for r in days), days=days))
            service.append(dict(arm=arm, scenario=scene, seed=seed, demand_scale=demand,
                delta_vs_normalized={k: mean(v) for k, v in components.items()}))
    old = read(HERE.parent/"mcwm_usage_v3/summary.json")
    old_control = {(r["scenario"], r["demand_scale"], r["seed"]): r for r in old["outcomes"] if r["arm"] == "normalized"}
    control_repeat = [dict(scenario=r["scenario"], demand_scale=r["demand_scale"], seed=r["seed"],
        delta_seconds=r["cost"]-old_control[r["scenario"], r["demand_scale"], r["seed"]]["cost"])
        for r in summary["outcomes"] if r["arm"] == "normalized"]
    result = dict(training_count=30, evaluation_count=300, cells=cells, overall=effects,
        diagnostics=diagnostics, reference_comparisons=references, service_decomposition=service,
        normalized_repeat=control_repeat, shared_gate_minus_whole=mean(
            r["mean_delta_seconds"] for r in effects if r["arm"] == "shared_gated")-
            mean(r["mean_delta_seconds"] for r in effects if r["arm"] == "shared_whole"),
        scope=P["interpretation"], timing_scope="Concurrent server wall time; not an isolated hardware speed comparison")
    write(HERE/"analysis.json", result)
    return result


def report(result):
    cells = {(r["scenario"], r["demand_scale"], r["arm"]): r for r in result["cells"]}
    lines = ["# 三方向探索结果", "", "30条训练与300个24000秒SUMO测试日均完成。下表为两训练种子、五个配对测试日的均值；差值单位为广义秒/已出发乘客，负值表示优于本轮归一化对照。", "",
        "| 源仿真场景 | 需求 | 对照时间 | 决策置信差值 | 源加权差值 | 共享整模型差值 | 共享逐机制差值 |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for scene in (s["name"] for s in P["scenarios"]):
        for demand in P["evaluation_demand_scales"]:
            values = [cells[scene, demand, a] for a in P["arms"]]
            lines.append(f"| {scene} | {demand:g} | {values[0]['cost']:.2f} | "+
                " | ".join(f"{r['delta_seconds']:+.2f}" for r in values[1:])+" |")
    lines += ["", "| 方法 | 总体时间差值 | 改善的种子条件/12 | 改善的配对测试日/60 | 完成率差值范围/百分点 |",
        "| --- | ---: | ---: | ---: | ---: |"]
    for r in result["overall"]:
        if r["arm"] == "normalized":
            continue
        lo, hi = r["completion_delta_pp_range"]
        lines.append(f"| {LABELS[r['arm']]} | {r['mean_delta_seconds']:+.2f} | {r['better_seed_conditions']} | {r['better_paired_days']} | {lo:+.3f} 至 {hi:+.3f} |")
    lines += ["", "归一化版继续作为总体主结果。决策置信和源加权在标定正常及最大偏差场景有局部收益，但中等偏差退化，不能表述为跨场景全面改进。共享逐机制比共享整模型平均降低17.25秒，六组训练的四次刷新均出现混合选择，机制并非未执行；它仍比归一化版平均高5.23秒。", "",
        "归一化版及决策置信版相对于H2O+、共同保障下的Daganzo式代理和前向比例规则，都在12个种子条件和60个配对测试日上降低时间，条件平均完成率未下降。新方向未整体超过归一化版，不等于既有MC-WM公交结果失效。", "",
        "决策校准的最终决策项约0.006至0.016，预测项约为1；当前尺度使决策误差作用偏弱。源物理经验加权实际生效，下一步不再扩大共享编码结构，优先单独检验决策误差尺度。", "",
        "## 范围", "", "本轮仍是两个开发训练种子和既有评价日，保留全部正负结果。新归一化对照比上一轮总体高4.75秒，因此主比较采用本轮对照；不能把几秒均值改善说成已确证的稳定优势。Daganzo式代理和前向比例规则采用共同保障层，既有规则日不重复计为新的训练样本。服务器并发时间不用于算法速度优劣结论。种子、逐日差值、完成率、候车与车内时间、规则比较及机制诊断完整保存于analysis.json；论文正文未改动。"]
    return "\n".join(lines)+"\n"


if __name__ == "__main__":
    result = analyze()
    print(json.dumps(dict(overall=result["overall"], shared_gate_minus_whole=result["shared_gate_minus_whole"])))
    # This report is generated from the complete paired results, not hand-selected days.
    (HERE/"结果小结.md").write_text(report(result))
