"""Summarize existing small result files; never fetch remote data or edit Word."""

import json
from pathlib import Path
from statistics import mean

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "workpapers/论文数据对照_20261009"
SCENES = {
    "normal": "常态", "pulse5": "5倍单次冲击", "repeated5": "5倍重复冲击",
    "static5": "5倍持续需求", "static10": "10倍持续需求", "burst50": "50倍短时冲击",
}
COUNT_KEYS = {
    "training_ready", "training_expected", "expected_training", "training_complete",
    "formal_training", "formal_evaluations", "expected_evaluations", "evaluation_ready",
    "evaluation_expected", "evaluation_complete", "expected_evaluation", "evaluations",
    "episodes", "training_episodes", "checkpoint_episode", "episodes_per_seed",
    "completed_evaluations", "dual_evaluations", "fixed_speed_evaluations",
    "reused_holding_evaluations", "reused_policies", "initialization_ready",
}


def read(path):
    return json.loads((ROOT / path).read_text())


def link(path):
    return f"[{path}](../../{path})"


def summarize_file(path):
    data = read(path)
    if not isinstance(data, dict):
        return {"source": path}
    flags = {k: data[k] for k in ["complete", "primary_complete", "budget_comparison_complete"]
             if k in data}
    for name in ["legacy", "cs", "mc", "ratio"]:
        if isinstance(data.get(name), dict) and "complete" in data[name]:
            flags[name + ".complete"] = data[name]["complete"]
    counts = {k: value for k, value in data.items()
              if k in COUNT_KEYS and isinstance(value, (int, float)) and not isinstance(value, bool)}
    if isinstance(data.get("counts"), dict):
        counts["counts"] = data["counts"]
    if isinstance(data.get("training_seeds"), list):
        counts["training_repetitions"] = len(data["training_seeds"])
    return {"source": path, "recorded_completion": flags, "recorded_counts": counts}


def selected_cs():
    source = "experiments/resac_native_recipe_v8/analysis_summary.json"
    data = read(source)
    rows = []
    for objective in ["passenger", "headway"]:
        for scene, name in SCENES.items():
            base, = [r for r in data["aggregates"] if
                     (r["mode"], r["objective"], r["arm"], r["budget"], r["scene"]) ==
                     ("holding", objective, "resac", 100, scene)]
            pairs = [r for r in data["cs_comparisons"] if
                     (r["mode"], r["objective"], r["arm"], r["budget"], r["scene"]) ==
                     ("holding", objective, "constant_symbolic", 100, scene)]
            assert len(pairs) == 6 and {r["seed"] for r in pairs} == {64001, 64002}
            assert all(r["reference"] == "resac" and r["people_difference"] == 0 for r in pairs)
            cost_delta = mean(r["cost_delta"] for r in pairs)
            reward_delta = mean(r["reward_delta"] for r in pairs)
            rows.append({
                "objective": objective, "scene": scene, "scene_name": name, "paired_cases": 6,
                "training_seeds": [64001, 64002], "reference": "resac", "arm": "constant_symbolic",
                "resac_cost_seconds": base["cost"], "cs_cost_seconds": base["cost"] + cost_delta,
                "cost_delta_seconds": cost_delta, "cost_reduction_pct": -cost_delta / base["cost"] * 100,
                "resac_reward": base["reward"], "cs_reward": base["reward"] + reward_delta,
                "headway_penalty_reduction_pct": reward_delta / abs(base["reward"]) * 100
                    if objective == "headway" else None,
                "cost_improved_pairs": sum(r["cost_delta"] < 0 for r in pairs),
                "reward_improved_pairs": sum(r["reward_delta"] > 0 for r in pairs),
                "plan_error_delta_seconds": mean(r["plan_delta"] for r in pairs),
                "imbalance_delta_seconds": mean(r["imbalance_delta"] for r in pairs),
            })
    return {
        "source": source, "mode": "holding", "training_episodes": 100,
        "lineage": "bus_control_repair_v3/smooth_soft -> bus_algorithm_matrix_v6/constant_symbolic -> resac_native_recipe_v8 matched comparison",
        "adopted_position": "用户2026-10-09选定的摘要第(2)项替换文字；未修改导师Word",
        "calculation": "各场景六个配对差取平均；CS均值=RE-SAC均值+平均差；百分比以RE-SAC均值为分母。时距惩罚以平均负奖励绝对值为分母。",
        "rows": rows,
    }


def main():
    ledger = read(OUT.relative_to(ROOT) / "实验结论登记.json")
    registry = read(OUT.relative_to(ROOT) / "版本登记.json")
    index = read(OUT.relative_to(ROOT) / "正文图表索引.json")
    directories = {d.name for d in (ROOT / "experiments").iterdir()
                   if d.is_dir() and not d.name.startswith("__")}
    registered = {r["version"] for r in ledger["campaigns"]}
    assert len(registered) == len(ledger["campaigns"])
    assert directories == registered, (directories - registered, registered - directories)
    assert {f["id"] for f in registry["families"]} == set(ledger["family_results"])

    campaigns = []
    for row in ledger["campaigns"]:
        assert all((ROOT / s).is_file() for s in row["sources"]), row["version"]
        records = [summarize_file(s) for s in row["sources"] if s.endswith(".json")]
        location_records = [str(p.relative_to(ROOT)) for p in sorted(
            (ROOT / "experiments" / row["version"]).iterdir()) if p.is_file() and
            p.suffix in {".json", ".md"} and any(
                name in p.stem for name in ["protocol", "submission", "node_runtime", "jobs"])]
        campaigns.append({**row, "local_summary_records": records,
                          "remote_location_records": location_records})
    selected = selected_cs()
    (OUT / "CSBAPR摘要采用结果.json").write_text(json.dumps(selected, ensure_ascii=False, indent=2) + "\n")
    (OUT / "实验版本结果总览.json").write_text(json.dumps({
        "date": ledger["date"], "basis": "已在本地的服务器分析摘要、冻结图源及现有协议；本次未连接服务器",
        "campaigns": campaigns, "family_results": ledger["family_results"],
        "selected_cs": selected,
    }, ensure_ascii=False, indent=2) + "\n")

    lines = ["# 实验版本结果总览", "", "整理日期：2026-10-09。", "",
             f"覆盖`experiments/`全部{len(campaigns)}个顶层实验目录，以及论文对照文档的{len(registry['families'])}个来源组。每批列主要结论、采用位置和对应汇总，未采用或退化分支也保留。",
             "结论来自已经回传的小型服务器分析汇总和本地冻结数据，不需要原始训练轨迹、离线大文件或权重下载。", "",
             "## 当前采用结果", "",
             "- RE-SAC：原研究归档、走廊双目标与集成参照比较；native-v8多线路原配方单列。",
             "- CS-BAPR：摘要替换文字采用旧训练软SINDy策略与native-v8 RE-SAC的匹配比较；旧正文MLP/BAPR表仍按旧来源登记。",
             "- H2O+：原正式完成保障确认、短预算作用分离和同预算离线/源/混合比较分别保留。",
             "- MC-WM：归一化使用、v13固定TD二十重复、v19单驻站时距及附录F最终225回合分别保留。", "",
             "## CS-BAPR摘要的新数据来源", "",
             "算法：去NAU、可学习基准动作、中心化神经残差、软SINDy约束；单驻站、100回合。旧端点`smooth_soft`在后续矩阵中叫`constant_symbolic`，不是固定动作或直接SINDy控制。",
             "比较取`resac_native_recipe_v8/analysis_summary.json`的`cs_comparisons`和`aggregates`；固定筛选`holding`、`constant_symbolic`、`budget=100`，两种奖励分别取各自策略。每场景包含两个训练重复×三个共同评价日。", "",
             "| 场景 | 乘客目标RE-SAC/秒 | 乘客目标CS-BAPR/秒 | 乘客成本降低/% | 时距目标惩罚降低/% |",
             "| --- | ---: | ---: | ---: | ---: |"]
    for scene in SCENES:
        passenger, = [r for r in selected["rows"] if r["scene"] == scene and r["objective"] == "passenger"]
        headway, = [r for r in selected["rows"] if r["scene"] == scene and r["objective"] == "headway"]
        lines.append(f"| {passenger['scene_name']} | {passenger['resac_cost_seconds']:.2f} | {passenger['cs_cost_seconds']:.2f} | {passenger['cost_reduction_pct']:.3f} | {headway['headway_penalty_reduction_pct']:.2f} |")
    lines += ["", "摘要范围：乘客目标六场景均值降低0.083%至0.549%；时距目标五场景惩罚降低10.41%至17.43%，持续10倍轻微退化0.27%。",
              "复算细节及各场景改善次数见[CSBAPR摘要采用结果.json](CSBAPR摘要采用结果.json)。正文定位见[论文数据与版本对照.md](论文数据与版本对照.md)。", ""]
    for group in dict.fromkeys(r["group"] for r in campaigns):
        lines += [f"## {group}", "", "| 版本 | 结果总结 | 采用情况 | 结果依据 |", "| --- | --- | --- | --- |"]
        for row in campaigns:
            if row["group"] == group:
                lines.append(f"| `{row['version']}` | {row['result']} | {row['use']} | " +
                             "<br>".join(link(s) for s in row["sources"]) + " |")
        lines.append("")
    lines += ["## 论文现有来源组的结论", "",
              "这一表包括相邻小论文、共同输入、观测及成本台账；其来源和图表位置与对照文档共享同一版本登记。", "",
              "| 版本代号 | 当前位置 | 主要结果/用途 |", "| --- | --- | --- |"]
    for family in registry["families"]:
        position = "、".join(family["figures"] + family["tables"] + family.get("text_positions", []))
        lines.append(f"| {family['id']} | {position} | {ledger['family_results'][family['id']]} |")
    lines += ["", "## 完成记录与远程原文件位置", "",
              "[实验版本结果总览.json](实验版本结果总览.json)逐批保留汇总中实际记录的完成标志、训练/评价计数及协议/派发记录入口。已有结果文件但没有全批完成标志的版本不自动补写为‘全部完成’。",
              "服务器大文件的节点和目录继续以各批`protocol`、`submission`、`jobs`、`node_runtime`及日志为准；这些位置记录不是本次检查服务器文件仍存在的证明。", "",
              "## 局限与未采用结论", "",
              "- 本次不改导师Word。用户已选定摘要替换文字，物理Word的旧摘要与第4章旧表尚未被替换。表4-10/4-11仍是稳定化MLP对BAPR，不可重标为软SINDy对RE-SAC。",
              "- 摘要软SINDy乘客目标各场景3/6配对改善，训练重复间有相反方向；时距奖励收益不等于全部时距误差都改善。同结构去符号策略成本接近，不能将全部收益归因SINDy。",
              "- 各批人群分母、仿真时域、反馈、保障、动作、预算和参照不同，表内差值有效，跨批绝对数不组成统一排名。直接SINDy、固定约2秒和学习软约束是三个不同端点。",
              "- TransitDuet独立300回合训练的末次主评价完成，但同训练轨迹第100回合的预算对照未齐。RE-SAC表3-2逐回合原始数据仍未恢复。",
              "- 未下载服务器大数据、未重新训练、未重新核查实时调度状态；本地归档摘要能够支持版本结论，但仓库不是所有权重和训练输入的完整镜像。", ""]
    (OUT / "实验版本结果总览.md").write_text("\n".join(lines))
    print(json.dumps({"campaigns": len(campaigns), "source_families": len(registry["families"]),
                      "selected_cs_rows": len(selected["rows"]), "word_modified": False}))


if __name__ == "__main__":
    main()
