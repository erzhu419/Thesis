"""Summarize paired service outcomes and draw a Chinese quantitative figure."""

import argparse
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.ticker import FormatStrFormatter, MaxNLocator
import numpy as np

from data import HERE

LABELS = {"offline_prefix": "离线初始化", "offline_continue": "仅离线续训",
          "h2o_mixed": "源仿真续训", "mcwm_assisted": "世界模型辅助"}
COLORS = {"offline_prefix": "#92979D", "offline_continue": "#D0904A",
          "h2o_mixed": "#447EAC", "mcwm_assisted": "#248878"}


def main(out):
    p = json.loads((out / "protocol.json").read_text())
    rows = []
    for method in p["methods"]:
        for scale in p["evaluation_demand_scales"]:
            for block in p["evaluation_blocks"]:
                path = out / "evaluation" / f"{method}_demand{scale:g}_seed{block['sumo_seed']}.json"
                result = json.loads(path.read_text())
                if result["horizon_seconds"] != p["evaluation_horizon_seconds"]:
                    raise RuntimeError("Incomplete evaluation horizon")
                passengers = result["passengers"]
                rows.append({"method": method, "demand_scale": scale, **block,
                             "generalized_seconds_per_passenger": result["generalized_seconds_per_departed_passenger"],
                             "completion_percent": 100*passengers["completion_rate"],
                             "wait_seconds_per_passenger": passengers["accrued_wait_seconds_per_departed_passenger"],
                             "ride_seconds_per_passenger": passengers["accrued_in_vehicle_seconds_per_departed_passenger"],
                             "mean_hold_seconds": result["mean_hold_seconds"],
                             "departed_passengers": passengers["departed_passengers"],
                             "completed_trips": result["service"]["completed_trips"],
                             "departed_trips": result["service"]["departed_trips"]})
    reference = {(r["demand_scale"], r["sumo_seed"]): r for r in rows if r["method"] == "h2o_mixed"}
    for row in rows:
        base = reference[(row["demand_scale"], row["sumo_seed"])]
        row["cost_reduction_vs_h2o_percent"] = 100*(1-row["generalized_seconds_per_passenger"]/base["generalized_seconds_per_passenger"])
        row["completion_change_vs_h2o_pp"] = row["completion_percent"]-base["completion_percent"]
    with (out / "paired_service_results.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    metrics = ("generalized_seconds_per_passenger", "completion_percent", "cost_reduction_vs_h2o_percent",
               "completion_change_vs_h2o_pp", "mean_hold_seconds")
    means = {method: {key: float(np.mean([r[key] for r in rows if r["method"] == method]))
                       for key in metrics} for method in p["methods"]}
    selected = [row for row in rows if row["method"] == "mcwm_assisted"]
    summary = {"completed_evaluations": len(rows), "training_seeds": [p["training_seed"]],
               "means": means, "mcwm_cost_improved_cells": sum(r["cost_reduction_vs_h2o_percent"]>0 for r in selected),
               "mcwm_completion_min_change_pp": min(r["completion_change_vs_h2o_pp"] for r in selected),
               "rows": rows}
    (out / "service_summary.json").write_text(json.dumps(summary, indent=2) + "\n")

    # Show both service cost and completion; individual scene dots are not confidence intervals.
    font = "/mnt/c/Windows/Fonts/msyh.ttc"
    font_manager.fontManager.addfont(font)
    plt.rcParams.update({"font.family": font_manager.FontProperties(fname=font).get_name(),
                         "font.size": 8, "axes.unicode_minus": False,
                         "pdf.fonttype": 42, "svg.fonttype": "none",
                         "axes.spines.top": False, "axes.spines.right": False})
    fig, axes = plt.subplots(2, 2, figsize=(7.1, 4.6), layout="constrained")
    for i, scale in enumerate(p["evaluation_demand_scales"]):
        for j, (key, xlabel) in enumerate((("generalized_seconds_per_passenger", "人均广义乘客时间 / 秒"),
                                          ("completion_percent", "行程完成率 / %"))):
            ax = axes[i, j]
            for y, method in enumerate(p["methods"]):
                values = [r[key] for r in rows if r["method"] == method and r["demand_scale"] == scale]
                ax.scatter(values, np.array([y-.075, y+.075]), s=24, color=COLORS[method], zorder=3)
                ax.plot([np.mean(values)], [y], marker="|", ms=10, mew=1.6, color="#222222", linestyle="none")
            ax.set_yticks(range(4), [LABELS[method] for method in p["methods"]] if j==0 else [])
            ax.set_ylim(3.5, -.5)
            ax.set_title(f"({chr(97+2*i+j)}) 需求倍率 {scale:g}", loc="left", fontsize=9)
            ax.set_xlabel(xlabel)
            if j == 1:
                ax.xaxis.set_major_locator(MaxNLocator(nbins=4))
                ax.xaxis.set_major_formatter(FormatStrFormatter("%.2f"))
            ax.grid(axis="x", color="#E4E6E7", linewidth=.5)
    for suffix in ("png", "pdf", "svg"):
        fig.savefig(out / f"公交世界模型_闭环对照.{suffix}", dpi=240)
    plt.close(fig)
    text = ["# 公交世界模型小规模闭环补充实验", "",
            "三种续训方法采用同一离线初始化和相同策略更新预算，两个混合分支共用源仿真回放；冻结初始化另作参照。所有策略采用相同的停站幅度变换与完成保障规则。",
            "", "| 方法 | 平均人均广义时间（秒） | 平均完成率（%） | 相对源续训的平均时间下降（%） |",
            "| --- | ---: | ---: | ---: |"]
    for method in p["methods"]:
        m = means[method]
        text.append(f"| {LABELS[method]} | {m['generalized_seconds_per_passenger']:.2f} | {m['completion_percent']:.3f} | {m['cost_reduction_vs_h2o_percent']:.3f} |")
    text += ["", f"世界模型辅助在{summary['mcwm_cost_improved_cells']}/4组配对场景中降低广义时间；完成率相对源续训的最小变化为{summary['mcwm_completion_min_change_pp']:.3f}个百分点。",
             "", "| 需求倍率 | 环境块 | 世界模型相对源续训的时间下降（%） | 完成率变化（百分点） |",
             "| --- | --- | ---: | ---: |"]
    for row in selected:
        text.append(f"| {row['demand_scale']:g} | {row['sumo_seed']} | {row['cost_reduction_vs_h2o_percent']:.3f} | {row['completion_change_vs_h2o_pp']:.3f} |")
    text += ["", "## 解释范围", "",
             "本组为一个训练种子、两种需求、两个共同环境块的补充验证，不进行显著性推断。离线初始化为既有50,000步策略，其他三组各增加2,000次策略更新和2,500次价值网络更新。世界模型方法另有模型拟合与可信度学习开销。",
             "", "世界模型采用校准选择的公交混合事件预测器，并复用MC-WM的贝尔曼可信度方法加权一步生成样本。源数据由共同冻结策略在简易仿真中采集，续训阶段共用这份回放，不额外进行SUMO训练交互。终止段只使用实测日志和源仿真转移，未交给生成模型。",
             "", "图中每种方法的两个圆点对应两个环境块，黑色短线为其均值。广义时间越小越好，完成率越大越好。"]
    (out / "公交闭环补充结果.md").write_text("\n".join(text) + "\n")
    print(json.dumps({key: value for key, value in summary.items() if key != "rows"}, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=HERE / "runs/control_v1")
    main(parser.parse_args().out)
