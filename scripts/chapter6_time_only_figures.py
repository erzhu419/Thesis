"""Redraw Chapter 6 time outcomes without replacing archived figures."""

import csv
import json
from pathlib import Path

import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.text import Text

import build_figures as b


OUT = b.ROOT / "output/第6章_删除完成率重绘_20261014"
SCENES = ("calibrated", "joint50", "joint75")
LABELS = ("正常标定", "联合50%偏差", "联合75%偏差")
DATA = []
MANIFEST = []


def load(relative):
    return json.loads((b.ROOT / relative).read_text())


def record(figure, scenario, demand, method, level, value, seed="", case=""):
    DATA.append(dict(figure=figure, scenario=scenario, demand=demand, method=method,
                     level=level, seed=seed, case=case, value_seconds=float(value)))


def paired_range(ax, values, y, color, marker="o", spread=.07):
    values = np.asarray(values)
    ax.plot([values.min(), values.max()], [y, y], color=color, lw=1, alpha=.5)
    ax.scatter(values, y + np.linspace(-spread, spread, len(values)), s=24,
               marker=marker, facecolors="white", edgecolors=color,
               linewidths=.9, zorder=3)
    ax.scatter(values.mean(), y, s=34, marker="D", color=color, zorder=4)


def format_axis(ax, labels, limits, xlabel):
    ax.axvline(0, color=b.GRAY, lw=.85, ls="--")
    ax.grid(axis="x", color="#DCE1E5", linewidth=.45, alpha=.75)
    ax.set(xlim=limits, yticks=range(len(labels)), yticklabels=labels, xlabel=xlabel)
    ax.tick_params(axis="y", length=0)
    ax.xaxis.label.set_fontsize(8.5)


def save(fig, number, title, original, sources, statistics):
    # Detect clipped labels and accidental retention of the removed outcome.
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    for text in fig.findobj(Text):
        if not text.get_visible() or not text.get_text():
            continue
        assert "完成率" not in text.get_text(), (number, text.get_text())
        box = text.get_window_extent(renderer)
        if not text.get_clip_on() and box.width and box.height:
            assert box.x0 >= -1 and box.y0 >= -1, (number, text.get_text())
            assert box.x1 <= fig.bbox.width + 1, (number, text.get_text())
            assert box.y1 <= fig.bbox.height + 1, (number, text.get_text())
    for extension in ("png", "pdf", "svg"):
        fig.savefig(OUT / f"{number}.{extension}", dpi=600)
    MANIFEST.append(dict(figure=number, title=title, original_figure=original,
                         sources=sources, statistics=statistics,
                         width_mm=round(fig.get_figwidth()*25.4, 2),
                         height_mm=round(fig.get_figheight()*25.4, 2)))
    b.plt.close(fig)
    print(number, flush=True)


def alignment():
    source = "experiments/mcwm_alignment_v2/summary.json"
    data = load(source)
    fig, ax = b.plt.subplots(figsize=(5.75, 3.7))
    fig.subplots_adjust(left=.23, right=.97, bottom=.16, top=.85)
    scenarios = (("calibrated", "正常标定"), ("joint25", "联合25%偏差"),
                 ("joint50", "联合50%偏差"))
    for i, (scene, _) in enumerate(scenarios):
        if i % 2 == 0:
            ax.axhspan(2*i-.48, 2*i+1.48, color="#F3F5F6", zorder=0)
        for j, (demand, color) in enumerate((("1.0", b.BLUE), ("1.25", b.GOLD))):
            row = data["results"][scene][demand]
            values = [r["h2o_matched"]["cost"]-r["mcwm"]["cost"] for r in row["by_seed"]]
            np.testing.assert_allclose(np.mean(values), row["h2o_mean"]-row["mcwm_mean"])
            paired_range(ax, values, 2*i+j, color, spread=.11)
            for sample, value in zip(row["by_seed"], values):
                record("图6-10", scene, demand, "mcwm", "training_mean", value, sample["seed"])
    format_axis(ax, [], (-40, 85), "H2O+减MC-WM的人均广义时间成本 / 秒")
    ax.set(yticks=[.5, 2.5, 4.5], yticklabels=[label for _, label in scenarios],
           ylim=(5.6, -.6), xticks=[-30, 0, 30, 60])
    handles = [Line2D([], [], marker="o", color=c, label=label, markerfacecolor="white")
               for c, label in ((b.BLUE, "正常需求"), (b.GOLD, "1.25倍需求"))]
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(.59, .98), ncol=2, fontsize=9)
    save(fig, "图6-10", "不同源标定条件下MC-WM相对H2O+混合训练的配对时间成本差",
         "图6-9 / fig6_mcwm_alignment", [source],
         "空心点为3次训练各自5个共同评价日的配对均值；菱形为训练均值，线段为最小至最大值，不是置信区间。正值表示MC-WM成本较低。")


def model_usage():
    source = "experiments/mcwm_usage_v3/summary.json"
    data = load(source)
    methods = (("mcwm", "原MC-WM", b.GRAY, "o"),
               ("normalized", "归一化置信度MC-WM", b.BLUE, "s"),
               ("fixed_half", "固定0.5权重", b.GREEN, "^"),
               ("source_repair", "增加源价值目标修正", b.GOLD, "v"))
    fig, axes = b.plt.subplots(1, 2, figsize=(5.75, 4.9), sharex=True, sharey=True)
    fig.subplots_adjust(left=.22, right=.98, bottom=.13, top=.78, wspace=.16)
    for panel, (ax, demand) in enumerate(zip(axes, (1., 1.25))):
        for y, scene in enumerate(SCENES):
            if y % 2 == 0:
                ax.axhspan(y-.48, y+.48, color="#F3F5F6", zorder=0)
            for index, (method, _, color, marker) in enumerate(methods):
                rows = sorted((r for r in data["outcomes"] if
                               (r["scenario"], r["demand_scale"], r["arm"]) ==
                               (scene, demand, method)), key=lambda r: r["seed"])
                assert len(rows) == 2
                values = [-r["cost_minus_h2o"] for r in rows]
                paired_range(ax, values, y+(index-1.5)*.19, color, marker, spread=.035)
                for row, value in zip(rows, values):
                    record("图6-13", scene, demand, method, "training_mean", value, row["seed"])
        format_axis(ax, LABELS, (-45, 190), "H2O+减方法的成本 / 秒")
        ax.set(ylim=(2.6, -.6), xticks=[0, 80, 160])
        ax.set_title("(a) 正常需求" if panel == 0 else "(b) 1.25倍需求", loc="left", pad=10)
    handles = [Line2D([], [], color=c, marker=m, markerfacecolor="white", label=label)
               for _, label, c, m in methods]
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(.57, .985),
               ncol=2, fontsize=8.2, columnspacing=1.5, labelspacing=.8)
    save(fig, "图6-13", "不同源偏差下模型使用方式的配对时间成本差",
         "图6-12 / fig6_normalized_service", [source],
         "空心点为两次训练各自5个共同评价日的配对均值；菱形为两次训练均值，线段连接两次结果，不是置信区间。正值表示成本低于H2O+。")


def rule_comparison():
    source = "experiments/mcwm_usage_v3/rule_analysis.json"
    data = load(source)
    fig, axes = b.plt.subplots(1, 2, figsize=(5.75, 3.9), sharex=True, sharey=True)
    fig.subplots_adjust(left=.22, right=.98, bottom=.16, top=.79, wspace=.16)
    for panel, (ax, demand) in enumerate(zip(axes, (1., 1.25))):
        for y, scene in enumerate(SCENES):
            if y % 2 == 0:
                ax.axhspan(y-.48, y+.48, color="#F3F5F6", zorder=0)
            for method, color, offset in (("h2o_matched", b.GRAY, -.15), ("normalized", b.BLUE, .15)):
                rows = sorted((r for r in data["comparisons"] if
                               (r["scenario"], r["arm"], r["demand_scale"]) ==
                               (scene, method, demand)), key=lambda r: r["seed"])
                assert len(rows) == 2
                for index, row in enumerate(rows):
                    assert len(row["days"]) == 5
                    values = [day["cost_reduction"] for day in row["days"]]
                    np.testing.assert_allclose(np.mean(values), row["cost_reduction"])
                    ax.scatter(values, y+offset+(index-.5)*.10+np.linspace(-.025, .025, 5),
                               s=9, color=color, alpha=.3, linewidths=0, zorder=2)
                    for day, value in zip(row["days"], values):
                        record("图6-15", scene, demand, method, "evaluation_day", value, row["seed"], day["case"])
                    record("图6-15", scene, demand, method, "training_mean", row["cost_reduction"], row["seed"])
                paired_range(ax, [row["cost_reduction"] for row in rows], y+offset, color, spread=.05)
        format_axis(ax, LABELS, (-65, 180), "规则减方法的成本 / 秒")
        ax.set(ylim=(2.55, -.55), xticks=[-50, 0, 50, 100, 150])
        ax.set_title("(a) 正常需求" if panel == 0 else "(b) 1.25倍需求", loc="left", pad=10)
    handles = [Line2D([], [], color=c, marker="o", markerfacecolor="white", label=label)
               for c, label in ((b.GRAY, "H2O+"), (b.BLUE, "归一化置信度MC-WM"))]
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(.57, .985), ncol=2, fontsize=8.5)
    save(fig, "图6-15", "学习策略相对车头时距规则的配对时间成本差",
         "图6-14 / fig6_normalized_rule_comparison", [source],
         "浅色小点保留各次训练的5个同日配对差；空心点为两次训练均值，菱形为总体均值，线段连接两次训练均值，不是置信区间。正值表示学习方法成本低于规则。各方法沿用原共同执行设置。")


def time_decomposition():
    source = "experiments/mcwm_seed_confirmation_v13/completed_analysis.json"
    data = load(source)
    fig, ax = b.plt.subplots(figsize=(5.75, 3.4))
    fig.subplots_adjust(left=.23, right=.98, bottom=.18, top=.82)
    for y, scene in enumerate(SCENES):
        row = next(r for r in data["comparisons"] if
                   (r["scenario"], r["arm"], r["reference"]) == (scene, "fixed0250", "h2o_matched"))
        wait = 2*row["wait_delta_seconds"]["mean"]
        onboard = row["in_vehicle_delta_seconds"]["mean"]
        net = row["cost_delta_seconds"]["mean"]
        np.testing.assert_allclose(wait+onboard, net, atol=1e-8)
        ax.barh(y, wait, height=.31, color=b.GREEN)
        ax.barh(y, onboard, left=wait, height=.31, color=b.RED)
        ax.scatter(net, y, color=b.BLUE, marker="D", s=38, zorder=4)
        ax.text(wait/2, y, f"{wait:.2f}", color="white", ha="center", va="center", fontsize=9)
        ax.text(net-2, y-.27, f"{net:.2f}", color=b.BLUE, ha="right", va="center", fontsize=9)
        ax.text(wait+onboard/2, y+.27, f"+{onboard:.2f}", color=b.RED, ha="center", va="center", fontsize=8.5)
        for method, value in (("weighted_wait", wait), ("in_vehicle", onboard), ("net_cost", net)):
            record("图6-18", scene, "1.0_and_1.25", method, "twenty_training_mean", value)
    format_axis(ax, LABELS, (-110, 12), "MC-WM减无世界模型基线的时间成本 / 秒")
    ax.set(ylim=(2.55, -.55), xticks=[-100, -75, -50, -25, 0])
    handles = [Patch(color=b.GREEN, label="2×候车时间差"), Patch(color=b.RED, label="车内时间差"),
               Line2D([], [], color=b.BLUE, marker="D", ls="", label="净成本差")]
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(.58, .99), ncol=3, fontsize=8.5)
    save(fig, "图6-18", "模型系数0.25的MC-WM乘客时间分解",
         "图6-17 / fig6_confirmation_tradeoffs", [source],
         "20次训练均值；每次先等权平均两档需求、各5个共同测试日。候车时间差乘2，车内时间差权重为1；红色段从候车贡献的终点起算，蓝色菱形为两项之和。负值表示MC-WM成本较低。")


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    alignment()
    model_usage()
    rule_comparison()
    time_decomposition()
    with (OUT / "作图数据.csv").open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(DATA[0]))
        writer.writeheader()
        writer.writerows(DATA)
    (OUT / "图件来源与统计.json").write_text(json.dumps(MANIFEST, ensure_ascii=False, indent=2)+"\n", encoding="utf-8")
    paragraphs = ["# 第六章时间指标重绘", "沿用原实验结果，仅删除完成率展示；未改变执行设置，未更新Word，未覆盖旧图。",
                  "按重构稿图号导出600 dpi PNG、矢量PDF和保留文字的SVG。"]
    for row in MANIFEST:
        paragraphs.extend([f"## {row['figure']} {row['title']}", row["statistics"]])
    (OUT / "图注与替换对应.md").write_text("\n\n".join(paragraphs)+"\n", encoding="utf-8")


if __name__ == "__main__":
    main()
