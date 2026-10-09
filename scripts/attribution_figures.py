"""Two Chinese attribution figures, reusing the thesis Python export style.

The paired forest separates learning-source contrasts from execution changes.
The decomposition and temporal heatmap show the guard's cross-policy effect.
Both are quantitative grids at 146 mm width, with editable SVG/PDF and PNG.
Intervals use six paired environment blocks, not independent training seeds.
"""

import json
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

import build_figures as b


HERE = b.ROOT / "experiments/sumo_eval/attribution_v1"
METHODS = ["hybrid", "offline", "source", "zero", "headway", "welfare"]
LABELS = dict(hybrid="混合学习", offline="纯离线", source="纯源学习",
              zero="零停站建议", headway="间隔规则", welfare="乘客福利规则")
LAYERS = [("no_floor", "无保障下限", b.GRAY, "o"),
          ("late20", "末段20秒", b.GOLD, "s"),
          ("early60", "提前60秒", b.BLUE, "D")]


def contrast(summary, name, metric):
    row = next(c for c in summary["comparisons"]
               if c["name"] == name and c["demand_scale"] == "all_equal_weight")
    return row["metrics"][metric]


def forest(summary):
    fig, axes = plt.subplots(1, 2, figsize=(5.75, 3.65), sharey=True)
    fig.subplots_adjust(left=.23, right=.97, bottom=.24, top=.87, wspace=.27)
    comparisons = METHODS[1:]
    for ax, metric, title, xlabel in zip(
            axes, ("cost_seconds", "completion_percentage_points"),
            ("(a) 人均广义时间差", "(b) 完成率差"),
            ("混合减参照 / 秒", "混合减参照 / 百分点")):
        for row, method in enumerate(comparisons):
            if row % 2 == 0:
                ax.axhspan(row - .47, row + .47, color="#F4F6F7", zorder=0)
            for offset, (layer, _, color, marker) in zip((-.24, 0, .24), LAYERS):
                value = contrast(summary, f"hybrid-minus-{method}/{layer}", metric)
                mean = value["mean_difference"]
                low, high = value["approximate_95pct_t_interval"]
                points = list(value["paired_block_differences"].values())
                ax.scatter(points, np.full(6, row + offset), s=8, color=color, alpha=.28, zorder=2)
                ax.errorbar(mean, row + offset, xerr=[[mean-low], [high-mean]],
                            fmt=marker, color=color, capsize=2, ms=4, lw=1, zorder=3)
        ax.axvline(0, color="#434950", lw=.8, ls="--")
        ax.set(yticks=range(5), yticklabels=[LABELS[m] for m in comparisons],
               ylim=(4.55, -.55), xlabel=xlabel)
        ax.set_title(title, fontsize=9.5, loc="left", pad=12)
        ax.grid(axis="x")
        ax.tick_params(axis="y", length=0, labelsize=9)
    axes[0].set(xlim=(-40, 18), xticks=[-40, -20, 0])
    axes[1].set(xlim=(-.115, .195), xticks=[-.1, 0, .1])
    handles = [Line2D([], [], color=color, marker=marker, lw=1, ms=4, label=label)
               for _, label, color, marker in LAYERS]
    fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(.55, .012),
               ncol=3, fontsize=8.2, handlelength=1.5, columnspacing=1.4)
    b.save(fig, "fig6_learning_execution_attribution",
           "混合策略优于纯离线续训，相对纯源和同执行层间隔规则的时间差尚未分辨",
           [str(HERE / "summary.json")],
           "1个训练种子、6个配对环境块；每块先等权平均3档需求；浅点为块差，实心标记为均值，横线为自由度5的近似95% t区间；未作多重校正；时间差负值较好，完成率差正值较好")


def guard(summary, rows):
    fig = plt.figure(figsize=(5.75, 5.5))
    ax = fig.add_axes([.23, .53, .73, .36])
    net = []
    for i, method in enumerate(METHODS):
        name = method + "/early60-minus-no_floor"
        waiting = 2 * contrast(summary, name, "waiting_seconds")["mean_difference"]
        riding = contrast(summary, name, "riding_seconds")["mean_difference"]
        effect = contrast(summary, name, "cost_seconds")
        mean = effect["mean_difference"]
        low, high = effect["approximate_95pct_t_interval"]
        assert np.isclose(waiting + riding, mean)
        ax.barh(i, waiting, height=.55, color=b.BLUE, alpha=.78)
        ax.barh(i, riding, height=.55, color=b.GOLD, alpha=.8)
        ax.errorbar(mean, i, xerr=[[mean-low], [high-mean]], fmt="D", ms=3.5,
                    color="#282E33", capsize=2, lw=1, zorder=3)
        ax.text(31, i, f"{mean:.1f}", va="center", fontsize=8.5)
        net.append(mean)
    ax.axvline(0, color=b.GRAY, lw=.8)
    ax.set(yticks=range(6), yticklabels=[LABELS[m] for m in METHODS], ylim=(5.6, -.6),
           xlim=(-145, 57), xticks=[-120, -80, -40, 0, 40], xlabel="提前保障减无下限 / 秒")
    ax.set_title("(a) 候车收益、车内代价与净时间变化", loc="left", fontsize=9.5, pad=12)
    ax.tick_params(axis="y", length=0, labelsize=9)
    ax.grid(axis="x")
    handles = [Line2D([], [], color=b.BLUE, lw=5, label="2 × 候车时间差"),
               Line2D([], [], color=b.GOLD, lw=5, label="车内时间差"),
               Line2D([], [], color="#282E33", marker="D", lw=1, ms=3.5, label="净差与区间")]
    fig.legend(handles=handles, ncol=3, loc="upper center", bbox_to_anchor=(.56, .995),
               fontsize=8, columnspacing=1.4, handlelength=1.4)

    ax = fig.add_axes([.23, .145, .62, .245])
    bins = np.arange(0, 24000, 1800)
    rates = np.full((6, len(bins)), np.nan)
    for i, method in enumerate(METHODS):
        selected = [r for r in rows if r["method"] == method and r["execution_layer"] == "early60"]
        for j, start in enumerate(bins):
            counts = [r["execution_bins"].get(str(start), {}) for r in selected]
            fractions = [100*c.get("floor_changed_decisions", 0)/c["decisions"]
                         for c in counts if c.get("decisions", 0)]
            if fractions:
                rates[i, j] = np.mean(fractions)
    cmap = plt.get_cmap("Blues").copy()
    cmap.set_bad("#DDE1E4")
    mesh = ax.imshow(rates, cmap=cmap, vmin=0, vmax=15, aspect="auto",
                     extent=[0, 7, 5.5, -.5], interpolation="nearest")
    ax.set(yticks=range(6), yticklabels=[LABELS[m] for m in METHODS],
           xlim=(0, 24000/3600), xticks=[0, 1, 2, 3, 4, 5, 6], xlabel="自运营开始的时间 / 小时")
    ax.tick_params(axis="both", length=0, labelsize=8.5)
    ax.set_title("(b) 保障实际增加停站的决策比例", loc="left", fontsize=9.5, pad=13)
    ax.axvline(1, color="#777777", ls=":", lw=.7)
    ax.axvline(5, color="#777777", ls=":", lw=.7)
    color_ax = fig.add_axes([.89, .145, .022, .245])
    colorbar = fig.colorbar(mesh, cax=color_ax, ticks=[0, 5, 10, 15])
    colorbar.ax.set_title("%", fontsize=8.5, pad=7)
    b.save(fig, "fig6_guard_attribution",
           "保障在各骨干中均以候车减少抵偿车内时间增加，触发具有共同的时段结构",
           [str(HERE / "summary.json"), str(HERE / "runs")],
           "上图先块内平均3档需求，再对6块求均值及近似95% t区间；下图各半小时先求单回合触发比例，再平均有决策的回合；灰色表示该时段无决策；时间单位从运营开始计，5小时对应75%时域；不作新增服务人数的个体因果归因")
    return [[None if np.isnan(value) else float(value) for value in row] for row in rates]


def main():
    summary = json.loads((HERE / "summary.json").read_text())
    rows = [json.loads(f.read_text()) for f in sorted(HERE.glob("runs/shard*/results/case*.json"))]
    forest(summary)
    rates = guard(summary, rows)
    (b.DATA / "learning_execution_attribution.json").write_text(json.dumps(
        dict(comparisons=summary["comparisons"], methods=METHODS,
             half_hour_floor_changed_percent=rates), ensure_ascii=False, indent=2) + "\n")
    path = b.OUT / "figure_sources.json"
    manifest = json.loads(path.read_text())
    replaced = {f["figure"] for f in b.MANIFEST}
    manifest = [f for f in manifest if f["figure"] not in replaced] + b.MANIFEST
    path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")


if __name__ == "__main__":
    main()
