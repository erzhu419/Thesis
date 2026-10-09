"""Plot completed thesis comparisons without changing experiments.

Python quantitative panels, 146 mm wide, exported as PNG/PDF/SVG.
The bus model plot retains every condition and paired training seed.
Representation panels show seed variability; factorial intervals use five seeds.
"""

import json
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

import build_figures as b


SCENES = [("normal", "正常运行"), ("demand_surge", "客流激增"),
          ("fast_switch", "快速切换")]


def model_alignment(summary):
    fig, axes = plt.subplots(1, 2, figsize=(5.75, 4.15), sharey=True,
                             gridspec_kw={"width_ratios": [1.25, 1]})
    fig.subplots_adjust(left=.22, right=.97, bottom=.14, top=.82, wspace=.38)
    labels = []
    for i, (scene, label) in enumerate([
            ("calibrated", "正常标定"), ("joint25", "联合25%偏差"),
            ("joint50", "联合50%偏差")]):
        for j, (scale, color) in enumerate([("1.0", b.BLUE), ("1.25", b.GOLD)]):
            row = summary["results"][scene][scale]
            y = 2*i+j
            labels.append(label if j == 0 else "")
            time = [r["h2o_matched"]["cost"]-r["mcwm"]["cost"] for r in row["by_seed"]]
            completion = [100*(r["mcwm"]["completion"]-r["h2o_matched"]["completion"])
                          for r in row["by_seed"]]
            for ax, values in zip(axes, (time, completion)):
                if i % 2 == 0:
                    ax.axhspan(y-.48, y+.48, color="#F3F5F6", zorder=0)
                ax.plot([min(values), max(values)], [y, y], color=color, alpha=.35, lw=1)
                ax.scatter(values, y+np.array([-.11, 0, .11]), s=22,
                           facecolor="white", edgecolor=color, lw=.9, zorder=3)
                ax.scatter(np.mean(values), y, s=32, color=color, marker="D", zorder=4)
    for ax, title, xlabel in zip(axes, ("(a) 人均广义时间减少量", "(b) 完成率变化"),
                                ("H2O+减MC-WM / 秒", "MC-WM减H2O+ / 百分点")):
        ax.axvline(0, color=b.GRAY, ls="--", lw=.8)
        ax.set(yticks=np.arange(0, 6, 2)+.5,
               yticklabels=["正常标定", "联合25%偏差", "联合50%偏差"],
               ylim=(5.6, -.6), xlabel=xlabel)
        ax.set_title(title, loc="left", fontsize=9, pad=11)
        ax.tick_params(axis="y", length=0)
        ax.grid(axis="x")
    axes[0].set(xlim=(-40, 85), xticks=[-30, 0, 30, 60])
    axes[1].set(xlim=(-.32, .40), xticks=[-.2, 0, .2, .4])
    handles = [Line2D([], [], color=c, marker="o", markersize=4, lw=1, label=l)
               for c, l in [(b.BLUE, "正常需求"), (b.GOLD, "1.25倍需求")]]
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(.57, .99),
               ncol=2, fontsize=8.5)
    b.save(fig, "fig6_mcwm_alignment",
           "正常标定与中等联合偏差下模型辅助取得小幅平均收益，较大偏差下基本持平",
           [str(b.ROOT / "experiments/mcwm_alignment_v2/summary.json")],
           "3个训练种子；每种子每档需求平均5个配对SUMO测试日；空心点为种子差，菱形为均值，线段为种子范围而非置信区间；两个面板正值均有利于MC-WM")


def representation(summary):
    arms = ["learned_full", "masked_full", "onehot_full"]
    colors = [b.BLUE, b.GRAY, b.GOLD]
    fig, axes = plt.subplots(2, 3, figsize=(5.75, 4.8))
    fig.subplots_adjust(left=.14, right=.98, bottom=.10, top=.86, hspace=.4, wspace=.5)
    for j, (scene, label) in enumerate(SCENES):
        for i, metric in enumerate(("cost", "headway_imbalance")):
            ax = axes[i, j]
            rows = summary["results"][scene]["means"]
            values = np.array([list(rows[a][metric]["by_training_seed"].values()) for a in arms])
            for seed in range(5):
                ax.plot(range(3), values[:, seed], color="#CBD1D6", lw=.6, alpha=.8, zorder=1)
            for k, color in enumerate(colors):
                ax.scatter(np.full(5, k), values[k], s=13, color=color, alpha=.35, zorder=2)
                ax.scatter(k, values[k].mean(), s=32, color=color, marker="D", zorder=3)
            ax.set(xticks=range(3), xticklabels=["嵌入", "屏蔽", "独热"], xlim=(-.3, 2.3))
            ax.tick_params(axis="x", length=0, labelsize=8)
            ax.tick_params(axis="y", labelsize=8)
            ax.ticklabel_format(axis="y", style="plain", useOffset=False)
            ax.grid(axis="y")
            if i == 0:
                ax.set_title(label, fontsize=9.5, pad=9)
            if j == 0:
                ax.set_ylabel("人均广义时间 / 秒" if i == 0 else "前后时距不均衡 / 秒")
    fig.text(.14, .965, "(a) 乘客时间", fontsize=9.5)
    fig.text(.14, .48, "(b) 运行间隔", fontsize=9.5)
    b.save(fig, "fig3_representation_repeated",
           "五种子重复中学习嵌入相对独热编码同时降低乘客时间和时距不均衡",
           [str(b.ROOT / "experiments/enhancement_v1/summary.json")],
           "各组正常条件训练100回合；每个小点为同一训练种子在3个共同测试日的均值，细线连接相同种子，大菱形为5种子均值；各面板按实际指标范围设置纵轴")


def disagreement(summary):
    fig, axes = plt.subplots(1, 2, figsize=(5.75, 3.85), sharey=True)
    fig.subplots_adjust(left=.18, right=.97, bottom=.18, top=.80, wspace=.38)
    factors = [("actor_on_minus_off", "行动者分歧项", b.BLUE, -.15, "o"),
               ("critic_on_minus_off", "评论家分歧项", b.GOLD, .15, "s")]
    for ax, metric, factor, title, xlabel in zip(
            axes, ("reward", "headway_imbalance"), (1e-3, -1.),
            ("(a) 累计奖励增量", "(b) 时距不均衡减少量"),
            ("启用减关闭 / 千", "关闭减启用 / 秒")):
        for y, (scene, _) in enumerate(SCENES):
            for key, _, color, offset, marker in factors:
                row = summary["results"][scene]["paired_differences"][key][metric]
                mean = factor*row["mean"]
                low, high = sorted(factor*np.array(row["ci95"]))
                ax.errorbar(mean, y+offset, xerr=[[mean-low], [high-mean]],
                            color=color, fmt=marker, ms=4, lw=1.2, capsize=3)
        ax.axvline(0, color=b.GRAY, ls="--", lw=.8)
        ax.set(yticks=range(3), yticklabels=[v for _, v in SCENES], ylim=(2.5, -.5), xlabel=xlabel)
        ax.set_title(title, loc="left", fontsize=9.5, pad=12)
        ax.tick_params(axis="y", length=0)
        ax.grid(axis="x")
    axes[0].set(xlim=(-8, 30), xticks=[0, 10, 20, 30])
    axes[1].set(xlim=(-4, 10), xticks=[-4, 0, 4, 8])
    fig.legend(handles=[Line2D([], [], color=c, marker=m, lw=1, ms=4, label=l)
                        for _, l, c, _, m in factors],
               loc="upper center", bbox_to_anchor=(.56, .99), ncol=2, fontsize=8.5)
    b.save(fig, "fig3_disagreement_factorial",
           "快速切换下行动者分歧项同时改善奖励和时距不均衡，评论家项的独立平均作用较小",
           [str(b.ROOT / "experiments/enhancement_v1/summary.json")],
           "2×2比较中先在另一因子的两种设置上等权平均启用减关闭差，再汇总5个训练种子；误差线为自由度4的描述性95% t区间，未作多重比较校正；正值表示改善")


def main():
    alignment = json.loads((b.ROOT / "experiments/mcwm_alignment_v2/summary.json").read_text())
    legacy = json.loads((b.ROOT / "experiments/enhancement_v1/summary.json").read_text())["legacy"]
    assert alignment["complete"] and alignment["evaluation_complete"] == 180
    assert legacy["complete"] and len(legacy["training_seeds"]) == 5
    model_alignment(alignment)
    representation(legacy)
    disagreement(legacy)
    (b.DATA / "completed_study_values.json").write_text(json.dumps(
        dict(mcwm_alignment=alignment, representation_and_disagreement=legacy),
        ensure_ascii=False, indent=2)+"\n")
    path = b.OUT / "figure_sources.json"
    manifest = json.loads(path.read_text())
    names = {row["figure"] for row in b.MANIFEST}
    manifest = [row for row in manifest if row["figure"] not in names]+b.MANIFEST
    path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2)+"\n")


if __name__ == "__main__":
    main()
