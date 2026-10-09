"""Plot all-seed bus confirmation from compact server-side aggregates."""

import json

import numpy as np

import build_figures as b


SOURCE = "experiments/mcwm_seed_confirmation_v13/completed_analysis.json"
D = json.loads((b.ROOT / SOURCE).read_text())
SCENES = ("calibrated", "joint50", "joint75")
LABELS = ("正常标定", "联合50%失配", "联合75%失配")
METHODS = (
    ("fixed0250", "固定0.25", b.BLUE),
    ("fixed0300", "固定0.30", b.GREEN),
    ("shared_gated", "上下文调节", b.GOLD),
)


def comparison(scene, arm, reference):
    return next(r for r in D["comparisons"] if
                (r["scenario"], r["arm"], r["reference"]) ==
                (scene, arm, reference))


def draw_estimate(ax, row, y, color, metric="cost_delta_seconds",
                  seed_metric="cost_seconds"):
    values = np.array([r[seed_metric] for r in row["seed_deltas"]])
    offsets = np.linspace(-.12, .12, len(values))
    ax.scatter(values, y + offsets, s=13, facecolors="white",
               edgecolors=color, linewidths=.65, alpha=.8, zorder=3)
    estimate = row[metric]
    low, high = estimate["ci95"]
    ax.plot([low, high], [y, y], color=color, lw=2, zorder=4)
    ax.scatter(estimate["mean"], y, color=color, marker="D", s=25, zorder=5)


def seed_benefits():
    fig, axes = b.plt.subplots(3, 1, figsize=(5.75, 5.75), sharex=True)
    fig.subplots_adjust(left=.18, right=.985, bottom=.105, top=.94, hspace=.62)
    for panel, (ax, scene, label) in enumerate(zip(axes, SCENES, LABELS)):
        for y, (arm, _, color) in enumerate(METHODS):
            row = comparison(scene, arm, "h2o_matched")
            draw_estimate(ax, row, y, color)
            ax.text(123, y, f"{row['better_seeds']}/20改善", ha="right",
                    va="center", fontsize=8, color=color)
        ax.axvline(0, color=b.GRAY, lw=.8, ls="--")
        ax.grid(axis="x")
        ax.set(yticks=range(3), yticklabels=[m[1] for m in METHODS],
               ylim=(2.45, -.45), xlim=(-170, 130),
               xticks=[-150, -100, -50, 0, 50, 100])
        ax.set_title(f"({chr(97 + panel)}) {label}", loc="left", pad=9)
    axes[-1].set_xlabel("方法减无模型骨干的广义时间 / 秒")
    b.save(fig, "fig6_confirmation_seeds",
           "20个独立初始化下，固定0.25的相对收益随所测源失配增大。",
           [SOURCE],
           "空心点为全部20个训练种子的配对差；菱形和线段为均值及双侧95%配对t区间。每种子先等权平均10个固定测试情形。负值表示改善；固定数值指模型损失项系数，不是样本置信权重。")


def rule_controls():
    fig, axes = b.plt.subplots(1, 2, figsize=(5.75, 4.15), sharex=True, sharey=True)
    fig.subplots_adjust(left=.245, right=.975, bottom=.155, top=.79, wspace=.16)
    for ax, reference, title in zip(axes, ("daganzo_duet", "headway"),
                                    ("(a) Daganzo式规则", "(b) 车头时距规则")):
        for y, scene in enumerate(SCENES):
            for arm, offset, color in (("h2o_matched", -.13, b.GRAY),
                                        ("fixed0250", .13, b.BLUE)):
                draw_estimate(ax, comparison(scene, arm, reference), y + offset, color)
        ax.axvline(0, color=b.GRAY, lw=.8, ls="--")
        ax.grid(axis="x")
        ax.set(xlim=(-115, 60), xticks=[-100, -50, 0, 50],
               yticks=range(3), yticklabels=LABELS, ylim=(2.5, -.5),
               xlabel="方法减规则的广义时间 / 秒", title=title)
        ax.xaxis.label.set_fontsize(8.5)
    handles = [b.plt.Line2D([], [], marker="o", color=c, ls="", label=l)
               for c, l in ((b.GRAY, "无模型骨干"), (b.BLUE, "MC-WM固定0.25"))]
    fig.legend(handles=handles, loc="upper center", ncol=2,
               bbox_to_anchor=(.59, .985), fontsize=8.5)
    b.save(fig, "fig6_confirmation_rules",
           "固定0.25配置在三种源条件下均取得相对两种规则的乘客成本增量。",
           [SOURCE],
           "空心点为20个策略各自对同一组规则测试的成本差；菱形和线段为跨训练种子的均值及95%配对t区间。规则不随源条件重训；所有方法共用动作范围和完成保障。")


def passenger_tradeoffs():
    fig, axes = b.plt.subplots(1, 2, figsize=(5.75, 3.65), sharey=True)
    fig.subplots_adjust(left=.245, right=.98, bottom=.17, top=.76, wspace=.35)
    for y, scene in enumerate(SCENES):
        row = comparison(scene, "fixed0250", "h2o_matched")
        wait = 2 * row["wait_delta_seconds"]["mean"]
        onboard = row["in_vehicle_delta_seconds"]["mean"]
        net = row["cost_delta_seconds"]["mean"]
        axes[0].barh(y, wait, height=.3, color=b.GREEN)
        axes[0].barh(y, onboard, left=wait, height=.3, color=b.RED)
        axes[0].scatter(net, y, color=b.BLUE, marker="D", s=28, zorder=4)
        axes[0].text(net - 4, y - .27, f"{net:.1f}", ha="right", va="center",
                     fontsize=8, color=b.BLUE)
        draw_estimate(axes[1], row, y, b.BLUE,
                      metric="completion_delta_pp", seed_metric="completion_percent")
    for ax in axes:
        ax.axvline(0, color=b.GRAY, lw=.8, ls="--")
        ax.grid(axis="x")
        ax.set(yticks=range(3), yticklabels=LABELS, ylim=(2.6, -.6))
    axes[0].set(title="(a) 广义时间分解", xlabel="方法减骨干 / 秒",
                xlim=(-110, 10), xticks=[-100, -50, 0])
    axes[1].set(title="(b) 乘客完成率", xlabel="方法减骨干 / 百分点",
                xlim=(-.3, 1.0), xticks=[0, .4, .8])
    handles = [b.plt.Rectangle((0, 0), 1, 1, color=c, label=l) for c, l in
               ((b.GREEN, "两倍候车时间"), (b.RED, "乘车时间"))]
    handles.append(b.plt.Line2D([], [], color=b.BLUE, marker="D", ls="", label="净成本差"))
    fig.legend(handles=handles, loc="upper center", ncol=3,
               bbox_to_anchor=(.54, .985), fontsize=8)
    b.save(fig, "fig6_confirmation_tradeoffs",
           "候车时间减少抵消了较小的乘车时间增加，平均完成率同时提高。",
           [SOURCE],
           "左图以20种子均值分解两倍候车差与乘车差，红色段从候车项终点起算，菱形为净差。右图为空心种子点、均值及双侧95%配对t区间；每种子先平均共同测试情形。")


if __name__ == "__main__":
    seed_benefits()
    rule_controls()
    passenger_tradeoffs()
    path = b.OUT / "figure_sources.json"
    entries = {r["figure"]: r for r in json.loads(path.read_text())}
    entries.update({r["figure"]: r for r in b.MANIFEST})
    path.write_text(json.dumps(list(entries.values()), ensure_ascii=False, indent=2) + "\n")
