"""Paired bus benefits and confidence-scale diagnostics at thesis column width."""

import json
import numpy as np
import build_figures as b

ROOT = b.ROOT / "experiments/mcwm_usage_v3"
D = json.loads((ROOT / "summary.json").read_text())
SCENES = ["calibrated", "joint50", "joint75"]
LABELS = ["正常标定", "联合50%失配", "联合75%失配"]
METHODS = [("mcwm", "原MC-WM", b.GRAY), ("normalized", "归一化置信", b.BLUE),
           ("fixed_half", "固定0.5权重", b.GREEN), ("source_repair", "加源目标修正", b.GOLD)]
SOURCES = ["experiments/mcwm_usage_v3/summary.json", "experiments/mcwm_usage_v3/analysis.json"]


def service():
    fig, axes = b.plt.subplots(1, 2, figsize=(5.75, 5.55), sharey=True)
    fig.subplots_adjust(left=.24, right=.975, bottom=.105, top=.86, wspace=.24)
    order = [(s, d) for s in SCENES for d in (1., 1.25)]
    for method_index, (arm, label, color) in enumerate(METHODS):
        offset = (method_index-1.5)*.16
        for y, (scene, demand) in enumerate(order):
            rows = sorted([r for r in D["outcomes"] if
                           (r["scenario"], r["demand_scale"], r["arm"]) == (scene, demand, arm)], key=lambda r:r["seed"])
            for ax, key, sign in zip(axes, ["cost_minus_h2o", "completion_pp_minus_h2o"], [-1, 1]):
                values = np.array([sign*r[key] for r in rows])
                ax.plot(values, [y+offset]*2, color=color, lw=.7)
                ax.scatter(values, [y+offset-.025, y+offset+.025], s=13, facecolors="white", edgecolors=color, zorder=3)
                ax.scatter(values.mean(), y+offset, s=21, marker="D", color=color,
                           label=label if y == 0 and ax is axes[0] else None, zorder=4)
    for ax in axes:
        ax.axvline(0, color=b.GRAY, lw=.7, ls="--")
        ax.grid(axis="x")
    axes[0].set(title="(a) 广义时间减少", xlabel="H2O+减方法 / 秒", xlim=(-45, 190), xticks=[0, 80, 160],
                yticks=range(6), yticklabels=[f"{LABELS[SCENES.index(s)]}\n需求 {d:.2f}倍" for s,d in order], ylim=(5.55, -.55))
    axes[1].set(title="(b) 完成率提高", xlabel="方法减H2O+ / 百分点", xlim=(-.28, 1.0), xticks=[0, .4, .8])
    fig.legend(*axes[0].get_legend_handles_labels(), loc="upper center", ncol=2, fontsize=8.5,
               bbox_to_anchor=(.58, .997), columnspacing=1.4)
    b.save(fig, "fig6_normalized_service", "归一化置信在所测条件的两个训练种子中均改善乘客时间与完成率。", SOURCES,
           "2训练种子，每种子5共同评价日；空心点为种子均值差，菱形为均值，线段连接两次训练，不是置信区间。")


def mechanism():
    fig, axes = b.plt.subplots(2, 2, figsize=(5.75, 5.8))
    fig.subplots_adjust(left=.135, right=.98, top=.865, bottom=.09, wspace=.42, hspace=.57)
    c = np.linspace(0, 1, 201)
    ax = axes[0,0]
    for values, label, color, style in [(c**10, "原MC-WM", b.GRAY, "-"),
            (c, "归一化置信", b.BLUE, "-"), (np.full_like(c,.5), "固定0.5权重", b.GREEN, "--")]:
        ax.plot(c, values, label=label, color=color, ls=style)
    ax.set(title="(a) 恒定置信的固定点", xlabel="恒定局部置信", ylabel="模型权重", xlim=(0,1), ylim=(0,1), xticks=[0,.5,1])
    for arm, _, color in METHODS[:3]:
        values = []
        for scene in SCENES:
            for seed in (6201, 6202):
                rows = [json.loads(p.read_text()) for p in sorted((ROOT/"runs"/scene/f"{arm}_seed{seed}"/"results/diagnostics").glob("update*.json"))]
                values.append(np.array([[r["model"]["calibrate"]["actual_weights"]["mean"],
                    100*r["scaled_model_gradient_norm"]/r["scaled_real_gradient_norm"],
                    r["model"]["calibrate"]["unweighted_prediction_loss"]["mean"]] for r in rows]))
        data = np.array(values)
        data[:,:,2] /= data[:,0:1,2]
        x = np.arange(1,21)
        for panel, ax in enumerate([axes[0,1], axes[1,0], axes[1,1]]):
            for run in data:
                ax.plot(x,run[:,panel],color=color,alpha=.16,lw=.5,ls="--" if arm == "fixed_half" else "-")
            ax.plot(x,data[:,:,panel].mean(0),color=color,lw=1.4,ls="--" if arm == "fixed_half" else "-")
            ax.set(xlabel="新增更新 / 千次",xticks=[1,10,20],xlim=(1,20))
    axes[0,1].set(title="(b) 实际模型权重",ylabel="权重均值",yscale="log")
    axes[1,0].set(title="(c) 模型梯度贡献",ylabel="梯度范数比 / %",yscale="log")
    axes[1,1].set(title="(d) 相对预测误差",ylabel="损失 / 首次诊断损失")
    for ax in axes.flat:
        ax.grid(axis="y")
    fig.legend(*axes[0,0].get_legend_handles_labels(),loc="upper center",ncol=3,fontsize=8.2,bbox_to_anchor=(.52,.99))
    b.save(fig,"fig6_normalized_mechanism","归一化恢复模型权重与梯度贡献，预测误差并未同步大幅改变。",SOURCES,
           "(a)为无终止且局部置信恒定的解析固定点；其余为3源条件各2训练种子。细线为每次训练，粗线为等权均值；梯度计入损失系数。")


def rules():
    data = json.loads((ROOT/"rule_analysis.json").read_text())
    fig, axes = b.plt.subplots(1, 2, figsize=(5.75, 3.65), sharex=True, sharey=True)
    fig.subplots_adjust(left=.13, right=.985, bottom=.16, top=.72, wspace=.13)
    for ax, demand in zip(axes, (1., 1.25)):
        for scene, marker in zip(SCENES, ("o", "s", "^")):
            for arm, color in (("h2o_matched", b.GRAY), ("normalized", b.BLUE)):
                rows = [r for r in data["comparisons"] if
                        (r["scenario"], r["arm"], r["demand_scale"]) == (scene, arm, demand)]
                points = [day for row in rows for day in row["days"]]
                ax.scatter([p["cost_reduction"] for p in points],
                           [p["completion_pp_gain"] for p in points],
                           marker=marker, color=color, s=20, alpha=.7,
                           edgecolors="white", linewidths=.3, zorder=3)
        ax.axvline(0, color=b.GRAY, ls="--", lw=.7)
        ax.axhline(0, color=b.GRAY, ls="--", lw=.7)
        ax.set(xlim=(-65, 180), ylim=(-.45, .9), xticks=[-50, 0, 50, 100, 150],
               xlabel="规则减方法的广义时间 / 秒")
        ax.grid(alpha=.2)
    axes[0].set(title="(a) 正常需求", ylabel="方法减规则的完成率 / 百分点")
    axes[1].set_title("(b) 1.25倍需求")
    method_handles = [b.plt.Line2D([], [], color=c, marker="o", ls="", label=l)
                      for c,l in [(b.GRAY,"H2O+"),(b.BLUE,"归一化MC-WM")]]
    scene_handles = [b.plt.Line2D([], [], color=b.GRAY, marker=m, ls="", label=l)
                     for m,l in zip(("o","s","^"),LABELS)]
    fig.legend(handles=method_handles, loc="upper center", ncol=2, bbox_to_anchor=(.54,1.0), fontsize=8.5)
    fig.legend(handles=scene_handles, loc="upper center", ncol=3, bbox_to_anchor=(.54,.91), fontsize=8.0)
    b.save(fig,"fig6_normalized_rule_comparison","归一化MC-WM在共同完成保障下超过车头时距规则，H2O+随条件改变方向。",
           SOURCES+["experiments/mcwm_usage_v3/rule_analysis.json"],
           "每点为一个固定策略与同日规则的差；3源条件、各2训练种子、各5评价日。所有方法均含共同完成保障；右上象限表示两指标均改善，不作独立60次训练推断。")


if __name__ == "__main__":
    service()
    mechanism()
    rules()
    path = b.OUT/"figure_sources.json"
    entries = {r["figure"]:r for r in json.loads(path.read_text())}
    entries.update({r["figure"]:r for r in b.MANIFEST})
    path.write_text(json.dumps(list(entries.values()),ensure_ascii=False,indent=2)+"\n")
