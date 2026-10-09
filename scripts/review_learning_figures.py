"""Chinese thesis figures: repeated ensemble tradeoffs and model mechanism contrasts.

Quantitative grids, 146 mm wide, all training seeds visible. Export editable
SVG/vector PDF and 360 dpi PNG; statistics use seed-level means, not pooled days.
"""

import json
import numpy as np
import build_figures as b

D = json.loads((b.ROOT / "experiments/review_round3/learning_summary.json").read_text())
SOURCE = ["experiments/review_round3/learning_summary.json"]
SCENARIOS = ["calibrated", "joint25", "joint50"]
LABELS = ["正常标定", "联合25%偏差", "联合50%偏差"]


def ensemble():
    fig, axes = b.plt.subplots(2, 2, figsize=(5.75, 5.6))
    fig.subplots_adjust(left=.13, right=.98, bottom=.10, top=.88, wspace=.42, hspace=.54)
    sizes = [2, 5, 10, 20]
    for ax, key, title, ylabel in zip(axes[0], ["front", "time"],
            ["(a) 计划间隔偏差", "(b) 乘客时间"], ["前向绝对偏差 / 秒", "人均广义时间 / 秒"]):
        for scene, color, offset, label in [("normal",b.BLUE,-.09,"常规运行"),("fast_switch",b.GOLD,.09,"快速切换")]:
            for seed in [32101,32102,32103,32104,32105]:
                rows = [next(r for r in D["ensemble"]["cells"] if (r["scene"],r["seed"],r["size"])==(scene,seed,k)) for k in sizes]
                ax.plot(np.arange(4)+offset,[r[key] for r in rows], color=color,alpha=.25,lw=.65,marker="o",ms=2.5)
            groups = [next(r for r in D["ensemble"]["groups"] if (r["scene"],r["size"])==(scene,k)) for k in sizes]
            ax.plot(np.arange(4)+offset,[r[key]["mean"] for r in groups],color=color,marker="D",ms=4,label=label)
        ax.set(title=title, ylabel=ylabel, xticks=range(4), xticklabels=sizes, xlabel="评论家数量")
        ax.grid(axis="y")
    fig.legend(*axes[0,0].get_legend_handles_labels(),loc="upper center",ncol=2,fontsize=8.5,bbox_to_anchor=(.53,.99))
    for ax,key,title,ylabel in zip(axes[1],["minutes","gpu_mb"],
            ["(c) 训练耗时", "(d) 训练显存"],["100回合训练 / 分钟", "峰值已分配显存 / MiB"]):
        for seed in [32101,32102,32103,32104,32105]:
            rows = [next(r for r in D["ensemble"]["training"] if (r["seed"],r["size"])==(seed,k)) for k in sizes]
            ax.plot(range(4),[r[key] for r in rows],color=b.GRAY,alpha=.45,lw=.65,marker="o",ms=3)
        ax.plot(range(4),[np.mean([r[key] for r in D["ensemble"]["training"] if r["size"]==k]) for k in sizes],color=b.GREEN,marker="D",ms=4)
        ax.set(title=title, ylabel=ylabel, xticks=range(4), xticklabels=sizes,xlabel="评论家数量")
        ax.grid(axis="y")
    b.save(fig,"fig3_ensemble_repeated","扩大集成没有持续收益，训练显存随规模上升。",SOURCE,
           "5训练种子；每场景3评价日先在种子内平均。细线为配对种子，菱形为均值；不绘置信带。")


def model_service():
    fig, axes = b.plt.subplots(1, 2, figsize=(5.75,4.15),sharey=True)
    fig.subplots_adjust(left=.25,right=.98,top=.85,bottom=.16,wspace=.20)
    order = [(s,d) for s in SCENARIOS for d in (1.,1.25)]
    for ax,reference,title,color in zip(axes,["h2o_matched","scaled_nomodel"],
            ["(a) 相对H2O+", "(b) 相对0.75损失组"],[b.BLUE,b.GREEN]):
        for y,(scenario,demand) in enumerate(order):
            row = next(r for r in D["mcwm"]["contrasts"] if (r["scenario"],r["demand"],r["arm"],r["reference"])==(scenario,demand,"mcwm",reference))
            values, mean, ci = row["time"]["values"],row["time"]["mean"],row["time"]["ci95"]
            ax.plot(ci,[y,y],color=color,lw=1.)
            ax.scatter(values,np.array([y-.1,y,y+.1]),s=19,facecolors="white",edgecolors=color,zorder=3)
            ax.scatter([mean],[y],s=32,marker="D",color=color,zorder=4)
        ax.axvline(0,color=b.GRAY,lw=.8,ls="--")
        ax.set(title=title,xlabel="MC-WM减参照 / 秒",xlim=(-280,280),xticks=[-200,0,200])
        ax.grid(axis="x")
    axes[0].set(yticks=range(6),yticklabels=[f"{LABELS[SCENARIOS.index(s)]}\n需求 {d:.2f}倍" for s,d in order],ylim=(5.55,-.55))
    fig.text(.63,.945,"人均广义时间：负值表示MC-WM较低",ha="center",fontsize=9)
    b.save(fig,"fig6_model_coefficient_control","模型相对两种参照的增量随标定条件改变，配对区间均跨零。",SOURCE,
           "3训练种子，每种子5共同测试日；空心点为种子差，菱形为均值，细线为未校正的双侧95% t区间。")


def model_mechanism():
    fig, axes = b.plt.subplots(2,2,figsize=(5.75,5.6))
    fig.subplots_adjust(left=.13,right=.98,top=.88,bottom=.10,wspace=.48,hspace=.56)
    panels = [("error","(a) 未加权预测误差","复合预测损失",False),
              ("confidence","(b) 局部置信","局部置信均值",False),
              ("weight","(c) 累积可信度","模型权重均值",True),
              ("gradient_ratio","(d) 模型梯度贡献","加权梯度范数比",True)]
    for ax,(key,title,ylabel,log) in zip(axes.flat,panels):
        for scenario,label,color in zip(SCENARIOS,LABELS,[b.BLUE,b.GOLD,b.GREEN]):
            rows = [r for r in D["mcwm"]["diagnostics"] if r["scenario"]==scenario]
            updates = sorted({r["update"] for r in rows})
            x = (np.asarray(updates)-50000)/1000
            values = np.array([[next(r[key] for r in rows if r["seed"]==seed and r["update"]==u) for u in updates] for seed in [6201,6202,6203]])
            for values_seed in values:
                ax.plot(x,values_seed,color=color,alpha=.23,lw=.6)
            ax.plot(x,values.mean(0),color=color,label=label)
        ax.set(title=title,ylabel=ylabel,xlabel="新增策略更新 / 千次",xticks=[1,5,10,15,20])
        if log:
            ax.set_yscale("log")
        ax.grid(axis="y")
    axes[0,1].set_ylim(.44,.54)
    fig.legend(*axes[0,0].get_legend_handles_labels(),loc="upper center",ncol=3,fontsize=8.5,bbox_to_anchor=(.53,.99))
    b.save(fig,"fig6_model_trust_diagnostics","预测误差和局部置信相对稳定，累积权重与模型梯度明显减弱。",SOURCE,
           "各条件3训练种子，细线为个体、粗线为算术均值。误差与置信来自同一校准块512行；梯度来自实际训练批。")


if __name__ == "__main__":
    ensemble()
    model_service()
    model_mechanism()
    path = b.OUT / "figure_sources.json"
    entries = {r["figure"]:r for r in json.loads(path.read_text())}
    entries.update({r["figure"]:r for r in b.MANIFEST})
    path.write_text(json.dumps(list(entries.values()),ensure_ascii=False,indent=2)+"\n")
