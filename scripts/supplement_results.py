"""Aggregate completed bus experiments and render thesis figures in Chinese."""

import csv
import json
from collections import defaultdict
from itertools import product
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm

import build_figures as b


EXP = b.ROOT / "experiments"
SCENES = {"normal": "正常运行", "congestion_severe": "持续拥堵",
          "demand_surge": "持续客流增加", "slow_switch": "慢速切换", "fast_switch": "快速切换"}
MISMATCH = {"calibrated": "已标定参照", "speed25": "速度降低25%",
            "od25": "OD强度增加25%", "signals25": "额外路口密度25%",
            "dwell25": "停站时间增加25%", "joint25": "联合偏差25%",
            "joint50": "联合偏差50%", "mechanistic": "去除经验运行时间标定"}
COLORS = {"h2o": b.BLUE, "mcwm": b.GREEN, "offline": b.GRAY, "online": b.RED}
NAMES = {"h2o": "H2O+混合续训", "mcwm": "世界模型辅助", "offline": "纯离线续训",
         "online": "纯源仿真训练"}


def read(path):
    return json.loads(path.read_text())


def write_csv(name, rows):
    with (b.DATA / f"{name}.csv").open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def finish(fig, name, conclusion, sources, statistics):
    b.save(fig, name, conclusion, sources, statistics)


def scalar_bus(d):
    n = d["generated_passengers"]
    return {k: v for k, v in d.items() if not isinstance(v, (list, dict))} | {
        "completion_percent": 100*d["arrived_passengers"]/n,
        "wait_seconds_per_passenger": d["wait_passenger_seconds"]/n,
        "ride_seconds_per_passenger": d["ride_passenger_seconds"]/n}


def representation():
    root = EXP / "legacy_bus/runs/representation_v1"
    methods = ["learned", "masked", "onehot"]
    labels = ["可学习嵌入", "屏蔽类别输入", "独热输入"]
    rows, params = [], {}
    for method, seed in product(methods, (16001, 16002)):
        directory = root / f"{method}_seed{seed}" / "results"
        assert read(directory / "train_summary.json")["completed_episodes"] == 100
        params[method] = read(directory / "parameters.json")
        for scene, evaluation_seed in product(("normal", "fast_switch"), (15001, 15002, 15003)):
            d = read(directory / f"{scene}_seed{evaluation_seed}.json")
            assert d["horizon_seconds"] == 24000
            assert len(d["mode_history"]) == (1 if scene == "normal" else 14)
            rows.append(scalar_bus(d))
    write_csv("categorical_paired_results", rows)
    fig, axes = plt.subplots(1, 2, figsize=(5.75, 3.65), sharey=True,
                             gridspec_kw={"width_ratios":[1.45,1]}, layout="constrained")
    means = []
    positions=[]; ticklabels=[]
    for j, scene in enumerate(("normal", "fast_switch")):
        for i, method in enumerate(methods):
            y=4*j+i;positions.append(y);ticklabels.append(labels[i])
            group = [r for r in rows if r["method"] == method and r["scene"] == scene]
            wait=2*np.mean([r["wait_seconds_per_passenger"] for r in group])
            ride=np.mean([r["ride_seconds_per_passenger"] for r in group])
            assert np.isclose(wait+ride,np.mean([r["generalized_seconds_per_passenger"] for r in group]))
            axes[0].barh(y,wait,height=.57,color=b.BLUE,label="候车贡献" if y==0 else None)
            axes[0].barh(y,ride,left=wait,height=.57,color="#B6CCC8",label="车内贡献" if y==0 else None)
            costs=[np.mean([r["generalized_seconds_per_passenger"] for r in group if r["training_seed"]==seed]) for seed in (16001,16002)]
            axes[0].plot(costs,[y,y],color="#333333",lw=.8,zorder=4)
            axes[0].scatter(costs,[y-.07,y+.07],s=16,facecolor="white",edgecolor="#333333",lw=.6,zorder=5)
            vals=[np.mean([r["reward"] for r in group if r["training_seed"]==seed])/1000 for seed in (16001,16002)]
            axes[1].plot(vals,[y,y],color=b.GRAY,lw=1)
            axes[1].scatter(vals,[y-.07,y+.07],s=23,color=b.BLUE if i==0 else b.GRAY,zorder=3)
            means.append(dict(method=method, scene=scene, cost=np.mean([r["generalized_seconds_per_passenger"] for r in group]),
                              reward=np.mean([r["reward"] for r in group]),
                              completion_percent=np.mean([r["completion_percent"] for r in group])))
    for ax in axes:
        ax.set(yticks=positions,yticklabels=ticklabels,ylim=(6.7,-1.2))
        ax.axhline(3,color="#BDC5CB",lw=.7)
        ax.text(.04,.97,"正常运行",transform=ax.transAxes,fontsize=8,color=b.GRAY,va="top")
        ax.text(.04,.46,"快速切换",transform=ax.transAxes,fontsize=8,color=b.GRAY,va="top")
        ax.grid(axis="x")
    axes[0].set(title="(a) 乘客时间构成",xlabel="人均广义时间 / 秒")
    axes[1].set(title="(b) 累计奖励",xlabel="奖励 / 千")
    axes[1].tick_params(axis="y",left=False,labelleft=False)
    axes[0].legend(loc="lower right",bbox_to_anchor=(1,-.29),ncol=2,fontsize=7.5)
    finish(fig, "fig3_categorical_ablation", "共同训练配置下，可学习类别表征改善奖励与乘客时间",
           [str(root.relative_to(b.ROOT))], "2个训练种子，各评价3次；条形为乘客时间均值，候车乘2、车内乘1；时间空心点与奖励实心点均先平均各训练种子的3次评价，线连接两个训练重复")
    return dict(means=means, parameters=params, evaluations=len(rows))


def legacy():
    root = EXP / "legacy_bus/runs/paired_v1"
    rows, raw = [], {}
    for method, scene, seed in product(("resac", "bapr", "no_hold", "headway"), SCENES, (15001, 15002, 15003)):
        d = read(root / f"{method}_{scene}/{method}_{scene}_seed{seed}.json")
        assert d["horizon_seconds"] == 24000
        raw[method, scene, seed] = d
        rows.append(scalar_bus(d))
    write_csv("legacy_bus_paired_results", rows)
    fig, axes = plt.subplots(1, 2, figsize=(5.75, 3.2), layout="constrained")
    means = []
    for i, scene in enumerate(SCENES):
        for col, key in enumerate(("reward", "generalized_seconds_per_passenger")):
            delta = [raw["bapr", scene, seed][key]-raw["resac", scene, seed][key] for seed in (15001, 15002, 15003)]
            if col == 0:
                delta = np.array(delta)/1000
            mean = np.mean(delta)
            favorable = mean > 0 if col == 0 else mean < 0
            axes[col].barh(i, mean, height=.55, color=b.BLUE if favorable else b.RED, alpha=.85)
            axes[col].scatter(delta, i+np.array([-.15, 0, .15]), s=15,
                              facecolor="white", edgecolor="#30373D", lw=.65, zorder=4)
        for method in ("resac", "bapr", "no_hold", "headway"):
            group = [r for r in rows if r["method"] == method and r["scene"] == scene]
            means.append(dict(method=method, scene=scene, cost=np.mean([r["generalized_seconds_per_passenger"] for r in group]),
                              reward=np.mean([r["reward"] for r in group])))
    for col, ax in enumerate(axes):
        ax.axvline(0, color=b.GRAY, lw=.8, ls="--")
        ax.set(yticks=range(5), yticklabels=list(SCENES.values()) if col == 0 else [], ylim=(4.5, -.5),
               title=["(a) 奖励差", "(b) 乘客时间差"][col],
               xlabel=["BAPR减RE-SAC / 千", "BAPR减RE-SAC / 秒"][col])
        ax.grid(axis="x", alpha=.16)
    finish(fig, "fig4_legacy_paired", "早期BAPR与RE-SAC的相对收益随运行条件和评价目标变化",
           [str(root.relative_to(b.ROOT))], "固定一个训练种子；条形为3个共同评价种子的平均配对差，空心点为逐种子差；蓝色有利于BAPR，红色有利于RE-SAC")
    return dict(means=means, evaluations=len(rows))


def recovery():
    root = EXP / "csbapr_bus/runs/recovery_v1"
    p = read(root / "protocol.json")
    labels = {"bapr-pristine": "前序BAPR", "csbapr-no-nau": "稳定化ReLU", "csbapr": "CS-BAPR算术输出"}
    fig, axes = plt.subplots(2, 1, figsize=(5.75, 4.9), layout="constrained")
    rows, curves = [], []
    for index, (method, color) in enumerate(zip(p["methods"], [b.GRAY, b.BLUE, b.GREEN])):
        waiting, imbalance = [], []
        normal_returns, pulse_returns = [], []
        for seed in p["training_seeds"]:
            base = read(root / f"{method}_seed{seed}_normal.json")
            pulse = read(root / f"{method}_seed{seed}_pulse.json")
            a, c = base["windows"], pulse["windows"]
            assert base["horizon_seconds"] == pulse["horizon_seconds"] == p["horizon_seconds"]
            times = np.array([w["time_end"] for w in c])/60
            delta = np.array([w["mean_waiting"]-v["mean_waiting"] for w, v in zip(c, a)])
            tolerance = np.maximum(p["recovery_relative_tolerance"]*np.array([w["mean_waiting"] for w in a]),
                                   p["recovery_absolute_tolerance"])
            ok = delta <= tolerance
            n = p["recovery_consecutive_windows"]
            starts = [i for i, w in enumerate(c) if w["time_start"] >= p["pulse_end"]
                      and i+n <= len(c) and all(ok[i:i+n])]
            recovery_minutes = (c[starts[0]]["time_start"]-p["pulse_end"])/60 if starts else None
            row = dict(method=method, training_seed=seed, evaluation_seed=pulse["evaluation_seed"],
                       normal_return_K=base["reward"]/1000, pulse_return_K=pulse["reward"]/1000,
                       excess_wait_passenger_hours=(pulse["wait_passenger_seconds"]-base["wait_passenger_seconds"])/3600,
                       final_excess_waiting=delta[-1], recovery_minutes=recovery_minutes,
                       observation_after_pulse_minutes=(p["horizon_seconds"]-p["pulse_end"])/60)
            rows.append(row)
            normal_returns.append(base["reward"]/1000)
            pulse_returns.append(pulse["reward"]/1000)
            waiting.append(delta/10000)
            imbalance.append([w["mean_headway_imbalance"] if w["decisions"] else None for w in c])
            curves.append(dict(method=method, seed=seed, minutes=times.tolist(), excess_waiting=delta.tolist(),
                               pulse_headway_imbalance=imbalance[-1]))
        for j, (values, marker, label) in enumerate([(normal_returns, 'o', '无冲击'), (pulse_returns, '^', '五倍需求脉冲')]):
            x=index+(j-.5)*.28
            axes[0].plot([x,x], [min(values),max(values)], color=color, lw=1)
            axes[0].scatter([x-.03,x+.03], values, s=18, marker=marker, facecolor='white', edgecolor=color, zorder=3)
            axes[0].scatter(x, np.mean(values), s=43, marker=marker, color=color, zorder=4,
                            label=label if index==0 else None)
        values=np.asarray(waiting)
        axes[1].plot(times, np.mean(values,axis=0), lw=1.3, color=color, label=labels[method])
        axes[1].fill_between(times,np.min(values,axis=0),np.max(values,axis=0),color=color,alpha=.12)
    axes[0].set(ylabel="控制回报 / 千", title="(a) 正常与脉冲条件下的控制回报", xticks=range(3),
                xticklabels=list(labels.values()), xlim=(-.5,2.5))
    axes[0].margins(y=.3)
    axes[0].legend(fontsize=8,loc='lower left',ncol=2)
    axes[0].grid(axis='y')
    axes[1].axvspan(p["pulse_start"]/60,p["pulse_end"]/60,color=b.GOLD,alpha=.15)
    axes[1].set(xlabel="运行时间 / 分钟", ylabel="额外候车人数 / 万人", title="(b) 同期队列响应")
    axes[1].legend(fontsize=8,loc='lower right');axes[1].grid()
    finish(fig, "fig4_pulse_recovery", "五倍需求脉冲下CS-BAPR控制回报高于前序BAPR，同期服务队列反映供给压力",
           [str(root.relative_to(b.ROOT))], "每方法2组训练与评价种子配对；上图空心点为训练重复，实心点为均值；下图线为均值，阴影为两次运行范围，浅色带为5倍需求持续期；未恢复按右删失保留")
    write_csv("csbapr_recovery_results", rows)
    return dict(rows=rows, curves=curves, protocol=p, evaluations=12)


def campaign():
    p = read(EXP / "mcwm_bus/budget_protocol_v2.json")
    jobs = read(EXP / "mcwm_bus/campaign_v3_jobs.json")
    rows, snapshots = [], []
    for job in jobs:
        cell = job["signature"].rsplit("/", 1)[-1]
        count, scene, method = cell.split("_")
        count = int(count.removeprefix("blocks"))
        root = EXP / "sumo_eval/runs/sumo125_v3" / cell / "results"
        c = read(root / "completed.json")
        assert c["evaluation_complete"]
        training = read(EXP / "mcwm_bus/runs/campaign_v3" / cell / "results/training_summary.json")
        snapshots.extend(dict(cell=cell, **s) for s in training["snapshots"])
        for tag, demand in product(c["methods"], p["evaluation_demand_scales"]):
            d = read(root / "evaluation" / f"{tag}_demand{demand:g}_seed13301.json")
            assert d["horizon_seconds"] == 24000 and d["sumo_version"][1] == "SUMO 1.25.0"
            event_budget = int(tag.rsplit("events", 1)[1]) if "events" in tag else 0
            rows.append(dict(cell=cell, method=method, blocks=count, scene=scene, demand=demand,
                             matched_budget=event_budget, cost=d["generalized_seconds_per_departed_passenger"],
                             completion_percent=100*d["passengers"]["completion_rate"],
                             wait_seconds=d["passengers"]["accrued_wait_seconds_per_departed_passenger"],
                             ride_seconds=d["passengers"]["accrued_in_vehicle_seconds_per_departed_passenger"]))
    assert len(rows) == 60
    write_csv("bus_budget_and_mismatch", rows)
    lookup = {(r["blocks"], r["scene"], r["method"], r["matched_budget"], r["demand"]): r for r in rows}
    fig, axes = plt.subplots(2, 1, figsize=(5.75, 4.4), layout="constrained")
    methods = ["offline", "online", "h2o", "mcwm", "prefix"]
    columns = list(product(p["offline_block_counts"], p["source_event_budgets"]))
    for i, (ax, demand) in enumerate(zip(axes, p["evaluation_demand_scales"])):
        matrix = np.array([[lookup[0 if method == "online" else blocks, "calibrated",
                                  "offline" if method == "prefix" else method,
                                  0 if method == "prefix" else budget, demand]["cost"]
                            for blocks,budget in columns] for method in methods])
        im=ax.imshow(matrix,cmap="Blues",aspect="auto",interpolation="nearest")
        for (r,c),value in np.ndenumerate(matrix):
            ax.text(c,r,f"{value:.1f}",ha="center",va="center",fontsize=8.5,
                    color="white" if im.norm(value)>.62 else "#222222")
        ax.axvline(1.5,color="white",lw=3)
        ax.set(xticks=range(4),xticklabels=["2块 / 512","2块 / 2048","6块 / 512","6块 / 2048"],
               yticks=range(5),yticklabels=[NAMES.get(m,"冻结离线初始化") for m in methods])
        ax.set_title(f"({chr(97+i)}) 需求{demand:g}倍",loc="left")
        ax.tick_params(length=0)
        fig.colorbar(im,ax=ax,fraction=.034,pad=.025,label="人均广义时间 / 秒")
    axes[-1].set_xlabel("离线日志块数 / 后续策略更新次数")
    finish(fig, "fig6_budget_comparison", "数据量与交互量的增加未带来单调收益；混合续训优于等更新量离线续训",
           ["experiments/sumo_eval/runs/sumo125_v3", "experiments/mcwm_bus/budget_protocol_v2.json"],
           "单训练种子；每格为1个共同环境块；列为日志量与更新预算；纯源及冻结初始化重复作为参照；混合及纯源分别对应4096与16384次名义源交互，离线源交互为0")
    contrasts = []
    fig, axes = plt.subplots(1, 2, figsize=(5.75, 4.1), layout="constrained")
    values_by_demand = []
    for j, demand in enumerate(p["evaluation_demand_scales"]):
        gains, completions = [], []
        for scene in MISMATCH:
            h = lookup[6, scene, "h2o", 16384, demand]
            m = lookup[6, scene, "mcwm", 16384, demand]
            gain = 100*(h["cost"]-m["cost"])/h["cost"]
            completion = m["completion_percent"]-h["completion_percent"]
            gains.append(gain); completions.append(completion)
            contrasts.append(dict(scene=scene, demand=demand, h2o_cost=h["cost"], mcwm_cost=m["cost"],
                                  cost_reduction_percent=gain, completion_change_pp=completion))
        values_by_demand.append((gains, completions))
    cmap = LinearSegmentedColormap.from_list("effect", [b.RED,"#FAFAFA",b.BLUE])
    for col, ax in enumerate(axes):
        matrix=np.array([v[col] for v in values_by_demand]).T
        bound=np.max(np.abs(matrix))
        im=ax.imshow(matrix,cmap=cmap,norm=TwoSlopeNorm(0,-bound,bound),aspect="auto")
        for (r,c),v in np.ndenumerate(matrix):
            ax.text(c,r,f"{v:+.3f}",ha="center",va="center",fontsize=8,
                    color="white" if abs(v)/bound>.65 else "#222222")
        labels=["已标定参照","速度 −25%","OD强度 +25%","路口密度 +25%","停站时间 +25%","联合偏差25%","联合偏差50%","无经验时间标定"]
        ax.set(yticks=range(8),yticklabels=labels if col==0 else [],xticks=[0,1],xticklabels=["正常需求","高需求"],
               title=["(a) 广义时间下降","(b) 完成率变化"][col])
        ax.set_xticks([.5],minor=True);ax.set_yticks(np.arange(.5,8),minor=True)
        ax.grid(which="minor",color="white",lw=1.4);ax.tick_params(which="both",length=0)
        fig.colorbar(im,ax=ax,orientation="horizontal",fraction=.04,pad=.07,
                     label=["相对H2O+ / %","世界模型减H2O+ / 百分点"][col])
    finish(fig, "fig6_transition_mismatch", "世界模型收益依赖失配类型，时间与完成率应共同评价",
           ["experiments/sumo_eval/runs/sumo125_v3"], "6块日志、16384次名义源交互；每格为1个共同评价环境块；蓝色表示时间减少或完成率增加，红色表示退化；两面板色标单位不同")
    write_csv("bus_mismatch_contrasts", contrasts)
    return dict(rows=rows, snapshots=snapshots, contrasts=contrasts, evaluations=len(rows))


def pilot():
    root = EXP / "mcwm_bus/runs/control_v1"
    s = read(root / "service_summary.json")
    for row in s["rows"]:
        path = root / "evaluation" / f"{row['method']}_demand{row['demand_scale']:g}_seed{row['sumo_seed']}.json"
        d = read(path)
        assert d["sumo_version"][1] == "SUMO 1.25.0"
        assert np.isclose(row["generalized_seconds_per_passenger"], d["generalized_seconds_per_departed_passenger"], rtol=0, atol=1e-8)
    methods = ["offline_prefix", "offline_continue", "h2o_mixed", "mcwm_assisted"]
    labels = ["冻结离线初始化", "纯离线续训", "H2O+混合续训", "世界模型辅助"]
    fig, axes = plt.subplots(2, 2, figsize=(5.75, 4.1), layout="constrained")
    for row, demand in enumerate((1., 1.25)):
        for col, (key, title) in enumerate((("generalized_seconds_per_passenger", "人均广义时间 / 秒"),
                                            ("completion_percent", "行程完成率 / %"))):
            ax = axes[row, col]
            for block, color, marker in zip(sorted({r['sumo_seed'] for r in s['rows']}),[b.BLUE,b.GOLD],["o","s"]):
                vals=[next(r[key] for r in s["rows"] if r["method"]==m and r["demand_scale"]==demand and r["sumo_seed"]==block) for m in methods]
                ax.plot(range(4),vals,marker=marker,color=color,lw=1.2,ms=3.5,label=f"环境块{1 if marker=='o' else 2}")
            ax.set(xticks=range(4),xticklabels=["冻结\n初始化","离线\n续训","H2O+","世界\n模型"],
                   ylabel=title, title=f"({chr(97+2*row+col)}) 需求{demand:g}倍")
            ax.ticklabel_format(axis="y", useOffset=False)
            ax.grid(axis="y", alpha=.16)
    axes[0,0].legend(frameon=False,fontsize=7.5)
    finish(fig, "fig6_world_model_pilot", "共同离线初始化下，世界模型额外收益较小且并非逐场景一致",
           [str(root.relative_to(b.ROOT))], "1个训练种子；每档需求2个共同环境块；同色点线连接同一环境块的4种控制器，仅表示配对关系；SUMO 1.25.0；不同初始化与源采样的预算试验分组分析")
    return {k: v for k, v in s.items() if k != "rows"}


def main():
    plt.rcParams.update({"font.size": 8.5, "axes.titlesize": 9, "axes.labelsize": 8.5,
                         "xtick.labelsize": 8, "ytick.labelsize": 8.5})
    summary = dict(representation=representation(), legacy=legacy(), recovery=recovery(),
                   campaign=campaign(), pilot=pilot())
    (b.DATA / "bus_supplement_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2)+"\n")
    manifest = b.OUT / "figure_sources.json"
    previous = read(manifest)
    names = {r["figure"] for r in b.MANIFEST}
    previous = [r for r in previous if r["figure"] not in names]
    manifest.write_text(json.dumps(previous+b.MANIFEST, ensure_ascii=False, indent=2)+"\n")
    from repeated_shock_figures import render as repeated_shocks
    repeated_shocks()
    print(json.dumps({key: value.get("evaluations", value.get("completed_evaluations"))
                      for key, value in summary.items()}, ensure_ascii=False))


if __name__ == "__main__":
    main()
