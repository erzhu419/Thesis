"""Expand dissertation analyses using existing summaries and small service traces."""

import json
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

import build_figures as b


H2O = b.PROJECTS / "sumo-rl/H2Oplus"
CALIBRATION = H2O / "bus_h2o/calibrated_env"
FORMAL = H2O / "reproducibility/h2oplus_v20_formal_confirmation_results/evaluation_v4"
SCALES = [0.75, 1.0, 1.25]
POLICIES = [
    ("headway-gain-0.05", "时距增益0.05"),
    ("headway-gain-0.1", "时距增益0.10"),
    ("headway-gain-0.25", "时距增益0.25"),
    ("headway-gain-0.5", "时距增益0.50"),
    ("random-hurdle-p0.5-beta1-3", "偏短随机停站"),
    ("random-hurdle-p0.5-uniform", "均匀随机停站"),
    ("myopic-welfare-v2", "短视乘客福利"),
]


def source(path):
    return str(path.relative_to(b.PROJECTS))


def snapshot(name, data):
    (b.DATA / f"{name}.json").write_text(
        json.dumps(data, ensure_ascii=False, indent=2) + "\n"
    )


def fidelity():
    path = CALIBRATION / "fidelity_audit_v7_formal_v4.json"
    report = json.loads(path.read_text())
    metrics = [
        ("generalized_welfare_relative_gap", "广义乘客时间"),
        ("wait_time_relative_gap", "候车时间"),
        ("in_vehicle_time_relative_gap", "车内时间"),
        ("trip_duration_relative_gap", "车辆行程时长"),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(5.75, 4.7))
    fig.subplots_adjust(left=.16, right=.98, top=.92, bottom=.13, wspace=.42, hspace=.62)
    for ax, (key, label) in zip(axes.flat, metrics):
        points = [report["strata"][f"{s:g}"]["comparison"] for s in SCALES]
        mean = np.array([p["gaps"][key] for p in points]) * 100
        ci = np.array([p["gap_bootstrap_95_ci"][key] for p in points]) * 100
        ax.errorbar(SCALES, mean, yerr=[mean-ci[:, 0], ci[:, 1]-mean],
                    fmt="o-", color=b.BLUE, lw=1, capsize=3)
        ax.axhline(0, color=b.GRAY, ls=":", lw=.8)
        ax.set(title=label, xlabel="需求倍率", ylabel="源相对目标差 / %", xticks=SCALES)
        ax.grid(axis="y", color="#EEEEEE")
    b.save(fig, "fig2_fidelity_gaps", "自然运行中不同乘客成本分量的偏差方向不同",
           [source(path)], "每需求10次源仿真与5个未参与标定的SUMO环境；沿用原报告95%自助法区间")
    snapshot("fidelity_heldout", {"design": report["design"], "strata": {
        k: {"gaps": v["comparison"]["gaps"],
            "ci": v["comparison"]["gap_bootstrap_95_ci"]}
        for k, v in report["strata"].items()}})

    path = CALIBRATION / "intervention_fidelity_audit_v7_formal_v3.json"
    report = json.loads(path.read_text())
    comparison = report["comparison"]["comparisons"]
    fig, axes = plt.subplots(1, 2, figsize=(5.75, 4.7), sharey=True)
    fig.subplots_adjust(left=.27, right=.98, bottom=.16, top=.84, wspace=.24)
    for ax, key, title, unit in [
        (axes[0], "generalized_seconds_per_person", "广义时间响应差", "%"),
        (axes[1], "completion_rate", "完成率响应差", "百分点"),
    ]:
        for j, (scale, color) in enumerate(zip(SCALES, [b.BLUE, b.GREEN, b.RED])):
            for i, (policy, _) in enumerate(POLICIES):
                r = comparison[f"{scale:g}"][policy]["metrics"][key]
                mean = 100*r["normalized_response_gap"]
                lo, hi = np.array(r["gap_bootstrap_95_ci"])*100
                ax.errorbar(mean, i+(j-1)*.20, xerr=[[mean-lo], [hi-mean]],
                            fmt="o", color=color, ms=3, capsize=2,
                            label=f"需求{scale:g}倍" if i == 0 else None)
        ax.axvline(0, color=b.GRAY, lw=.8, ls=":")
        ax.set(title=title, xlabel=f"源响应 − 目标响应 / {unit}",
               yticks=range(7), yticklabels=[label for _, label in POLICIES], ylim=(6.6, -.6))
        ax.tick_params(labelsize=8)
        ax.grid(axis="x", color="#EEEEEE")
    axes[0].set_title("(a) 广义时间响应差", fontsize=9)
    axes[1].set_title("(b) 完成率响应差", fontsize=9)
    fig.legend(*axes[0].get_legend_handles_labels(), loc="upper center", ncol=3, frameon=False, fontsize=8)
    b.save(fig, "fig2_intervention_response", "停站响应比较检验控制效果而不只是无控制基准",
           [source(path)], "每策略每需求10次源与10次SUMO；先减各环境零停站结果；沿用原95%区间；时间按SUMO零停站均值归一化")
    snapshot("intervention_response", {"design": report["design"], "comparisons": comparison})


def physical_process():
    directory = H2O / "bus_h2o/datasets_v7/formal_v2_zero/zero"
    paths = sorted(directory.glob("*/*.full_service.npz"))
    assert len(paths) == 30
    runtimes = defaultdict(list)
    for path in paths:
        with np.load(path) as data:
            for line in np.unique(data["segment_line_index"]):
                runtimes[int(line)].extend(data["segment_road_runtime_seconds"][data["segment_line_index"] == line])
    labels = ["102上", "102下", "122上", "122下", "311上", "311下",
              "406上", "406下", "705上", "705下", "7上", "7下"]
    fig, ax = plt.subplots(figsize=(5.75, 3.8))
    fig.subplots_adjust(left=.14, right=.98, top=.93, bottom=.22)
    stats = []
    for line in range(12):
        q = np.quantile(runtimes[line], [.05, .25, .5, .75, .95])
        stats.append(dict(label=labels[line], whislo=q[0], q1=q[1], med=q[2], q3=q[3], whishi=q[4], fliers=[]))
    ax.bxp(stats, showfliers=False, patch_artist=True,
           boxprops=dict(facecolor="#CEE5ED", linewidth=.7), medianprops=dict(color=b.BLUE), widths=.6)
    ax.set(ylabel="站间道路运行时间 / 秒", xlabel="方向性服务")
    ax.tick_params(axis="x", labelrotation=50, labelsize=8)
    ax.grid(axis="y", color="#EEEEEE")
    b.save(fig, "fig2_segment_runtime", "站间运行时间在方向性服务之间及同一服务内部均存在差异",
           [source(p) for p in paths], "30个零停站SUMO回合中实际离站至下一站到达的时间；箱体25%-75%，中线50%，须5%-95%；描述性汇总")
    snapshot("segment_runtime_summary", {str(k): {"n": len(v), "q05_q25_q50_q75_q95": np.quantile(v, [.05,.25,.5,.75,.95]).tolist()} for k,v in runtimes.items()})

    path = directory / "demand_1_jitter120s_seed1201/sumo_seed1101.full_service.npz"
    with np.load(path) as data:
        fig, axes = plt.subplots(2, 1, figsize=(5.75, 4.65), sharex=True, sharey=True)
        fig.subplots_adjust(left=.12, right=.98, bottom=.13, top=.94, hspace=.35)
        for ax, line, title in zip(axes, [4, 5], ["(a) 311路上行", "(b) 311路下行"]):
            mask = data["stop_line_index"] == line
            for index, bus in enumerate(np.unique(data["stop_bus_code"][mask])):
                m = mask & (data["stop_bus_code"] == bus)
                order = np.argsort(data["stop_station_index"][m])
                x = data["stop_arrival_time_seconds"][m][order]/3600
                y = data["stop_station_index"][m][order]+1
                ax.plot(x, y, color=b.BLUE if index % 5 == 0 else "#A1AFBC", alpha=.95, lw=1 if index % 5 == 0 else .6)
            ax.set(ylabel="站点序号")
            ax.set_title(title, loc="left")
            ax.grid(color="#EEEEEE", linewidth=.5)
        axes[-1].set_xlabel("仿真时间 / 小时")
        b.save(fig, "fig2_baseline_trajectories", "车辆到站轨迹呈现异步运行和沿程间隔变化",
               [source(path)], "固定选择标准需求、首个环境块的零停站回合；线段连接实际到站事件，不表示站间瞬时轨迹，不用于估计控制收益")


def formal_service():
    methods = {"parent": "h2oplus-v19-completion-safety-baseline",
               "candidate": "h2oplus-v19-completion-safety-quarter-load-sixty"}
    observations = {}
    paths = []
    for arm, method in methods.items():
        for path in sorted(FORMAL.glob(f"*/*/{method}-seed*-events000025000/*/sumo_seed*.json")):
            d = json.loads(path.read_text())
            seed = int(d["method"].split("-seed")[1].split("-")[0])
            key = (int(d["sumo_seed"]), float(d["scenario"]["demand_scale"]), seed, arm)
            assert key not in observations
            assert d["episode"]["observed_end_time_seconds"] == 24000
            p, o = d["passengers"], d["operations"]
            observations[key] = {
                "wait": p["accrued_wait_seconds_per_departed_passenger"],
                "in_vehicle": p["accrued_in_vehicle_seconds_per_departed_passenger"],
                "generalized": d["passenger_welfare"]["accrued_generalized_seconds_per_departed_passenger"],
                "completion": p["completion_rate"], "unfinished": p["unfinished_passengers"],
                "hold": o["hold_seconds"]["mean"], "bunching": o["headway"]["bunching_rate"],
                "large_gap": o["headway"]["large_gap_rate"],
                "lines": {line: {"hold": v["hold_seconds"]["mean"],
                                  "bunching": v["headway"]["bunching_rate"],
                                  "large_gap": v["headway"]["large_gap_rate"]}
                          for line, v in o["per_line"].items()},
            }
            assert np.isclose(2*observations[key]["wait"] + observations[key]["in_vehicle"], observations[key]["generalized"])
            paths.append(source(path))
    expected = {(block, scale, seed, arm) for block in range(8101, 8111)
                for scale in SCALES for seed in range(6201, 6206) for arm in methods}
    assert set(observations) == expected, (len(observations), len(expected))
    metrics = ["wait", "in_vehicle", "generalized", "completion", "unfinished", "hold", "bunching", "large_gap"]
    summaries = {}
    for scale in [*SCALES, None]:
        group = [(k, v) for k,v in observations.items() if scale is None or k[1] == scale]
        means = {arm: {m: float(np.mean([v[m] for k,v in group if k[3] == arm])) for m in metrics} for arm in methods}
        diff = {m: means["candidate"][m]-means["parent"][m] for m in metrics}
        summaries["all" if scale is None else f"{scale:g}"] = {"means": means, "difference": diff}
    assert np.isclose(summaries["all"]["difference"]["generalized"], -92.12, atol=.015)
    line_differences = {}
    for line in sorted(next(iter(observations.values()))["lines"]):
        line_differences[line] = {m: [] for m in ["hold", "bunching", "large_gap"]}
        for block in range(8101, 8111):
            for m in line_differences[line]:
                deltas = [observations[block, scale, seed, "candidate"]["lines"][line][m]
                          - observations[block, scale, seed, "parent"]["lines"][line][m]
                          for scale in SCALES for seed in range(6201, 6206)]
                line_differences[line][m].append(float(np.mean(deltas)))
    snapshot("h2o_service_decomposition", {"summaries": summaries, "line_block_differences": line_differences,
             "statistics": "300 existing episodes; 150 matched pairs; independent unit is 10 environment blocks"})
    fig, axes = plt.subplots(1, 4, figsize=(5.75, 3.45), sharey=True)
    fig.subplots_adjust(left=.15,right=.99,bottom=.25,top=.88,wspace=.12)
    for ax, key, title in zip(axes,["0.75","1","1.25","all"],["0.75倍需求","1倍需求","1.25倍需求","总体"]):
        d=summaries[key]["difference"]
        vals=[2*d["wait"],d["in_vehicle"],d["generalized"]]
        ax.bar([0,1,2], vals, bottom=[0,vals[0],0], color=[b.BLUE,b.RED,b.GREEN], width=.65)
        ax.plot([.34,.66],[vals[0]]*2,color=b.GRAY,lw=.7)
        ax.plot([1.34,1.66],[vals[2]]*2,color=b.GRAY,lw=.7)
        for i,v in enumerate(vals):
            end=v if i!=1 else vals[2]
            ax.text(i,end+(-8 if i!=1 else 7),f"{v:+.1f}",ha="center",va="top" if i!=1 else "bottom",fontsize=7)
        ax.axhline(0,color=b.GRAY,lw=.7)
        ax.set(xticks=[0,1,2],xticklabels=["候车","车内","合计"],ylim=(-200,55))
        ax.tick_params(axis="x",labelrotation=45,labelsize=8)
        ax.set_title(title,fontsize=8.5)
        ax.grid(axis="y")
    axes[0].set_ylabel("广义时间变化 / 秒每人")
    b.save(fig,"fig6_passenger_decomposition","广义时间改善由候车节省与车内代价共同决定",paths,
           "同一300回合重新分解既有主指标，不是新实验；各需求50个配对、总体150个配对的均值；不增加显著性检验")
    labels = [line.replace("S","上").replace("X","下") for line in line_differences]
    fig, axes = b.pair_panels(labels, height=4.8, left=.16)
    for ax, metric, multiplier, title, unit in [
        (axes[0],"hold",1,"(a) 平均附加停站变化","秒/事件"),
        (axes[1],"bunching",100,"(b) 串车事件比例变化","百分点"),
    ]:
        for y, line in enumerate(line_differences):
            values = np.array(line_differences[line][metric])*multiplier
            ax.scatter(values, y+np.linspace(-.12,.12,10),color=b.GRAY,alpha=.45,s=10)
            ax.scatter(values.mean(),y,color=b.BLUE,marker="D",s=25,zorder=3)
        ax.axvline(0,color=b.GRAY,lw=.8,ls=":")
        ax.set_title(title,fontsize=9)
        ax.set_xlabel(f"完成保障 − 直接参照 / {unit}",fontsize=8)
    b.save(fig,"fig6_line_response","完成保障对不同方向性服务的停站与时距作用并不相同",paths,
           "灰点为10个独立环境块中的配对均值，蓝菱形为总体均值；串车沿用原评价器定义，分线路结果为事后描述")
    print(json.dumps(summaries,ensure_ascii=False,indent=2))


def main():
    fidelity()
    physical_process()
    formal_service()
    path = b.OUT / "figure_sources.json"
    old = json.loads(path.read_text())
    names = {item["figure"] for item in b.MANIFEST}
    path.write_text(json.dumps([item for item in old if item["figure"] not in names]+b.MANIFEST,
                               ensure_ascii=False,indent=2)+"\n")


if __name__ == "__main__":
    main()
