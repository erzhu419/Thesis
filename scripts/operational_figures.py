"""Explain corridor control and timetable execution using existing local material."""

import ast
import csv
import json
from collections import defaultdict

import numpy as np
import openpyxl
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

import build_figures as b


def snapshot(name, data):
    (b.DATA / f"{name}.json").write_text(
        json.dumps(data, ensure_ascii=False, indent=2) + "\n")


def holding_geometry():
    from mechanism_figures import reward_geometry
    reward_geometry(b)


def corridor_inputs():
    root = b.PROJECTS / "Catetorical-Feature-SAC-in-bus-simulation/env/data"
    workbook = openpyxl.load_workbook(root / "passenger_OD.xlsx", read_only=True, data_only=True)
    rates = defaultdict(float)
    od_rows = list(workbook.active.values)[1:]
    for row in od_rows:
        rates[row[0].hour] += sum(float(x or 0) for x in row[2:])
    od = np.array([[float(x or 0) for x in row[2:]] for row in od_rows]).reshape(14,20,20).sum(axis=0)
    workbook.close()
    workbook = openpyxl.load_workbook(root / "route_news.xlsx", read_only=True, data_only=True)
    rows = list(workbook.active.values)
    hours = [value.hour for value in rows[0][5:]]
    speeds = np.array([row[5:] for row in rows[1:]], dtype=float)
    workbook.close()
    assert set(hours) == set(rates) and speeds.shape == (42, 14)
    mean = speeds.mean(axis=0)
    q10, q90 = np.quantile(speeds, [.1, .9], axis=0)
    snapshot("corridor_hourly_inputs", {"hours": hours, "od_rate": [rates[h] for h in hours],
             "link_speed_mean": mean.tolist(), "link_speed_q10": q10.tolist(),
             "link_speed_q90": q90.tolist(), "n_directed_links": len(speeds)})
    fig = plt.figure(figsize=(5.75, 5.35), layout="constrained")
    gs = fig.add_gridspec(2, 2, height_ratios=[1.25,1], width_ratios=[1,1])
    ax = fig.add_subplot(gs[0,0])
    im = ax.imshow(od, origin="upper", cmap="Blues", interpolation="nearest")
    ax.set(title="(a) 日累计OD设定", xlabel="到达站点序号", ylabel="出发站点序号",
           xticks=[0,4,9,14,19], xticklabels=[1,5,10,15,20], yticks=[0,4,9,14,19], yticklabels=[1,5,10,15,20])
    fig.colorbar(im, ax=ax, orientation="horizontal", fraction=.065, pad=.10, label="设定人数")
    ax = fig.add_subplot(gs[0,1])
    ax.barh(hours, [rates[h] for h in hours], height=.70, color=b.BLUE)
    ax.set(title="(b) 小时需求", xlabel="设定强度 / 人每小时", ylabel="时段起点 / 时", yticks=hours)
    ax.grid(axis="x")
    ax = fig.add_subplot(gs[1,:])
    im = ax.imshow(speeds, aspect="auto", cmap="YlGnBu", interpolation="nearest",
                   extent=[hours[0]-.5,hours[-1]+.5,42.5,.5], vmin=0, vmax=15)
    ax.axhline(21.5,color="white",lw=1)
    ax.set(title="(c) 道路运行条件的时空异质性", xlabel="时段起点 / 时", ylabel="有向路段序号",
           xticks=hours, yticks=[1,10,21,32,42])
    fig.colorbar(im, ax=ax, fraction=.028, pad=.02, label="速度 / 米每秒")
    b.save(fig, "fig3_corridor_inputs", "小时需求与路段条件为类别共享策略提供不同运行上下文",
           [str(root / "passenger_OD.xlsx"), str(root / "route_news.xlsx")],
           "仿真输入而非真实客流观测；OD按14个小时累加，小时需求按OD求和；速度矩阵保留42条有向路段与14个小时原值")


def demand_schedules():
    path = b.PROJECTS / "CS-BAPR/scripts/test_multiline_convergence.py"
    tree = ast.parse(path.read_text())
    schedules = next(ast.literal_eval(node.value) for node in tree.body
                     if isinstance(node, ast.Assign)
                     and any(isinstance(t, ast.Name) and t.id == "OSCILLATING_SCHEDULES" for t in node.targets))
    labels = {"commuter_day": "通勤型", "square_wave_20x": "方波型", "escalating_10x": "递增型"}
    fig, axes = plt.subplots(3, 1, figsize=(5.75, 4.65), sharex=True, sharey=True)
    fig.subplots_adjust(left=.12, right=.97, top=.94, bottom=.12, hspace=.52)
    summary = {}
    for i, (key, label) in enumerate(labels.items()):
        seq = schedules[key]
        times = [v[0] for v in seq] + [18000]
        values = [v[1] for v in seq]
        durations = np.diff(times)
        mean = float(np.dot(durations, values) / 18000)
        summary[key] = {"schedule": seq, "time_mean_multiplier": mean,
                        "switches": len(seq)-1, "peak_duration_seconds": int(durations[np.array(values) == max(values)].sum())}
        ax = axes[i]
        ax.step(np.array(times)/3600, values + [values[-1]], where="post", color=b.COLORS[i], lw=1.6)
        ax.axhline(mean, color=b.GRAY, ls="--", lw=.8)
        ax.set(title=f"({chr(97+i)}) {label}：{len(seq)-1}次切换，时间平均倍率{mean:.1f}",
               ylabel="需求倍率", ylim=(-1, 54), yticks=[1, 20, 50])
        ax.grid(color="#EEEEEE", lw=.6)
    axes[-1].set(xlabel="回合运行时间 / 小时", xticks=np.arange(0, 5.1, .5))
    snapshot("csbapr_demand_schedules", summary)
    b.save(fig, "fig4_demand_schedules", "变化场景在峰值持续时间、累计倍率和切换频率上具有不同结构",
           [str(path)], "实际评价脚本中的三组阶梯函数；时间平均倍率由积分计算，并非实际到达人数或模型响应曲线")


def transit_execution():
    root = b.PROJECTS / "TransitDuet/transit_duet/results_remote/revision3_heldout"
    methods = [("H_timetable_v5_stable_300", "TransitDuet"), ("H_hiro_final", "仅目标传递"),
               ("H_haar_final", "强化学习代理"), ("H_fixed_timetable_300_final", "固定300秒"),
               ("H_fixed_timetable_360_final", "固定360秒"), ("baseline_rule_daganzo", "Daganzo型"),
               ("baseline_rule_xuan", "Xuan型")]
    summary = {r["method"]: r for r in csv.DictReader((root / "main_results_heldout.csv").open())}
    records = {method: {int(p.stem[4:]): json.loads(p.read_text()) for p in (root/method).glob("seed*.json")}
               for method, _ in methods}
    primary = methods[0][0]
    # Confirm common test conditions before interpreting seed differences.
    for method, group in records.items():
        assert len(group) == 10 and set(group) == set(records[primary])
        for seed, row in group.items():
            base = records[primary][seed]
            assert row["n_test_episodes"] == 50
            assert row["fleet_schedule"] == base["fleet_schedule"]
            assert row["test_episode_seeds"] == base["test_episode_seeds"]
    labels = [name for _, name in methods]
    fig, axes = plt.subplots(1, 2, figsize=(5.75, 4.1), sharey=True)
    fig.subplots_adjust(left=.245, right=.975, top=.84, bottom=.16, wspace=.25)
    y = np.arange(len(methods))
    for shift, key, label, color in [(-.13, "planned", "计划时刻间隔", b.BLUE),
                                     (.13, "actual", "实际发车间隔", b.GREEN)]:
        means = [float(summary[m][f"{key}_dispatch_headway_std_mean"]) for m, _ in methods]
        sd = [float(summary[m][f"{key}_dispatch_headway_std_seed_sd"]) for m, _ in methods]
        axes[0].errorbar(means, y+shift, xerr=sd, fmt="o", ms=3.5, color=color, capsize=2, label=label)
    for i, (method, _) in enumerate(methods):
        values = [r["metrics"]["dispatch_lateness_mean_mean"] for r in records[method].values()]
        axes[1].scatter(values, i+np.linspace(-.13,.13,10), color=b.GRAY, s=12, alpha=.5)
        axes[1].scatter(np.mean(values), i, color=b.RED, marker="D", s=24, zorder=3)
    axes[0].set(yticks=y, yticklabels=labels, title="(a) 回合内间隔标准差", xlabel="标准差 / 秒")
    axes[0].invert_yaxis()
    axes[1].set_xscale("log")
    axes[1].set(title="(b) 平均发车延误", xlabel="平均延误 / 秒（对数）", xlim=(1, 2000),
                xticks=[1, 10, 100, 1000], xticklabels=["1", "10", "100", "1000"])
    for ax in axes:
        ax.grid(axis="x", color="#EEEEEE", lw=.6)
    fig.legend(*axes[0].get_legend_handles_labels(), loc="upper center", ncol=2, frameon=False, fontsize=8.5)
    b.save(fig, "fig4_dispatch_execution", "计划时刻间隔、实际发车规律和执行延误是不同的运营指标",
           [str(root / "main_results_heldout.csv"), *[str(root / m / f"seed{s}.json") for m,_ in methods for s in sorted(records[m])]],
           "每方法10个种子、每种子50个既有测试回合；左图为种子均值的均值与标准差，右图灰点为种子均值、菱形为总体均值；非新实验")

    keys = ["wait_mean", "avg_onboard_time_min_mean", "avg_total_passenger_time_min_mean"]
    refs = [("H_fixed_timetable_300_final", "固定300秒"), ("H_hiro_final", "仅目标传递")]
    differences = {}
    fig, axes = plt.subplots(1, 2, figsize=(5.75, 3.2), sharey=True)
    fig.subplots_adjust(left=.17, right=.98, top=.77, bottom=.21, wspace=.25)
    for ax, (ref, label) in zip(axes, refs):
        differences[ref] = {}
        for i, key in enumerate(keys):
            values = np.array([records[primary][s]["metrics"][key]-records[ref][s]["metrics"][key]
                               for s in sorted(records[primary])])
            differences[ref][key] = {"seed_differences": values.tolist(), "mean": float(values.mean()),
                                     "negative_seeds": int((values<0).sum())}
            ax.scatter(values, i+np.linspace(-.14,.14,10), color=b.GRAY, alpha=.6, s=15)
            ax.scatter(values.mean(), i, color=b.BLUE, marker="D", s=32, zorder=3)
        ax.axvline(0, color=b.GRAY, lw=.8, ls=":")
        ax.set(title=f"相对{label}", xlabel="TransitDuet − 参照 / 分钟", xlim=(-2.6,.7), xticks=[-2,-1,0])
        ax.grid(axis="x",color="#EEEEEE",lw=.6)
    axes[0].set(yticks=[0,1,2], yticklabels=["候车时间", "车内时间", "总出行时间"], ylim=(2.4,-.4))
    fig.legend(handles=[Line2D([0],[0],color=b.GRAY,marker="o",lw=0,label="配对种子均值差"),
                        Line2D([0],[0],color=b.BLUE,marker="D",lw=0,label="10种子平均差")],
               loc="upper center",ncol=2,frameon=False,fontsize=8.5)
    snapshot("transit_paired_passenger_components", differences)
    b.save(fig, "fig4_paired_passenger_components", "服务收益的组成及跨种子一致性取决于所选参照",
           [str(root/m/f"seed{s}.json") for m in [primary,*[r[0] for r in refs]] for s in sorted(records[m])],
           "既有独立测试的事后配对分解；同种子共享50个测试条件；10个点为种子均值差，无新增显著性检验")


def main():
    holding_geometry()
    corridor_inputs()
    demand_schedules()
    transit_execution()
    path = b.OUT / "figure_sources.json"
    old = json.loads(path.read_text())
    names = {row["figure"] for row in b.MANIFEST}
    path.write_text(json.dumps([row for row in old if row["figure"] not in names]+b.MANIFEST,
                               ensure_ascii=False, indent=2)+"\n")


if __name__ == "__main__":
    main()
