"""Render thesis-only Chinese figures from existing local results and task files."""

from pathlib import Path
import csv
import json
import xml.etree.ElementTree as ET
import sys

import additional_figures
import thesis_schematics

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.collections import LineCollection
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle, FancyArrowPatch
import numpy as np
from PIL import Image, ImageOps, ImageDraw


ROOT = Path(__file__).resolve().parents[1]
PROJECTS = ROOT.parent
OUT = ROOT / "manuscript/figures"
DATA = OUT / "source_data"
PUBLISHED = json.loads((DATA / "published_values.json").read_text())
FONT = "/mnt/c/Windows/Fonts/msyh.ttc"
font_manager.fontManager.addfont(FONT)
plt.rcParams.update({
    "font.family": font_manager.FontProperties(fname=FONT).get_name(),
    "font.size": 9.5, "axes.titlesize": 10, "axes.labelsize": 9.5,
    "xtick.labelsize": 8.5, "ytick.labelsize": 9,
    "axes.unicode_minus": False, "axes.spines.top": False,
    "axes.spines.right": False, "axes.linewidth": .8,
    "axes.labelcolor": "#222222", "text.color": "#222222",
    "xtick.color": "#333333", "ytick.color": "#333333",
    "xtick.major.size": 3, "ytick.major.size": 3,
    "axes.axisbelow": True, "legend.frameon": False,
    "mathtext.fontset": "stix", "lines.linewidth": 1.4,
    "svg.fonttype": "none", "pdf.fonttype": 42,
    "savefig.facecolor": "white", "figure.facecolor": "white",
})
BLUE, GREEN, RED, PURPLE, GRAY, GOLD = (
    "#286A98", "#27877C", "#C65A46", "#80638E", "#787F87", "#B68A37"
)
COLORS = [BLUE, GREEN, RED, PURPLE, GRAY, GOLD]
MANIFEST = []
TEXT_BOXES = []


def save(fig, name, conclusion, sources, statistics="概念示意，无实验统计量"):
    for ax in fig.axes:
        ax.set_axisbelow(True)
        for line in ax.get_xgridlines() + ax.get_ygridlines():
            line.set_color("#DCE1E5")
            line.set_linewidth(.45)
            line.set_alpha(.75)
    # Check text against both the page and its diagram node before export.
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    for artist, bounds in TEXT_BOXES:
        if artist.figure is not fig:
            continue
        box = artist.get_window_extent(renderer)
        x, y, w, h = bounds
        p0, p1 = artist.axes.transData.transform([(x, y), (x + w, y + h)])
        assert box.x0 >= p0[0] and box.x1 <= p1[0], (name, artist.get_text())
        assert box.y0 >= p0[1] and box.y1 <= p1[1], (name, artist.get_text())
    for artist in fig.findobj(matplotlib.text.Text):
        if not artist.get_visible() or not artist.get_text():
            continue
        box = artist.get_window_extent(renderer)
        # Matplotlib keeps invisible out-of-range tick labels in its artist tree.
        if artist in [a.xaxis.get_offset_text() for a in fig.axes] and not artist.get_text():
            continue
        if box.width and box.height and artist.get_clip_on() is False:
            assert box.x0 >= -1 and box.y0 >= -1, (name, artist.get_text(), box)
            assert box.x1 <= fig.bbox.width + 1 and box.y1 <= fig.bbox.height + 1, (name, artist.get_text(), box)
    for ext in ("png", "pdf", "svg"):
        fig.savefig(OUT / f"{name}.{ext}", dpi=360)
    MANIFEST.append({
        "figure": name, "conclusion": conclusion, "sources": sources,
        "statistics": statistics, "width_mm": 146.05,
        "height_mm": round(fig.get_figheight() * 25.4, 2),
        "formats": ["PNG 360 dpi", "PDF vector", "SVG editable text"],
    })
    plt.close(fig)
    print(name, flush=True)


def rows(relative, copy_name):
    with (PROJECTS / relative).open(newline="") as file:
        data = list(csv.DictReader(file))
    with (DATA / copy_name).open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(data[0]))
        writer.writeheader()
        writer.writerows(data)
    return data


def diagram(height=4.7):
    fig, ax = plt.subplots(figsize=(5.75, height))
    fig.subplots_adjust(left=.025, right=.975, bottom=.025, top=.975)
    ax.set(xlim=(0, 1), ylim=(0, 1))
    ax.axis("off")
    return fig, ax


def node(ax, x, y, w, h, text, color=BLUE, dashed=False, size=9.5):
    ax.add_patch(Rectangle((x, y), w, h, facecolor="white", edgecolor=color,
                           lw=1.1, linestyle="--" if dashed else "-", zorder=3))
    artist = ax.text(x + w / 2, y + h / 2, text, ha="center", va="center",
                     fontsize=size, linespacing=1.5, zorder=4)
    TEXT_BOXES.append((artist, (x, y, w, h)))


def arrow(ax, start, end, color=GRAY, dashed=False, rad=0):
    ax.add_patch(FancyArrowPatch(start, end, arrowstyle="-|>", mutation_scale=10,
                                lw=1, color=color, linestyle="--" if dashed else "-",
                                connectionstyle=f"arc3,rad={rad}", zorder=2))


def heading(ax, x, y, label, color=BLUE):
    ax.text(x, y, label, va="center", ha="left", weight="bold", color=color, fontsize=10)




def network():
    base = PROJECTS / "sumo-rl/SUMO_ruiguang"
    netpath = base / "b_network/5g_changsha_bus_network_with_signal_d.net.xml"
    routepath = base / "d_bus_rou/2_bus_timetable.rou.xml"
    edges = {}
    for edge in ET.parse(netpath).getroot().findall("edge"):
        if edge.get("function") == "internal":
            continue
        lane = edge.find("lane")
        if lane is not None and lane.get("shape"):
            edges[edge.get("id")] = np.array([[float(v) for v in p.split(",")[:2]] for p in lane.get("shape").split()]) / 1000
    routes = {}
    for vehicle in ET.parse(routepath).getroot().findall("vehicle"):
        service = vehicle.get("id").rsplit("_", 1)[0]
        if service not in routes:
            routes[service] = vehicle.find("route").get("edges").split()
    assert len(routes) == 12
    fig, ax = plt.subplots(figsize=(5.75, 5.7))
    fig.subplots_adjust(left=.13, right=.98, bottom=.22, top=.98)
    ax.add_collection(LineCollection(list(edges.values()), colors="#D2D2D2", linewidths=.45))
    families = ["7", "102", "122", "311", "406", "705"]
    for service, edge_ids in routes.items():
        missing = set(edge_ids) - edges.keys()
        assert not missing, (service, missing)
        color = COLORS[families.index(service[:-1])]
        ax.add_collection(LineCollection([edges[key] for key in edge_ids], colors=color,
                                         linewidths=1.15, linestyles="-" if service.endswith("X") else "--", alpha=.85))
    ax.autoscale()
    ax.set_aspect("equal")
    ax.set_xlabel("仿真平面坐标 x / 千米")
    ax.set_ylabel("仿真平面坐标 y / 千米")
    legend = [Line2D([0], [0], color=c, lw=1.5, label=f"{n}路") for n, c in zip(families, COLORS)]
    legend += [Line2D([0], [0], color="#222222", lw=1.2, label="方向一"),
               Line2D([0], [0], color="#222222", lw=1.2, ls="--", label="方向二")]
    fig.legend(handles=legend, loc="lower center", bbox_to_anchor=(.5, .01), ncol=4, frameon=False, fontsize=9)
    (DATA / "sumo_routes.json").write_text(json.dumps(routes, ensure_ascii=False, indent=2) + "\n")
    save(fig, "fig2_2_sumo_network", "实际SUMO路网中的六条线路与十二条方向性服务", [str(netpath), str(routepath)], "原始路段几何；每条方向性服务采用首班路线；坐标由米换算为千米")




def pair_panels(labels, height=4.3, left=.265):
    fig, axes = plt.subplots(1, 2, figsize=(5.75, height), sharey=True)
    fig.subplots_adjust(left=left, right=.975, bottom=.17, top=.90, wspace=.27)
    for ax in axes:
        ax.set_yticks(range(len(labels)), labels)
        ax.set_ylim(len(labels)-.5, -.5)
        ax.grid(axis="x", color="#E4E4E4", lw=.6)
        ax.set_axisbelow(True)
    axes[1].tick_params(axis="y", left=False)
    return fig, axes


def resac_results():
    source = DATA / "resac_archive_evaluation.json"
    data = json.loads(source.read_text())["rows"]
    labels = ["SAC", "K=2", "K=5", "K=10", "K=20", "K=40", "仅权重\n范数", "仅集成\n分歧"]
    fig, axes = plt.subplots(2, 1, figsize=(5.75, 4.7), sharex=True, sharey=True)
    fig.subplots_adjust(left=.16, right=.97, bottom=.13, top=.86, hspace=.35)
    for ax, kind, title in zip(axes, ("train", "operational"),
                               ("(a) 最后100回合训练", "(b) 确定性运行评价")):
        for i, row in enumerate(data):
            extension = row["extension_train"] if kind == "train" else row["extension_operational"]["reward"]
            archive = row["archive_train"] if kind == "train" else row["archive_operational"]["reward"]
            ax.plot([i - .15, i - .15], np.array(extension["values"]) / 1000,
                    color=BLUE, lw=.9, marker="o", ms=3)
            ax.plot(i - .15, extension["mean"] / 1000, color=BLUE, marker="_", ms=10)
            ax.plot(i + .15, archive / 1000, marker="D", mfc="white", mec=RED, ms=4, linestyle="none")
        ax.set(ylabel="回报 / 千", xticks=range(len(data)), xticklabels=labels)
        ax.set_title(title, loc="left", fontsize=9)
        ax.grid(axis="y")
    fig.legend(handles=[Line2D([], [], color=BLUE, marker="o", ms=3, lw=.9,
                              label="扩展种子32、40"),
                        Line2D([], [], color=RED, marker="D", mfc="white", ms=4,
                              lw=0, label="历史归档8")],
               loc="upper center", ncol=2, bbox_to_anchor=(.57, .99), fontsize=8.5)
    save(fig, "fig3_2_resac_ensemble", "历史策略与扩展训练分列，集成规模收益并非单调", [str(source.relative_to(ROOT))],
         "蓝点为两个训练重复，短横为均值，连接线为两点范围而非置信区间；空心菱形为归档策略；运行值先平均10回合；历史SAC宽32，扩展宽64")


def temperature():
    data = PUBLISHED["temperature"]
    fig, ax = plt.subplots(figsize=(5.75, 3.8))
    fig.subplots_adjust(left=.23, right=.95, bottom=.20, top=.95)
    for y, gain in enumerate(data["gains"]):
        ax.barh(y, gain / 1000, height=.52, color=BLUE if gain > 0 else RED, zorder=3)
    mean, (low, high) = data["mean"], data["ci95_t"]
    ax.errorbar(mean/1000, 5.5, xerr=[[(mean-low)/1000], [(high-mean)/1000]], fmt="D", color="#222222", capsize=4, ms=5)
    ax.axhline(4.8, color="#DDDDDD", lw=.8)
    ax.axvline(0, color=GRAY, ls="--", lw=.8)
    ax.set_yticks([0, 1, 2, 3, 4, 5.5], [f"种子 {s}" for s in data["seeds"]] + ["均值及95%区间"])
    ax.set_ylim(6.1, -.6)
    ax.set_xlim(-2.5, 17)
    ax.set_xlabel("冻结温度 − 自适应温度：主要晚期窗口回报差 / 千")
    ax.grid(axis="x", color="#EEEEEE", lw=.6)
    save(fig, "fig3_3_temperature_paired", "冻结温度缓解后期退化，但并非每个种子均获益", [data["source"]], "5个配对种子原始差值；汇总采用原报告df=4的近似t区间，不替换为第500日端点")


def csbapr():
    src = "CS-BAPR/paper/source_data/reported_tables.csv"
    data = rows(src, "csbapr_reported_tables.csv")
    methods = [("bapr-pristine", "原始BAPR"), ("bapr", "早期BAPR"), ("csbapr-no-nau", "ReLU"),
               ("csbapr", "NAU/NMU"), ("csbapr-tanh", "tanh"), ("csbapr-kan", "KAN")]
    fig, axes = plt.subplots(1, 2, figsize=(5.75, 4.1), sharey=True)
    fig.subplots_adjust(left=.13, right=.98, bottom=.28, top=.90, wspace=.15)
    for ax, table, levels, title in zip(axes, ("2", "3"), ([1, 2, 5, 10, 20, 50], [5, 10, 20, 50]), ("(a) 固定需求放大", "(b) 回合内突变")):
        for (method, label), color, marker in zip(methods, COLORS, "os^DvX"):
            selected = [r for r in data if r["table"] == table and r["method"] == method]
            assert len(selected) == len(levels)
            values = {int(r["metric"].split()[0].replace("m=", "")) if table == "2" else int(r["metric"].split()[1][:-1]): float(r["value"]) for r in selected}
            ax.plot(range(len(levels)), [values[v] for v in levels], color=color, marker=marker, lw=1, ms=4, label=label)
        ax.set_xticks(range(len(levels)), [str(v) for v in levels])
        ax.set_xlabel("需求倍数（离散设置）")
        ax.set_title(title)
        ax.grid(axis="y", color="#E5E5E5", lw=.6)
    axes[0].set_ylabel("回报中位数 / 千")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, ncol=3, loc="lower center", bbox_to_anchor=(.5, .005), frameon=False, fontsize=9)
    save(fig, "fig4_1_csbapr_demand", "固定放大与突变需求下不同结构的收益存在条件差异", [src], "每点为10个种子的中位数；需求倍数为等距排列的离散条件，连线仅辅助追踪；无误差带")



def transitduet_results():
    data = PUBLISHED["transitduet"]["rows"]
    fig, axes = plt.subplots(1, 2, figsize=(5.75, 3.9), sharey=True)
    fig.subplots_adjust(left=.26, right=.98, bottom=.16, top=.85, wspace=.32)
    for ax, column, title, population, color, limit in zip(
        axes, [1, 3], ["(a) 候车时间", "(b) 总出行时间"],
        ["已上车乘客", "已完成乘客"], [BLUE, GREEN], [10, 24]
    ):
        ax.errorbar([r[column] for r in data], range(len(data)),
                    xerr=[r[column + 1] for r in data], fmt="o", color=color,
                    ms=4, capsize=3, lw=1)
        ax.set(yticks=range(len(data)), yticklabels=[r[0] for r in data],
               ylim=(len(data)-.5, -.5), xlabel="平均时间 / 分钟", xlim=(0, limit))
        ax.set_title(f"{title}\n{population}", fontsize=9, pad=9)
        ax.grid(axis="x")
    save(fig, "fig4_3_transitduet_results", "不同乘客集合的候车与总出行时间分别比较",
         [PUBLISHED["transitduet"]["source"]],
         "10个种子组，每组50个测试回合；点为均值、误差线为种子组均值的样本标准差；两指标不堆叠，规则不含训练重复")


def offline_results():
    src = "offline-sumo/paper/generated/figure3_operational_tradeoff_data.csv"
    data = [r for r in rows(src, "offline_operational.csv") if r["controller_group"] == "Learned controllers"]
    assert len(data) == 8 and all(int(r["n_replicates"]) == 5 for r in data)
    fig, axes = pair_panels([r["display_method"].replace("OnlineSAC", "在线SAC") for r in data], height=4.5, left=.235)
    for ax, prefix, title in zip(axes, ("headway_abs_dev_mean_s", "passenger_total_mean_s"), ("(a) 车头时距绝对偏差", "(b) 乘客总时间")):
        for y, row in enumerate(data):
            ax.errorbar(float(row[prefix+"_mean"]), y, xerr=float(row[prefix+"_std"]), fmt="o", color=BLUE if row["method"] in ("BC", "CQL", "RE-SAC") else GREEN, capsize=2, ms=4)
        ax.set_title(title, fontsize=9)
        ax.set_xlabel("时间 / 秒\n越小越好")
    fig.legend(handles=[Line2D([0], [0], marker="o", color=BLUE, lw=0, label="离线数据"), Line2D([0], [0], marker="o", color=GREEN, lw=0, label="新增目标交互")], loc="lower center", bbox_to_anchor=(.55, 0), ncol=2, frameon=False, fontsize=9)
    fig.subplots_adjust(bottom=.24)
    save(fig, "fig5_1_offline_tradeoff", "更规则的车头时距不必然对应更低的乘客时间", [src], "事后选定检查点；5个训练种子，每种子10个完整SUMO回合；均值及跨训练种子标准差；新增目标交互组30,000次决策")



def h2o_results():
    data = PUBLISHED["h2o"]["rows"]
    fig, axes = pair_panels([r[0] for r in data], height=3.7, left=.22)
    for y, row in enumerate(data):
        color = BLUE if y == 3 else GRAY
        axes[0].scatter(row[1], y, color=color, s=32, marker="D" if y == 3 else "o", zorder=3)
        axes[1].scatter(row[2]*100, y, color=color, s=32, marker="D" if y == 3 else "o", zorder=3)
    for ax in axes:
        ax.axvline(0, color=GRAY, ls="--", lw=.7)
        ax.axhline(2.5, color="#CCCCCC", lw=.7)
    axes[0].set(xlim=(-175, 15), xlabel="时间差 / 秒\n负值表示时间减少", title="(a) 每出发乘客时间差")
    axes[1].set(xlim=(-.42, .9), xlabel="完成率差 / 百分点\n正值表示完成率提高", title="(b) 行程完成率差")
    axes[1].axvline(-.3, color=RED, ls=":", lw=1)
    axes[1].text(-.3, -.62, "非劣界", color=RED, fontsize=8.5, ha="center")
    axes[0].set_title("(a) 每出发乘客时间差", fontsize=9)
    axes[1].set_title("(b) 行程完成率差", fontsize=9)
    fig.subplots_adjust(top=.85, bottom=.23)
    save(fig, "fig5_3_h2o_paired_effect", "完成保障候选相对直接参照的服务增量", [PUBLISHED["h2o"]["source"]], "展示原报告的配对均值，无误差线；总体推断单位为10个独立环境块；完成率非劣界为-0.3个百分点")



ENVIRONMENTS = [
    ("ant_wall_broken_soft_ceiling", "Ant执行器变化"),
    ("gravity_soft_ceiling", "HalfCheetah重力变化"),
    ("friction_walker_soft_ceiling", "Walker2d摩擦变化"),
]


def mcwm_curves():
    src = "MC-WM/paper/generated/final_confirmation/learning_curve_summary.csv"
    data = rows(src, "mcwm_learning_curves.csv")
    methods = [("mcwm_calibration_risk_target_done", "校准选择MC-WM"), ("mcwm_core", "固定残差核心"),
               ("direct_target_mbpo", "直接目标MBPO"), ("target_sac", "目标SAC"), ("darc", "DARC"), ("raw_sim", "原始源策略")]
    fig, axes = plt.subplots(3, 1, figsize=(5.75, 6.0), sharex=True)
    fig.subplots_adjust(left=.15, right=.97, bottom=.21, top=.955, hspace=.42)
    for index, (ax, (env, label)) in enumerate(zip(axes, ENVIRONMENTS)):
        for (method, name), color, marker in zip(methods, COLORS, "os^DvX"):
            selected = sorted([r for r in data if r["environment"] == env and r["method"] == method], key=lambda r: int(r["online_step"]))
            assert selected and all(int(r["n_seeds"]) == 10 for r in selected)
            x = mcwm_progress(selected, method)
            mean, lo, hi = [np.array([float(r[key]) for r in selected]) for key in ("return_mean", "return_ci_low", "return_ci_high")]
            ax.plot(x, mean, label=name, color=color, marker=marker, markersize=3, linewidth=1)
            ax.fill_between(x, lo, hi, color=color, alpha=.10, linewidth=0)
        ax.set_title(f"({chr(97+index)}) {label}", loc="left")
        ax.set_ylabel("验证回报")
        ax.grid(color="#EEEEEE", lw=.6)
    axes[-1].set(xlabel="预设训练预算完成比例 / %", xticks=[25, 50, 75, 100], xlim=(10, 102))
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, ncol=2, loc="lower center", bbox_to_anchor=(.53, .01), frameon=False, fontsize=9)
    save(fig, "fig6_2_mcwm_learning", "在预设的相同训练阶段比较不同机制的验证过程", [src], "10个种子，均值及原CSV的95%自助法区间；8个预设检查点：适应组在线阶段40000目标步，纯源组50000源步；横轴不包含验证成本，实际访问量另图报告；未插值")


def mcwm_progress(selected, method):
    budget = 50000 if method == "raw_sim" else 40000
    steps = np.array([int(r["online_step"]) for r in selected])
    assert np.array_equal(steps, np.arange(1, 9) * (budget // 8)), (method, steps)
    return steps / budget * 100


def mcwm_choice():
    src = "MC-WM/paper/generated/final_exact_routing/exact_routing_counterfactuals.csv"
    data = rows(src, "mcwm_exact_paired.csv")
    labels = ["Ant执行器：选择直接模型", "HalfCheetah重力：选择直接模型", "Walker2d摩擦：选择残差模型"]
    fig, axes = plt.subplots(3, 2, figsize=(5.75, 5.5))
    fig.subplots_adjust(left=.10, right=.97, top=.86, bottom=.20, hspace=.95, wspace=.28)
    fig.text(.31, .95, "(a) 回报差", ha="center", fontsize=10)
    fig.text(.77, .95, "(b) 平均违例数差", ha="center", fontsize=10)
    for y, summary in enumerate(PUBLISHED["exact_routing"]["rows"]):
        selected = sorted([r for r in data if r["environment"] == summary[0]], key=lambda r: int(r["seed"]))
        assert len(selected) == 10
        for ax, key, offset in zip(axes[y], ("automatic_minus_alternative_return", "automatic_minus_alternative_violations"), (1, 4)):
            values = [float(r[key]) for r in selected]
            ax.boxplot(values, vert=False, positions=[0], widths=.14, whis=(0,100),
                       patch_artist=True, showfliers=False, manage_ticks=False,
                       boxprops=dict(facecolor="#DFEAF1",edgecolor=BLUE,lw=.8),
                       medianprops=dict(color=BLUE,lw=1.2),
                       whiskerprops=dict(color=GRAY,lw=.8),capprops=dict(color=GRAY,lw=.8))
            ax.scatter(values, np.linspace(-.10,.10,10), color=GRAY, s=11, alpha=.75, zorder=3)
            mean, low, high = summary[offset:offset+3]
            assert abs(np.mean(values) - mean) < .06
            ax.errorbar(mean, .25, xerr=[[mean-low], [high-mean]], fmt="D", color=BLUE, capsize=3, ms=4, zorder=3)
            ax.axvline(0, color=GRAY, ls="--", lw=.8)
            ax.set_ylim(.39, -.22)
            ax.set_yticks([])
            ax.spines["left"].set_visible(False)
            ax.grid(axis="x", color="#EEEEEE", lw=.6)
            ax.xaxis.set_major_locator(matplotlib.ticker.MaxNLocator(4))
        axes[y, 0].set_title(labels[y], loc="left", fontsize=9)
    axes[-1, 0].set_xlabel("自动选择 − 未选类别\n正值表示回报增加")
    axes[-1, 1].set_xlabel("自动选择 − 未选类别\n负值表示违例减少")
    fig.legend(handles=[Line2D([0], [0], marker="o", color=GRAY, lw=0, markersize=4, label="单个配对种子"), Line2D([0], [0], marker="D", color=BLUE, lw=1, markersize=4, label="均值与95%区间")], loc="lower center", bbox_to_anchor=(.53, .01), ncol=2, frameon=False, fontsize=9)
    save(fig, "fig6_3_mcwm_model_choice", "模型选择的回报收益因环境而异，违例代价也须独立考察", [src, PUBLISHED["exact_routing"]["source"]], "每环境10个配对种子；箱体25%-75%、中线为中位数、须为最小最大值；灰点为原始差值，菱形及线段为原结果表均值和95%区间；各环境独立横轴范围")


def main():
    OUT.mkdir(exist_ok=True)
    thesis_schematics.render(sys.modules[__name__])
    for render in (network, resac_results, temperature, csbapr,
                   transitduet_results, offline_results, h2o_results,
                   mcwm_curves, mcwm_choice):
        render()
    additional_figures.render(sys.modules[__name__])
    from mechanism_figures import bapr_comparison
    bapr_comparison(sys.modules[__name__])
    (OUT / "figure_sources.json").write_text(json.dumps(MANIFEST, ensure_ascii=False, indent=2) + "\n")
    sheet = Image.new("RGB", (1800, ((len(MANIFEST)+3)//4)*650), "#EEEEEE")
    draw = ImageDraw.Draw(sheet)
    for i, item in enumerate(MANIFEST):
        image = Image.open(OUT / f"{item['figure']}.png").convert("RGB")
        image = ImageOps.contain(image, (430, 610))
        x, y = (i % 4)*450, (i // 4)*650
        sheet.paste(image, (x+(450-image.width)//2, y+25))
        draw.text((x+8, y+5), item["figure"], fill="black")
    sheet.save(ROOT / "output/figure_contact_sheet.jpg", quality=92)


if __name__ == "__main__":
    main()
