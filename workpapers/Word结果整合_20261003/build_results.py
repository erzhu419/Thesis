"""Create mean-only thesis figures from the completed server summaries."""

import json
from pathlib import Path
from statistics import mean

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
OUT = HERE / "figures"
OUT.mkdir(exist_ok=True)


def read(name):
    return json.loads((ROOT / "experiments" / name / "summary.json").read_text())


def reduction(reference, result):
    return 100 * (reference - result) / abs(reference)


def results():
    legacy = read("bus_legacy_dual_v1")
    followup = read("bus_alignment_followup_v5")
    aligned = read("bus_dual_aligned_v4")
    assert legacy["complete"] and followup["complete"] and aligned["complete"]
    re = []
    for objective in ("headway", "passenger"):
        field = "plan_error_seconds" if objective == "headway" else "cost_seconds"
        for mode in ("holding", "dual"):
            for scene in ("normal", "demand_surge", "fast_switch"):
                values = {}
                for arm in ("ensemble_sac_control", "resac"):
                    rows = [r for r in legacy["seed_means"] if
                        (r["kind"], r["mode"], r["objective"], r["scene"], r["arm"])
                        == ("resac", mode, objective, scene, arm)]
                    assert len(rows) == 5 and all(r["cases"] == 3 for r in rows)
                    values[arm] = mean(r[field] for r in rows)
                re.append(dict(objective=objective, mode=mode, scene=scene,
                    control=values["ensemble_sac_control"], resac=values["resac"],
                    reduction=reduction(values["ensemble_sac_control"], values["resac"])))
    cs = [r for r in followup["cs_aggregates"] if r["family"] == "cs_recipe"
          and r["arm"] in ("pristine_mlp", "stabilized_mlp", "stabilized_nau")]
    assert len(cs) == 72 and all(r["complete"] for r in cs)
    dual = []
    for objective in ("passenger", "headway"):
        source = (aligned["results"] if objective == "passenger"
                  else followup["mc_aggregates"])
        for scene in ("calibrated", "joint75"):
            for arm in ("h2oplus", "mcwm"):
                rows = [r for r in source if r["scene"] == scene and r["arm"] == arm
                    and r["layer"] == "guarded" and
                    (r.get("objective") == objective if objective == "passenger"
                     else r["variant"] == "clip10")]
                assert len(rows) == 1 and rows[0]["cases"] == 20
                dual.append({**rows[0], "objective":objective})
    rules = dict(cost=2684.6370958278, ridge=-324.9248689375968)
    # Both objectives use the same resident Daganzo evaluation endpoints.
    for r in aligned["results"]:
        if r["layer"] == "guarded":
            assert abs(r["cost"] - r["cost_delta_vs_daganzo"] - rules["cost"]) < 1e-7
            assert abs(r["ridge"] - r["ridge_delta_vs_daganzo"] - rules["ridge"]) < 1e-7
    bundle = dict(resac=re, cs=cs, dual=dual, daganzo=rules)
    (HERE / "integration_values.json").write_text(json.dumps(bundle, ensure_ascii=False, indent=2)+"\n")
    return bundle


def style():
    font = "/mnt/c/Windows/Fonts/msyh.ttc"
    font_manager.fontManager.addfont(font)
    plt.rcParams.update({"font.family":font_manager.FontProperties(fname=font).get_name(),
        "font.size":10, "axes.titlesize":10, "axes.labelsize":10,
        "xtick.labelsize":9, "ytick.labelsize":9, "axes.unicode_minus":False,
        "axes.spines.top":False, "axes.spines.right":False,
        "legend.frameon":False, "axes.axisbelow":True, "pdf.fonttype":42,
        "savefig.facecolor":"white", "figure.facecolor":"white"})


def save(fig, name):
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    for ax in fig.axes:
        box = ax.get_tightbbox(renderer)
        assert box.x0 >= -1 and box.y0 >= -1, name
        assert box.x1 <= fig.bbox.width+1 and box.y1 <= fig.bbox.height+1, name
    for text in fig.findobj(matplotlib.text.Text):
        if not text.get_visible() or not text.get_text() or text.get_clip_on():
            continue
        box = text.get_window_extent(renderer)
        if box.width and box.height:
            assert box.x0 >= -1 and box.y0 >= -1, (name, text.get_text())
            assert box.x1 <= fig.bbox.width+1 and box.y1 <= fig.bbox.height+1, (name, text.get_text())
    for suffix in ("png", "pdf"):
        fig.savefig(OUT / f"{name}.{suffix}", dpi=360)
    plt.close(fig)


def make_figures(data):
    style()
    blue, green, gold = "#286A98", "#27877C", "#B68A37"
    fig, axes = plt.subplots(1, 2, figsize=(7, 3.4))
    fig.subplots_adjust(left=.1, right=.98, bottom=.23, top=.83, wspace=.4)
    for ax, obj, title in zip(axes, ("headway", "passenger"),
                             ("(a) 计划时距偏差", "(b) 人均广义时间")):
        for offset, mode, color, label in ((-.18,"holding",blue,"仅驻站"),(.18,"dual",green,"驻站与速度")):
            rows = [r for r in data["resac"] if r["objective"] == obj and r["mode"] == mode]
            bars = ax.bar(np.arange(3)+offset, [r["reduction"] for r in rows], .34,
                          color=color, label=label)
            ax.bar_label(bars, fmt="%.1f", padding=3, fontsize=8)
        ax.set(title=title, xticks=np.arange(3), xticklabels=["正常", "客流激增", "快速切换"],
               ylabel="RE-SAC相对集成SAC的降幅 / %", ylim=(0,26))
        ax.grid(axis="y", alpha=.2)
    fig.legend(*axes[0].get_legend_handles_labels(), loc="upper center", bbox_to_anchor=(.5,.99), ncol=2)
    save(fig, "fig3_13_objectives_actions")

    fig, axes = plt.subplots(1, 2, figsize=(7, 4.1))
    fig.subplots_adjust(left=.15, right=.98, bottom=.15, top=.84, wspace=.65)
    scenes = ("normal","pulse5","repeated5","static5","static10","burst50")
    labels = ["正常", "单次5倍", "重复5倍", "持续5倍", "持续10倍", "短时50倍"]
    for ax, mode, title in zip(axes, ("holding","dual"), ("(a) 仅驻站", "(b) 驻站与速度")):
        for offset, arm, color, label in ((-.18,"stabilized_mlp",blue,"稳定化感知机"),
                                        (.18,"stabilized_nau",gold,"稳定化算术输出")):
            values = []
            for scene in scenes:
                def get(a):
                    return next(r for r in data["cs"] if r["mode"]==mode and r["objective"]=="headway"
                                and r["scene"]==scene and r["arm"]==a)
                values.append(reduction(-get("pristine_mlp")["recorded_reward_per_decision"],
                                        -get(arm)["recorded_reward_per_decision"]))
            bars = ax.barh(np.arange(6)+offset, values, .34, color=color, label=label)
            ax.bar_label(bars, fmt="%.1f", padding=3, fontsize=8)
        ax.set(title=title, yticks=np.arange(6), yticklabels=labels, xlim=(0,80),
               xlabel="平均时距惩罚降幅 / %")
        ax.invert_yaxis()
        ax.grid(axis="x", alpha=.2)
    fig.legend(*axes[0].get_legend_handles_labels(), loc="upper center", bbox_to_anchor=(.5,.99), ncol=2)
    save(fig, "fig4_11_stabilization")

    fig, axes = plt.subplots(1, 2, figsize=(7, 3.5))
    fig.subplots_adjust(left=.1, right=.98, bottom=.2, top=.83, wspace=.4)
    for ax, objective, metric, title in zip(axes, ("passenger","headway"), ("cost","ridge"),
            ("(a) 乘客时间目标", "(b) 车头时距目标")):
        for offset, arm, color, label in ((-.18,"h2oplus",blue,"H2O+"),(.18,"mcwm",green,"MC-WM")):
            rows = [r for r in data["dual"] if r["objective"]==objective and r["arm"]==arm]
            vals = [reduction(abs(data["daganzo"][metric]),abs(r[metric])) for r in rows]
            bars = ax.bar(np.arange(2)+offset, vals, .34, color=color, label=label)
            ax.bar_label(bars, fmt="%.2f", padding=3, fontsize=8)
        ax.axhline(0, color="#666666", linewidth=.8)
        ax.set(title=title, xticks=np.arange(2), xticklabels=["正常标定", "联合75%偏差"],
               ylabel="相对Daganzo规则的目标指标降幅 / %", ylim=(-2.3,6.1))
        ax.grid(axis="y", alpha=.2)
    fig.legend(*axes[0].get_legend_handles_labels(), loc="upper center", bbox_to_anchor=(.5,.99), ncol=2)
    save(fig, "fig6_18_dual_objectives")


if __name__ == "__main__":
    data = results()
    make_figures(data)
    print("Verified means; exported three paired-objective figures.")
