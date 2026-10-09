"""Replace F-2 in the latest Word with the completed paired controller results."""
from pathlib import Path
from zipfile import ZipFile
import copy
import csv
import json
import statistics

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager as fm
import numpy as np
from lxml import etree as E

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
SOURCE = ROOT / "排版完成_综述补充50篇中文文献_20261006.docx"
BACKUP = HERE / "修改前_最新交付版.docx"
TARGET = ROOT / "排版完成_F2最终控制器结果更新_20261006.docx"
DATA = ROOT / "workpapers/最终控制器同条件比较_20261006/analysis.json"
report = json.loads(DATA.read_text())
assert report["status"] == "complete" and report["completed_evaluations"] == 225
assert SOURCE.read_bytes() == BACKUP.read_bytes(), "The latest manuscript changed after backup"

for font in ("/mnt/c/Windows/Fonts/simsun.ttc", "/mnt/c/Windows/Fonts/times.ttf"):
    fm.fontManager.addfont(font)
plt.rcParams.update({"font.family": ["Times New Roman", "SimSun"], "font.size": 11,
    "axes.unicode_minus": False, "axes.linewidth": .7, "axes.labelsize": 11,
    "xtick.labelsize": 10.5, "ytick.labelsize": 10.5, "pdf.fonttype": 42,
    "savefig.facecolor": "white"})
methods = ["mcwm_aligned", "h2oplus_final", "sac_guarded"]
labels = ["MC-WM", "H2O+", "纯源 SAC"]
colors = ["#236C9B", "#C18436", "#64717B"]
fig = plt.figure(figsize=(7.2, 7.2 * 4033078 / 5270500))
grid = fig.add_gridspec(2, 2, left=.125, right=.965, bottom=.11, top=.91,
                       hspace=.70, wspace=.38, height_ratios=[1, 1.03])
ax_time, ax_completion = fig.add_subplot(grid[0, 0]), fig.add_subplot(grid[0, 1])
source_rows = []
for ax, metric, title, ylabel in [
    (ax_time, "cost_seconds", "(a) 乘客时间", "人均广义时间（秒）"),
    (ax_completion, "completion_percent", "(b) 服务完成情况", "乘客完成率（%）"),
]:
    for x, (method, color) in enumerate(zip(methods, colors)):
        values = [report["method_means_by_training_seed"][method][str(s)][metric]
                  for s in report["training_seeds"]]
        mean = statistics.mean(values)
        half = 2.7764451051977987 * statistics.stdev(values) / np.sqrt(5)
        ax.scatter(x + np.linspace(-.14, .14, 5), values, s=22, facecolors="white",
                   edgecolors=color, linewidths=.9, zorder=3)
        ax.errorbar(x, mean, yerr=half, fmt="D", color=color, markersize=5,
                    capsize=4, linewidth=1.3, zorder=4)
        for seed, value in zip(report["training_seeds"], values):
            source_rows.append([metric, method, seed, value])
        ax.text(x, 1.01, f"{mean:.2f}", color=color, ha="center", va="bottom",
                transform=ax.get_xaxis_transform(), fontsize=11)
    ax.set_xticks(range(3), labels)
    ax.set_xlim(-.5, 2.5)
    ax.set_ylabel(ylabel)
    ax.set_title(title, loc="left", pad=24, fontsize=12, fontweight="normal")
    ax.grid(axis="y", color="#E4E7EA", linewidth=.5)
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(direction="out", length=3)
ax_time.set_ylim(2325, 2460)
ax_time.set_yticks([2340, 2380, 2420, 2460])
ax_completion.set_ylim(96.1, 97.1)
ax_completion.set_yticks([96.2, 96.5, 96.8, 97.1])

ax = fig.add_subplot(grid[1, :])
position = ax.get_position()
ax.set_position([.30, position.y0, .665, position.height])
comparisons = [
    ("mcwm_aligned_minus_sac_guarded", "MC-WM − 纯源 SAC", colors[0]),
    ("mcwm_aligned_minus_h2oplus_final", "MC-WM − H2O+", colors[0]),
    ("h2oplus_final_minus_sac_guarded", "H2O+ − 纯源 SAC", colors[1]),
]
for y, (key, label, color) in enumerate(comparisons):
    c = report["comparisons"][key]
    mean = c["mean_difference"]["cost_seconds"]
    low, high = c["training_seed_paired_t95_interval"]["cost_seconds"]
    values = [c["by_training_seed"][str(s)]["cost_seconds"] for s in report["training_seeds"]]
    ax.scatter(values, y + np.linspace(-.10, .10, 5), s=20, facecolors="white",
               edgecolors=color, linewidths=.8, zorder=3)
    ax.errorbar(mean, y, xerr=[[mean-low], [high-mean]], fmt="D", color=color,
                markersize=5, capsize=4, linewidth=1.4, zorder=4)
    ax.text(55, y, f"{mean:+.2f}", ha="right", va="center", fontsize=11, color=color)
    source_rows.append(["paired_cost_mean", key, "five_seeds", mean])
    source_rows.append(["paired_cost_ci_low", key, "five_seeds", low])
    source_rows.append(["paired_cost_ci_high", key, "five_seeds", high])
ax.axvline(0, color="#595959", linewidth=.8, linestyle=(0, (3, 3)))
ax.set_yticks(range(3), [p[1] for p in comparisons])
ax.set_ylim(2.45, -.45)
ax.set_xlim(-95, 58)
ax.set_xticks([-80, -40, 0, 40])
ax.set_xlabel("人均广义时间差（秒；负值表示前者更优）", labelpad=6)
fig.text(.125, position.y1 + .033, "(c) 同一训练种子下的配对差异",
         fontsize=12, fontweight="normal")
ax.spines[["top", "right", "left"]].set_visible(False)
ax.tick_params(axis="y", length=0, labelsize=10)
ax.grid(axis="x", color="#E4E7EA", linewidth=.5)
fig.savefig(HERE / "图F-2.png", dpi=600)
fig.savefig(HERE / "图F-2.pdf")
plt.close(fig)
with (HERE / "图F-2_作图数据.csv").open("w", encoding="utf-8-sig", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["metric", "method_or_comparison", "training_seed", "value"])
    writer.writerows(source_rows)

NS = {"w": "http://schemas.openxmlformats.org/wordprocessingml/2006/main",
      "m": "http://schemas.openxmlformats.org/officeDocument/2006/math"}
W = "{" + NS["w"] + "}"
with ZipFile(SOURCE) as z:
    infos = z.infolist()
    parts = {item.filename: z.read(item.filename) for item in infos}
doc = E.fromstring(parts["word/document.xml"])
body = doc.find("w:body", NS)
before = copy.deepcopy(doc)
changes = []

def text(p):
    return "".join(p.xpath(".//w:t/text()", namespaces=NS))

def replace_span(p, old, new):
    original = text(p)
    assert original.count(old) == 1, old
    start, end = original.index(old), original.index(old) + len(old)
    offset = 0
    inserted = False
    for t in p.xpath(".//w:t", namespaces=NS):
        value = t.text or ""
        stop = offset + len(value)
        if offset < end and stop > start:
            left = value[:max(0, start-offset)]
            right = value[max(0, end-offset):] if stop > end else ""
            t.text = left + (new if not inserted else "") + right
            inserted = True
        offset = stop
    assert text(p) == original.replace(old, new, 1)
    changes.append(dict(body_index=body.index(p), before=original, after=text(p)))

new_texts = {
2885: "本附录汇集公交世界模型的接入检验、控制效果比较和源标定偏差探索。第F.1与F.3节保留单训练种子的早期结果；第F.2节在统一训练与执行条件下，比较五个训练种子的最终控制器。第6.4.3至6.4.5节的多训练种子比较构成正文的主要公交证据。",
2897: "本节比较MC-WM、H2O+和纯源SAC的最终控制效果。三组采用相同的源仿真生成规则、需求轮换计划和名义25,000次源事件预算，分别保留五个训练种子的固定端点。MC-WM与H2O+共用对应种子完成50,000次IQL更新的离线初始化，再续训20,000次；纯源SAC从随机参数开始，仅使用源仿真经验。MC-WM采用归一化置信备份和固定0.25模型损失系数，各方法保留自身的完整训练日程。源仿真按0.75、1.0和1.25倍需求轮换，实际预算内三组均只进入前两个运行日；实际源事件数与名义值相差不超过1次。",
2898: "所有策略均在目标SUMO环境中评价，纯源仅指训练数据来源。评价采用五个新运行日与0.75、1.0和1.25倍需求，三种方法各含五个训练种子，共225回合。每回合在SUMO 1.25.0中运行24,000秒，同一运行日和需求下共用乘客输入；训练期间不追加目标SUMO交互。三组均采用第5.3节的共同停站执行规则，且不叠加额外的0.25动作缩放。",
2899: "图F-2同时展示人均广义时间、乘客完成率及方法间的配对差异。统计先在每个训练种子内等权平均十五个测试情形，再汇总五个种子。人均广义时间为两倍候车时间与乘车时间之和，以全部已出发乘客为分母；完成率单独报告。",
2902: "空心圆为各训练种子的均值，菱形为五个种子的总体均值，误差线为基于五个训练种子的95% t置信区间。图（c）先在同一种子内计算前一种方法减后一种方法的时间差，再汇总五个配对差；负值表示前者更优，区间跨零表示尚不能确定稳定优势。225回合不作为225个独立训练重复。",
2903: "MC-WM、H2O+和纯源SAC的人均广义时间分别为2,360.49、2,383.51和2,400.74秒，乘客完成率分别为96.8445%、96.6864%和96.5819%。MC-WM相对纯源SAC平均减少40.25秒（1.68%），五个训练种子均改善，配对95%区间为−65.35至−15.16秒；完成率提高0.2626个百分点。相对H2O+，MC-WM平均减少23.02秒（0.97%），三个种子改善、两个种子变差，区间为−69.23至23.19秒。H2O+相对纯源SAC平均减少17.23秒，四个种子改善，区间为−53.51至19.04秒。",
2904: "MC-WM相对H2O+的候车时间减少13.04秒，乘车时间增加3.05秒，因而广义时间净减少23.02秒。在1.25倍需求下，MC-WM相对H2O+和纯源SAC的平均时间分别减少42.55秒和78.44秒，表明本组收益主要体现在候车负担较重的情形。局限在于训练重复仅有五个种子，区间未作多重比较调整；纯源SAC不含离线预训练开销，各方法的训练日程也不完全相同。因此，该比较支持MC-WM相对纯源的控制收益及相对H2O+的平均改善，尚不足以认定其稳定优于H2O+，也不能把全部差异归因于世界模型单一部件。",
}
for index, new in new_texts.items():
    p = body[index]
    if index == 2899:
        old = text(p)
        prefix, suffix = old.split("图F-2")
        replace_span(p, prefix, "")
        replace_span(p, suffix, new.removeprefix("图F-2"))
        continue
    assert not p.xpath(".//m:oMath|.//w:instrText", namespaces=NS)
    replace_span(p, text(p), new)
replace_span(body[2896], "训练预算", "控制效果")
replace_span(body[249], "训练预算", "控制效果")
replace_span(body[2901], "离线数据规模与后续更新预算对目标公交服务的影响", "统一条件下不同控制方法的公交服务表现")
replace_span(body[1485], ".2节另在这一预算体系下比较模型辅助。", ".2节补充完整训练日程下的控制效果比较。")
replace_span(body[1653], "的模型组采用附加价值更新", ".1与F.3的模型组采用附加价值更新")

record = E.Element(W + "p")
record.append(copy.deepcopy(body[2897].find("w:pPr", NS)))
run = E.SubElement(record, W + "r")
E.SubElement(run, W + "t").text = (
    "第F.2节的最终控制器比较复用五个正式H2O+策略和五个纯源SAC策略，新增五条MC-WM续训。"
    "MC-WM复用H2O+对应种子的离线初始化；十五个固定策略在五个新运行日和三档需求下完成225个目标SUMO评价回合。"
    "该组使用独立生成的测试运行日，与前述20种子比较分别统计，复用策略不增加独立训练种子数。")
body.insert(len(body)-1, record)
changes.append(dict(body_index=body.index(record), before=None, after=text(record)))

parts["word/document.xml"] = E.tostring(doc, xml_declaration=True, encoding="UTF-8", standalone=True)
parts["word/media/image62.png"] = (HERE / "图F-2.png").read_bytes()
with ZipFile(TARGET, "w") as z:
    for item in infos:
        z.writestr(item, parts[item.filename])

# Detect accidental changes outside the explicitly revised prose and figure.
allowed = {row["body_index"] for row in changes if row["before"] is not None}
before_body = before.find("w:body", NS)
for i in range(len(before_body)-1):
    if i not in allowed:
        assert E.tostring(before_body[i]) == E.tostring(body[i]), i
for xpath in (".//m:oMath", ".//w:sectPr", ".//w:instrText", ".//w:bookmarkStart", ".//w:bookmarkEnd"):
    a, b = before.xpath(xpath, namespaces=NS), doc.xpath(xpath, namespaces=NS)
    assert [E.tostring(x) for x in a] == [E.tostring(x) for x in b], xpath
with ZipFile(SOURCE) as a, ZipFile(TARGET) as b:
    changed_parts = [n for n in a.namelist() if a.read(n) != b.read(n)]
assert sorted(changed_parts) == ["word/document.xml", "word/media/image62.png"]
(HERE / "修改记录.json").write_text(json.dumps(dict(source=str(SOURCE), target=str(TARGET),
    data=str(DATA), changed_parts=changed_parts, changes=changes,
    equations_fields_bookmarks_sections_preserved=True,
    all_other_body_elements_preserved=True), ensure_ascii=False, indent=2) + "\n")

# Small Word excerpt for native rendering using the document's own styles.
excerpt = copy.deepcopy(doc)
eb = excerpt.find("w:body", NS)
for child in list(eb)[:2884]:
    eb.remove(child)
excerpt_parts = dict(parts)
excerpt_parts["word/document.xml"] = E.tostring(excerpt, xml_declaration=True, encoding="UTF-8", standalone=True)
with ZipFile(HERE / "附录F_排版核对.docx", "w") as z:
    for item in infos:
        z.writestr(item, excerpt_parts[item.filename])
print(json.dumps(dict(word=str(TARGET), modified_paragraphs=len(changes), changed_parts=changed_parts), ensure_ascii=False))
