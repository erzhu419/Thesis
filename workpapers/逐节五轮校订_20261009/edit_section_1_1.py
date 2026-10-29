from pathlib import Path
from shutil import copy2
from zipfile import ZipFile
from xml.etree import ElementTree as ET
from html import escape
import re


ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "output/排版完成_参考文献复核_20261007_导师批改.docx"
OUTPUT = ROOT / "output/排版完成_逐节校订_第1轮_20261009.docx"
BACKUP = Path(__file__).with_name("校订前_导师批改.docx")
W = "{http://schemas.openxmlformats.org/wordprocessingml/2006/main}"
W14 = "{http://schemas.microsoft.com/office/word/2010/wordml}"

# Each tuple contains the prose before and, where present, after existing citations.
REVISIONS = {
    253: (
        "公交服务质量不仅取决于班次和车辆的配置，还取决于车辆在运行过程中能否保持连续、均衡的服务。同一线路的车辆受到路段拥堵、信号等待和乘客上下车的影响，即使从起点按等间隔发车，沿途车头时距仍可能逐渐失衡。延误车辆与前车的间隔增大，积累的候车乘客增多，上下客时间随之延长；后续车辆面对较少的候车乘客，反而更快接近前车。这一相互强化的过程使局部延误发展为串车，并增加乘客等待和车辆载客不均衡。已有车头时距控制与动态驻站研究围绕这一问题，分析了反馈控制对线路稳定性和服务效率的影响。",
    ),
    254: (
        "公交在途控制通过调整车辆运行中的动作减轻扰动，驻站控制是其中一种直接方式：车辆完成必要上下客后，根据相邻车辆间隔、服务目标和车内外乘客状态，决定是否额外停留以及停留多久。额外驻站可能改善后续车辆的到站间隔，减少部分乘客的候车时间，但也会增加车内乘客的停留时间。因此，控制效果不能只看车头时距是否均衡，还应结合候车时间、车内时间、车辆周转和行程完成情况判断。Delgado等关于驻站与限制上客的联合研究表明，控制方式的适用性与需求强度、车辆容量和相邻车辆状态有关。",
    ),
    255: (
        "在双向、按时刻表运行的公交系统中，上述服务权衡还与车辆的班次接续有关。同一车辆完成一个班次后可能承担新的班次，沿途驻站不仅影响当前乘客，也可能影响车辆何时可用于后续发车。多线路系统中的共享站点和共同走廊，又使不同线路的运行相互影响。因此，驻站控制不是孤立地调整一辆车的停留时间，而是在运行计划、线路结构和乘客服务约束下，判断当前动作及其后续影响。",
    ),
    256: (
        "强化学习通过环境交互学习驻站决策，把当前运行状态与动作的后续服务影响联系起来。已有公交研究采用多智能体强化学习，并针对不同车辆在不同时间到站的决策特点设计学习方法，进一步将控制目标扩展至多线路协同和多种运营指标。",
        "这些研究为利用更丰富的运行信息学习驻站策略提供了基础，也提出了策略在训练条件之外能否继续有效的问题。",
    ),
    257: (
        "公交车辆在不同站点、班次和时段触发决策，各车辆积累的运行经验并不均衡。本文首先利用类别信息保留车辆的服务位置和运行背景，使不同车辆能够共享控制经验。在此基础上，需要区分两类不确定性：一是在给定客流与路况规律下，乘客到达和车辆运行仍存在随机波动；二是大型活动、交通管制等使原有运行规律发生突变或持续变化。前者要求策略合理评价随机后果，后者还要求判断运行条件的变化，并使驻站响应适应新的条件。",
    ),
    258: (
        "仿真使策略训练能够在不干预实际运营的条件下反复进行。本文自建的快速公交仿真以事件推进运行过程，为大量训练交互提供较低成本；微观交通仿真则进一步描述道路、车辆和路口之间的相互作用。SUMO提供了开放的微观交通仿真基础，可用于建立公交场景及控制接口。",
        "本文将SUMO作为高保真代理环境，用于采集目标离线数据，并检验策略在更复杂运行条件下的服务效果。",
    ),
    259: (
        "低成本仿真中有效的策略，在使用环境中不一定取得相同效果。本文将承担训练交互的仿真称为源环境，将策略接受运行评价的环境称为目标环境。二者可能在道路速度、客流起讫点分布、路口通行延误和必要停站服务时间等方面存在差异。本文关注这些差异造成的分布偏移，尤其是相同状态和动作下后继运行结果的分布变化，即动力学偏移。动力学随机化研究通过在仿真训练中改变模型参数改善迁移，说明训练分布的构造能够影响策略的现实适应能力。",
        "但在缺少目标数据时，仍需事先判断应将哪些变化纳入训练；获得目标离线数据后，则可以依据目标运行记录约束学习过程，或修正源仿真的预测。",
    ),
    260: (
        "H2O及其扩展H2O+将预先采集的目标离线数据与源仿真中新产生的交互经验结合，提供了不同于纯离线学习和目标环境在线微调的数据利用方式。",
        "本文借鉴这一混合训练框架，将H2O+适配于公交驻站控制：对齐异步到站事件，表示不额外驻站与额外驻站的动作选择，并统一运行时域内乘客时间和行程完成情况的评价。源仿真仍需提供动作后的运行结果；若标定不充分，仅混合两类经验未必能消除转移偏差，因而还需要研究如何利用目标数据修正预测。",
    ),
    261: (
        "本论文以“从仿真到现实的公交系统驻站控制方法研究”为题，按照目标数据的可获得程度和环境变化逐步展开。训练前未获得目标域转移数据时，先研究RE-SAC如何应对给定运行规律下的随机波动，再利用BAPR识别运行条件，并通过CS-BAPR的结构化策略应对非平稳变化；获得固定目标离线数据后，将H2O+适配于公交任务，结合目标记录与低成本源仿真训练；进一步针对源仿真标定不足，研究MC-WM如何选择和修正世界模型，并利用模型预测辅助驻站决策。上述研究以车头时距和乘客服务指标检验控制效果，旨在形成与目标信息条件相适应的公交驻站学习方法。",
    ),
}


def paragraph_text(paragraph):
    return "".join(node.text or "" for node in paragraph.iter(W + "t"))


def revise_block(block, prose, paragraph):
    parents = {child: parent for parent in paragraph.iter() for child in parent}
    nodes = list(paragraph.iter(W + "t"))
    groups, current = [], []
    for index, node in enumerate(nodes):
        ancestor = parents[node]
        ordinary = True
        while ancestor is not paragraph:
            if ancestor.tag == W + "hyperlink":
                ordinary = False
            if ancestor.tag == W + "r":
                alignment = ancestor.find(W + "rPr/" + W + "vertAlign")
                if alignment is not None:
                    ordinary = False
                assert ancestor.find(W + "rPr") is None or not ordinary
            ancestor = parents[ancestor]
        if ordinary:
            current.append(index)
        elif current:
            groups.append(current)
            current = []
    if current:
        groups.append(current)
    assert len(groups) == len(prose), (groups, prose)
    replacements = {}
    for group_index, (indices, text) in enumerate(zip(groups, prose)):
        replacements[indices[0]] = (" " if group_index else "") + text
        replacements.update({index: "" for index in indices[1:]})
    position = -1

    def replace_text(match):
        nonlocal position
        position += 1
        if position not in replacements:
            return match.group(0)
        return match.group(1) + escape(replacements[position], quote=False) + match.group(3)

    pattern = r"(<w:t(?:\s[^>]*)?>)(.*?)(</w:t>)"
    revised = re.sub(pattern, replace_text, block, flags=re.S)
    assert position + 1 == len(nodes)
    return revised


def main():
    with ZipFile(SOURCE) as archive:
        original_xml = archive.read("word/document.xml").decode("utf-8")
        original_body = ET.fromstring(original_xml).find(W + "body")
        updated_xml = original_xml
        for index, prose in REVISIONS.items():
            paragraph_id = original_body[index].get(W14 + "paraId")
            pattern = r'<w:p\b[^>]*w14:paraId="' + paragraph_id + r'"[^>]*>.*?</w:p>'
            matches = list(re.finditer(pattern, updated_xml, flags=re.S))
            assert len(matches) == 1, (index, paragraph_id)
            match = matches[0]
            revised = revise_block(match.group(0), prose, original_body[index])
            updated_xml = updated_xml[:match.start()] + revised + updated_xml[match.end():]
        updated_body = ET.fromstring(updated_xml).find(W + "body")
        assert len(updated_body) == len(original_body)
        changed = [i for i, (a, b) in enumerate(zip(original_body, updated_body))
                   if ET.tostring(a) != ET.tostring(b)]
        assert changed == list(REVISIONS), changed
        if not BACKUP.exists():
            copy2(SOURCE, BACKUP)
        with ZipFile(OUTPUT, "w") as target:
            for item in archive.infolist():
                data = updated_xml.encode("utf-8") if item.filename == "word/document.xml" else archive.read(item.filename)
                target.writestr(item, data)
    with ZipFile(SOURCE) as before, ZipFile(OUTPUT) as after:
        assert before.namelist() == after.namelist()
        changed_parts = [name for name in before.namelist() if before.read(name) != after.read(name)]
        assert changed_parts == ["word/document.xml"], changed_parts
    print("Changed body paragraphs:", changed)
    print("All paragraphs outside section 1.1 unchanged, including both abstracts.")
    print("Other DOCX parts unchanged: images, styles, equations, comments and relationships.")
    for index in changed:
        print(index, paragraph_text(updated_body[index]))
    print("OUTPUT", OUTPUT)


if __name__ == "__main__":
    main()
