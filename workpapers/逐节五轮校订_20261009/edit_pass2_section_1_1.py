from pathlib import Path
from shutil import copy2
from zipfile import ZipFile
import re
import xml.etree.ElementTree as ET

from edit_section_1_2_1 import M, W
from edit_section_3_2 import prose_groups
from edit_section_4_4 import replace_paragraph


ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "output/排版完成_逐节校订_第1轮_20261009.docx"
DOCUMENT = ROOT / "output/排版完成_逐节校订_第2轮_20261010.docx"
BACKUP = Path(__file__).with_name("校订前_第2轮_1.1.docx")
REVISIONS = {
    257: (
        "公交车辆在不同站点、班次和时段决策，运行经验并不均衡。本文以类别信息和连续运行量表示状态，使各车共享经验时仍能区分运行背景。客流与路况规律不变时，乘客到达和车辆运行仍有随机波动；大型活动、交通管制等改变运行规律时，原有策略可能不再适用。前者需要评价动作的随机后果，后者还需根据近期信息判断变化，学习相应的驻站响应。",
    ),
    259: (
        "低成本仿真中有效的策略，在使用环境中不一定取得相同效果。本文将承担训练交互的仿真称为源环境，将策略接受运行评价的环境称为目标环境。二者可能在道路速度、客流起讫点分布、路口通行延误和必要停站服务时间等方面存在差异。本文关注这些差异造成的分布偏移，尤其是相同状态和动作下后继运行结果的分布变化，即动力学偏移。动力学随机化研究通过在仿真训练中改变模型参数改善迁移，说明训练分布的构造能够影响策略的现实适应能力。",
        " 本文区分的是能否预先获得包含状态、动作及其后续结果的目标运行记录，而不是是否掌握线路、时刻表等基础资料。缺少这类记录时，需事先判断哪些变化应纳入训练；获得固定目标离线数据后，则可依据已有目标经验约束学习，或修正源仿真的预测。",
    ),
    260: (
        "H2O及其扩展H2O+将预先采集的目标离线数据与源仿真中新产生的交互经验结合，提供了不同于纯离线学习和目标环境在线微调的数据利用方式。",
        " 本文将这一框架用于公交驻站控制，处理同时到站车辆的决策顺序，以混合动作分布区分不额外驻站和连续驻站时长，并同时评价乘客时间与行程完成情况。训练中新经验仍来自源仿真；若标定不充分，同一动作在源目标环境中的后果可能不同，因而还需利用目标记录修正预测。",
    ),
    261: (
        "本论文以“从仿真到现实的公交系统驻站控制方法研究”为题，按照目标数据的可获得程度和运行条件变化逐步展开。训练前未获得目标域转移训练数据时，先以RE-SAC研究平稳随机运行下的保守价值评价，再研究BAPR的条件识别及CS-BAPR的训练反馈与驻站响应；获得固定目标离线数据后，将H2O+适配于公交任务，结合目标记录与低成本源仿真训练；进一步针对源仿真标定不足，研究MC-WM如何比较源预测修正与直接目标建模，并利用所选模型辅助价值学习。各章以车头时距和乘客服务指标检验控制效果，形成与目标信息条件相适应的公交驻站学习方法。这一研究旨在减少训练对新增目标环境交互的依赖，并缓解源仿真误差对控制的影响，为真实运行数据辅助的公交策略训练提供方法基础。",
    ),
}


def main():
    assert not BACKUP.exists(), "Keep the pre-edit backup."
    assert not DOCUMENT.exists(), "Do not overwrite an existing second-pass manuscript."
    with ZipFile(SOURCE) as source:
        original = source.read("word/document.xml").decode("utf-8")
        body = ET.fromstring(original).find(W + "body")
        assert "".join(body[252].itertext()) == "1.1 研究背景与意义"
        assert "".join(body[262].itertext()) == "1.2 国内外研究综述"
        updated = original
        for index, prose in REVISIONS.items():
            updated = replace_paragraph(updated, body[index], prose)

        # Only prose may change; fields, equations and all layout XML stay intact.
        mask = lambda xml: re.sub(r"(<w:t(?:\s[^>]*)?>).*?(</w:t>)", r"\1\2", xml, flags=re.S)
        assert mask(original) == mask(updated), "Non-text XML changed."
        final_body = ET.fromstring(updated).find(W + "body")
        assert len(body) == len(final_body)
        changed = [i for i, (a, b) in enumerate(zip(body, final_body)) if ET.tostring(a) != ET.tostring(b)]
        assert changed == list(REVISIONS), changed
        protected_count = 0
        for i in changed:
            for tag in (M + "oMath", W + "hyperlink", W + "instrText", W + "fldChar", W + "pPr"):
                assert [ET.tostring(n) for n in body[i].iter(tag)] == [ET.tostring(n) for n in final_body[i].iter(tag)], (i, tag)
            nodes, _, protected = prose_groups(body[i])
            new_nodes = list(final_body[i].iter(W + "t"))
            assert all(ET.tostring(nodes[j]) == ET.tostring(new_nodes[j]) for j in protected), i
            protected_count += len(protected)
        copy2(SOURCE, BACKUP)
        with ZipFile(DOCUMENT, "w") as target:
            for item in source.infolist():
                target.writestr(item, updated.encode("utf-8") if item.filename == "word/document.xml" else source.read(item.filename))
    with ZipFile(BACKUP) as before, ZipFile(DOCUMENT) as after:
        assert before.namelist() == after.namelist()
        assert [n for n in before.namelist() if before.read(n) != after.read(n)] == ["word/document.xml"]
    print({"changed_blocks": changed, "protected_text_nodes": protected_count,
           "backup": str(BACKUP), "output": str(DOCUMENT)})


if __name__ == "__main__":
    main()
