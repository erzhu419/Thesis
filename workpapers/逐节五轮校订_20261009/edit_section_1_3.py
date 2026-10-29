from pathlib import Path
from shutil import copy2
from zipfile import ZipFile
import re
import xml.etree.ElementTree as ET

from edit_section_1_2_1 import DOCUMENT, W, W14, revise_paragraph


BACKUP = Path(__file__).with_name("校订前_1.3.docx")
REVISIONS = {
    336: (
        "本文以公交在途驻站控制为对象，研究在目标环境信息有限、运行条件可能变化时，如何利用仿真训练获得有效的控制策略。车辆完成必要上下客后，策略根据当前运行信息选择规定范围内的附加驻站时长。研究以既定线路、班次和车辆供给为前提，分别以均衡车头时距和降低乘客广义时间成本为优化目标，并结合候车时间、车内时间、运行间隔及行程完成情况评价控制效果。",
    ),
    337: (
        "研究按照目标域训练数据的可获得性及环境变化逐步展开。未预先获得目标域训练数据时，第3章在随机统计规律保持平稳的条件下，以RE-SAC改进SAC的动作评价；第4章进一步考虑这些规律随时间变化，研究BAPR的条件识别和CS-BAPR的策略适应。获得固定目标域离线数据后，第5章在H2O+基础上结合目标数据与快速仿真交互开展跨保真训练；第6章进一步针对快速仿真标定不足，利用MC-WM修正运行预测并辅助策略学习。各章均在第2章的控制任务与评价指标基础上，依次说明问题、方法和实验结果。",
    ),
    338: ("围绕上述目标，本文开展以下四方面研究。",),
    344: (
        "（1）第3章研究平稳随机环境下的车队经验共享与鲁棒驻站控制。将车辆、站点类别的特征嵌入与连续运行状态结合，使不同车辆共享策略，同时保留各自的服务差异。",
        "在SAC基础上提出RE-SAC",
        "，利用多个价值网络共同评价动作，对缺少一致支持的高价值估计作保守调整。通过与SAC比较，并对类别表征、奖励函数、集成规模和动作价值修正方式开展对照实验，分析经验共享与鲁棒评价对车头时距和乘客时间的作用。",
    ),
    345: (
        "（2）第4章研究非平稳环境下的条件识别与驻站策略适应。客流和路况的统计规律变化后，历史经验可能不再代表当前运行条件，因而利用近期运行记录描述条件变化。在BAPR",
        "的模式识别与置信调节研究基础上，提出CS-BAPR",
        "，研究软符号约束、策略输出结构和训练稳定化的作用。通过与BAPR、RE-SAC的对照，评价固定需求放大、单次冲击和重复变化条件下的控制效果。公交实验考察训练阶段调节后固定策略的表现，受控执行器实验进一步分析模式识别与动作补偿的关系。",
    ),
    346: (
        "（3）第5章研究固定目标域离线数据辅助的跨保真公交驻站训练。基于已有H2O/H2O+框架，完成公交事件状态、驻站动作与训练流程的适配。",
        "先比较离线学习与新增目标环境交互的在线微调，再结合固定SUMO离线数据与快速仿真在线交互训练策略。对于同时到站的车辆，编码已执行动作及当前决策状态；将动作分为不额外驻站和选择额外驻站时长两部分，结合驻站幅度调整与行程完成保障形成控制器。分别比较混合训练与纯离线训练、同一策略采用不同执行规则的效果，并与Daganzo式驻站规则比较，分析训练方法和执行规则对乘客服务的作用。",
    ),
    347: (
        "（4）第6章研究源仿真标定不足时的世界模型修正与策略学习，提出MC-WM",
        "。利用目标域离线数据学习源预测的残差修正，并与直接学习目标预测的模型比较；用未参与拟合的数据选择模型，按预测可信度控制模型样本参与价值更新的权重。公交模型预测下一事件的车辆与站点类别、连续运行状态、事件间隔和乘客成本，并对置信度递推更新作归一化，防止模型样本权重在递推中迅速减小。通过不同标定偏差下与H2O+、无世界模型混合训练基线的比较，检验预测修正对车头时距和乘客服务的作用。",
    ),
}
REMOVED = range(339, 344)


def paragraph_pattern(paragraph):
    return r'<w:p\b[^>]*w14:paraId="' + paragraph.get(W14 + "paraId") + r'"[^>]*>.*?</w:p>'


def main():
    with ZipFile(DOCUMENT) as source:
        original = source.read("word/document.xml").decode("utf-8")
        body = ET.fromstring(original).find(W + "body")
        assert "".join(body[335].itertext()) == "1.3 研究目标与内容"
        assert "".join(body[348].itertext()) == "1.4 技术路线"
        updated = original
        for index, prose in REVISIONS.items():
            matches = list(re.finditer(paragraph_pattern(body[index]), updated, flags=re.S))
            assert len(matches) == 1
            match = matches[0]
            revised = revise_paragraph(match.group(), body[index], prose)
            mask = lambda raw: re.sub(r"(<w:t(?:\s[^>]*)?>).*?(</w:t>)", r"\1\2", raw, flags=re.S)
            assert mask(match.group()) == mask(revised)
            updated = updated[:match.start()] + revised + updated[match.end():]

        # Merge the repeated questions into the four research items; retain bookmarks.
        anchors = []
        for index in REMOVED:
            paragraph = body[index]
            assert all(child.tag in (W + "pPr", W + "r", W + "bookmarkStart", W + "bookmarkEnd") for child in paragraph)
            assert all(child.tag in (W + "rPr", W + "t") for run in paragraph.findall(W + "r") for child in run)
            match = re.search(paragraph_pattern(paragraph), updated, flags=re.S)
            anchors.extend(re.findall(r'<w:bookmark(?:Start|End)\b[^>]*/>', match.group()))
            updated = updated[:match.start()] + updated[match.end():]
        lead = re.search(paragraph_pattern(body[338]), updated, flags=re.S)
        assert lead and anchors
        revised_lead = lead.group().replace("</w:pPr>", "</w:pPr>" + "".join(anchors), 1)
        updated = updated[:lead.start()] + revised_lead + updated[lead.end():]

        new_body = ET.fromstring(updated).find(W + "body")
        kept = [i for i in range(len(body)) if i not in REMOVED]
        assert len(new_body) == len(kept)
        changed = [i for i, after in zip(kept, new_body) if ET.tostring(body[i]) != ET.tostring(after)]
        assert changed == list(REVISIONS), changed
        bookmark_xml = lambda value: re.findall(r'<w:bookmark(?:Start|End)\b[^>]*/>', value)
        assert bookmark_xml(original) == bookmark_xml(updated)
        if not BACKUP.exists():
            copy2(DOCUMENT, BACKUP)
        temporary = DOCUMENT.with_suffix(".section-edit.docx")
        with ZipFile(temporary, "w") as target:
            for item in source.infolist():
                data = updated.encode("utf-8") if item.filename == "word/document.xml" else source.read(item.filename)
                target.writestr(item, data)
    temporary.replace(DOCUMENT)
    with ZipFile(BACKUP) as before, ZipFile(DOCUMENT) as after:
        assert before.namelist() == after.namelist()
        assert [name for name in before.namelist() if before.read(name) != after.read(name)] == ["word/document.xml"]
    print("Revised original blocks:", changed, "merged duplicate blocks:", list(REMOVED))
    print("Both abstracts and every other section unchanged; citations, equations, bookmarks and document objects preserved.")
    for index, paragraph in zip(kept, new_body):
        if index in REVISIONS:
            print(index, "".join(paragraph.itertext()))
    print("OUTPUT", DOCUMENT)


if __name__ == "__main__":
    main()
