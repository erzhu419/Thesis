from pathlib import Path
from shutil import copy2
from zipfile import ZipFile
import re
import xml.etree.ElementTree as ET

from edit_section_1_2_1 import M, W
from edit_section_3_2 import prose_groups
from edit_section_4_4 import replace_paragraph


ROOT = Path(__file__).resolve().parents[2]
DOCUMENT = ROOT / "output/排版完成_逐节校订_第2轮_20261010.docx"
BACKUP = Path(__file__).with_name("校订前_第2轮_1.3.docx")
REVISIONS = {
    336: (
        "本文以公交在途驻站控制为对象，研究目标环境信息有限、运行条件可能变化时，如何利用仿真训练获得有效策略。车辆完成必要上下客后，策略根据当前运行信息选择规定范围内的附加驻站时长。研究以既定线路、班次和车辆供给为前提，分别以改善车头时距稳定性和降低乘客广义时间成本为目标，以时距偏差、人均广义时间成本及行程完成情况评价控制效果。",
    ),
    337: (
        "研究按目标域转移训练数据的可获得性和运行规律是否变化逐步展开。未预先获得这些数据时，第3章在平稳随机环境下以RE-SAC改进SAC的动作评价；第4章转向非平稳环境，研究BAPR的条件识别与CS-BAPR的策略适应。获得固定目标离线数据后，第5章在H2O+基础上结合目标记录与快速仿真交互开展跨保真训练；第6章针对源仿真标定不足，以MC-WM修正预测并辅助策略学习。各章沿用第2章的任务与指标，依次说明问题、方法和实验结果。",
    ),
    339: (
        "（1）第3章研究平稳随机环境下的车队经验共享与鲁棒驻站控制。结合类别特征嵌入与连续运行观测，使全车队共享策略，同时区分车辆身份与服务位置。",
        "在SAC基础上提出RE-SAC",
        "，由多个价值网络共同评价动作，并以集成分歧和网络权重范数构造保守修正。通过与SAC比较，以及类别表征、奖励函数、集成规模和价值修正的对照实验，分别检验共享表征与鲁棒评价对时距控制和乘客服务的作用。",
    ),
    340: (
        "（2）第4章研究非平稳环境下的条件识别与驻站策略适应。运行规律变化后，历史经验可能不再代表当前条件，需要结合近期运行记录判断变化。在BAPR",
        "的模式识别与置信调节研究基础上，提出CS-BAPR",
        "，研究训练稳定化、驻站输出结构和软符号约束的作用。通过与BAPR、RE-SAC比较，评价固定需求放大、单次冲击和反复变化下的控制效果。公交实现以近期奖励和价值分歧调节训练，评价时执行固定策略；受控执行器实验另行检验在线模式识别与动作补偿。",
    ),
    341: (
        "（3）第5章研究固定目标离线数据辅助的跨保真公交驻站训练。基于已有H2O/H2O+框架，完成公交状态、驻站动作和训练流程的适配。",
        "先比较离线学习与追加目标交互的在线微调，再结合固定SUMO数据与快速仿真交互训练策略。编码同时到站车辆的状态及本轮依次选定的动作，以混合动作分布表示不额外驻站的概率和连续驻站时长，并结合驻站幅度调整与行程完成保障形成控制器。比较混合训练与纯离线训练，对照同一策略的不同执行规则，并以Daganzo式规则为参照，分别分析训练与执行对乘客服务的作用。",
    ),
    342: (
        "（4）第6章针对源仿真标定不足，提出MC-WM",
        "。用目标离线数据学习源预测的残差修正，并与直接目标模型比较；以未参与拟合的数据选择模型，按预测可信度调节模型样本对价值更新的贡献。公交模型预测下一批事件数量、线路与站点类别、连续运行状态、事件间隔和乘客成本。对置信度递推更新作归一化，缓解累计递推使模型样本权重过度减小的问题。通过不同标定偏差下与H2O+、无世界模型混合训练基线的比较，以及模型使用方式的对照，检验模型辅助学习对时距控制和乘客服务的作用。",
    ),
}


def main():
    assert not BACKUP.exists(), "Keep the pre-edit backup."
    with ZipFile(DOCUMENT) as source:
        original = source.read("word/document.xml").decode("utf-8")
        body = ET.fromstring(original).find(W + "body")
        assert "".join(body[335].itertext()) == "1.3 研究目标与内容"
        assert "".join(body[343].itertext()) == "1.4 技术路线"
        assert "".join(body[2169].itertext()).startswith("附录A")
        updated = original
        for index, prose in REVISIONS.items():
            updated = replace_paragraph(updated, body[index], prose)

        # Only prose changes; preserve equations, citations and layout XML.
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
        copy2(DOCUMENT, BACKUP)
        with ZipFile(DOCUMENT.with_suffix(".tmp"), "w") as target:
            for item in source.infolist():
                target.writestr(item, updated.encode("utf-8") if item.filename == "word/document.xml" else source.read(item.filename))
    DOCUMENT.with_suffix(".tmp").replace(DOCUMENT)
    with ZipFile(BACKUP) as before, ZipFile(DOCUMENT) as after:
        assert before.namelist() == after.namelist()
        assert [n for n in before.namelist() if before.read(n) != after.read(n)] == ["word/document.xml"]
    print({"changed_blocks": changed, "protected_text_nodes": protected_count,
           "backup": str(BACKUP), "output": str(DOCUMENT)})


if __name__ == "__main__":
    main()
