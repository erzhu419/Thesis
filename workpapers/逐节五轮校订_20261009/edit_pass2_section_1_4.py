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
BACKUP = Path(__file__).with_name("校订前_第2轮_1.4.docx")
REVISIONS = {
    344: (
        "技术路线分为决策建模、仿真策略学习和目标数据辅助改进三个环节。首先，在既定线路、班次与车辆供给下，将车辆运行、上下客服务和驻站决策组织为事件决策过程，明确动作与后续状态、奖励的对应关系。分别以车头时距稳定性和乘客广义时间成本为目标，以运营指标比较控制效果。策略学习需区分随机运行后果与有限经验造成的估计不足。",
    ),
    345: (
        "尚无预先采集的目标域转移训练数据时，先通过仿真生成运行经验。第3章针对平稳随机环境，以类别特征组织车队共享学习，用RE-SAC改进SAC的动作价值评价。域随机化、鲁棒评价和尾部风险分析分别为多条件训练、最不利后果及较差结果的评价提供依据。",
        "第4章转向运行规律变化的非平稳环境，在BAPR的条件识别研究基础上，以CS-BAPR进一步研究训练调节、结构化驻站输出和符号参考约束。公交实验利用近期反馈调节训练，评价时保持策略参数固定；执行器实验则检验运行中的模式识别与动作补偿。",
    ),
    346: (
        "获得固定目标离线数据后，第5章基于H2O+完成公交适配，将目标记录与源仿真新增经验共同用于策略更新，并分别检验学习策略与行程完成保障的作用。第6章进一步针对源仿真标定不足，以MC-WM拟合源模型的预测残差及直接目标模型，再用未参与拟合的数据选择模型，按预测可信度加权模型样本。对置信度递推更新作归一化，缓解累计递推使模型样本权重过度减小的问题。两章公交混合训练均保持目标数据固定，新增交互在源仿真中完成。",
    ),
    347: (
        "理论分析分别考察价值修正、模式误判与动作补偿、策略响应约束以及模型误差对价值估计的影响，仿真实验检验相应控制效果。表1-2汇总各章的运行条件、可用数据与方法重点。",
    ),
    350: (
        "给出上述技术路线。公交实验分别比较两种奖励下的控制效果，用时距指标、乘客时间及行程完成情况联系训练目标与服务结果。跨保真试验进一步比较源仿真标定偏差和执行规则的影响。受控连续任务用于补充分析模式识别、动作补偿及模型学习机制，其交互条件与公交试验分别说明。",
    ),
}
TABLE_REVISIONS = {
    (1, 1): "给定随机规律；无预先目标域转移训练数据",
    (1, 2): "车队经验共享与保守价值评价",
    (2, 1): "运行规律变化；利用近期运行反馈",
    (2, 2): "条件识别、训练调节与符号参考约束",
    (3, 2): "公交事件编码与源、目标经验混合",
    (4, 2): "残差与直接模型选择、模型样本加权",
    (4, 3): "MC-WM与置信度递推更新",
}


def main():
    assert not BACKUP.exists(), "Keep the pre-edit backup."
    with ZipFile(DOCUMENT) as source:
        original = source.read("word/document.xml").decode("utf-8")
        body = ET.fromstring(original).find(W + "body")
        assert "".join(body[343].itertext()) == "1.4 技术路线"
        assert "".join(body[351].itertext()) == "1.5 论文结构"
        assert "".join(body[2169].itertext()).startswith("附录A")
        edits = [(body[i], prose) for i, prose in REVISIONS.items()]
        rows = body[349].findall(W + "tr")
        for (row, column), text in TABLE_REVISIONS.items():
            paragraphs = rows[row].findall(W + "tc")[column].findall(W + "p")
            assert len(paragraphs) == 1
            edits.append((paragraphs[0], (text,)))
        updated = original
        for paragraph, prose in edits:
            updated = replace_paragraph(updated, paragraph, prose)

        # Preserve native objects and formatting; change only this section's prose.
        mask = lambda xml: re.sub(r"(<w:t(?:\s[^>]*)?>).*?(</w:t>)", r"\1\2", xml, flags=re.S)
        assert mask(original) == mask(updated), "Non-text XML changed."
        final_body = ET.fromstring(updated).find(W + "body")
        assert len(body) == len(final_body)
        changed = [i for i, (a, b) in enumerate(zip(body, final_body)) if ET.tostring(a) != ET.tostring(b)]
        assert changed == [344, 345, 346, 347, 349, 350], changed
        protected_count = 0
        for i in changed:
            for tag in (M + "oMath", W + "hyperlink", W + "instrText", W + "fldChar", W + "pPr"):
                assert [ET.tostring(n) for n in body[i].iter(tag)] == [ET.tostring(n) for n in final_body[i].iter(tag)], (i, tag)
            initial_paragraphs = list(body[i].iter(W + "p"))
            final_paragraphs = list(final_body[i].iter(W + "p"))
            assert len(initial_paragraphs) == len(final_paragraphs)
            for before, after in zip(initial_paragraphs, final_paragraphs):
                nodes, _, protected = prose_groups(before)
                new_nodes = list(after.iter(W + "t"))
                assert all(ET.tostring(nodes[j]) == ET.tostring(new_nodes[j]) for j in protected), i
                protected_count += len(protected)
        copy2(DOCUMENT, BACKUP)
        temporary = DOCUMENT.with_suffix(".tmp")
        with ZipFile(temporary, "w") as target:
            for item in source.infolist():
                target.writestr(item, updated.encode("utf-8") if item.filename == "word/document.xml" else source.read(item.filename))
    temporary.replace(DOCUMENT)
    with ZipFile(BACKUP) as before, ZipFile(DOCUMENT) as after:
        assert before.namelist() == after.namelist()
        assert [n for n in before.namelist() if before.read(n) != after.read(n)] == ["word/document.xml"]
    print({"changed_blocks": changed, "paragraph_count": len(REVISIONS),
           "table_cell_count": len(TABLE_REVISIONS), "protected_text_nodes": protected_count,
           "backup": str(BACKUP), "output": str(DOCUMENT)})


if __name__ == "__main__":
    main()
