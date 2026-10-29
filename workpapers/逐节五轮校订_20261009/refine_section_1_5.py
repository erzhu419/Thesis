from zipfile import ZipFile
import re
import xml.etree.ElementTree as ET

from edit_section_1_2_1 import DOCUMENT, W, W14, revise_paragraph
from edit_section_1_5 import BACKUP, mask_prose


REVISIONS = {
    352: "全文共7章。第1章梳理公交控制、强化学习及仿真到现实研究，明确研究问题、研究内容与技术路线。",
    353: "第2章建立事件驱动的驻站决策模型，明确状态、动作、奖励和评价指标，介绍强化学习、不确定性与迁移基础，以及仿真平台、标定检验和实验设置。",
    354: "第3章研究平稳随机环境下的鲁棒控制，以类别嵌入支持共享车队策略，以RE-SAC改进价值评价，并用SAC对照和组件消融分析时距与乘客时间表现。",
    355: "第4章研究非平稳运行，介绍BAPR的模式识别及CS-BAPR的训练调节与结构化输出，以客流变化实验和组件对照检验相对RE-SAC的控制效果。",
    356: "第5章在固定目标离线数据可用时，将H2O+适配于公交驻站，结合目标经验与源仿真交互训练，分别检验迁移学习和完成保障的服务效果。",
    357: "第6章针对源仿真标定不足，研究MC-WM的模型选择、预测修正与样本加权，以受控动力学实验分析机制，以公交实验检验标定偏差下的控制效果。",
    358: "第7章总结主要结论、贡献、局限及实际运营应用方向。数学附录为正文中的算子性质、动作补偿和误差分析提供证明。",
}


def main():
    with ZipFile(DOCUMENT) as source:
        original = source.read("word/document.xml").decode("utf-8")
        body = ET.fromstring(original).find(W + "body")
        updated = original
        for index, prose in REVISIONS.items():
            paragraph_id = body[index].get(W14 + "paraId")
            pattern = r'<w:p\b[^>]*w14:paraId="' + paragraph_id + r'"[^>]*>.*?</w:p>'
            matches = list(re.finditer(pattern, updated, flags=re.S))
            assert len(matches) == 1
            match = matches[0]
            revised = revise_paragraph(match.group(0), body[index], (prose,))
            updated = updated[:match.start()] + revised + updated[match.end():]
        assert mask_prose(original) == mask_prose(updated)
        temporary = DOCUMENT.with_suffix(".section-edit.docx")
        with ZipFile(temporary, "w") as target:
            for item in source.infolist():
                data = updated.encode("utf-8") if item.filename == "word/document.xml" else source.read(item.filename)
                target.writestr(item, data)
    temporary.replace(DOCUMENT)
    with ZipFile(BACKUP) as before, ZipFile(DOCUMENT) as after:
        assert before.namelist() == after.namelist()
        assert [name for name in before.namelist() if before.read(name) != after.read(name)] == ["word/document.xml"]
        raw_before = before.read("word/document.xml").decode("utf-8")
        raw_after = after.read("word/document.xml").decode("utf-8")
        assert mask_prose(raw_before) == mask_prose(raw_after)
        before_body = ET.fromstring(raw_before).find(W + "body")
        after_body = ET.fromstring(raw_after).find(W + "body")
        assert len(before_body) == len(after_body)
        changed = [i for i, (a, b) in enumerate(zip(before_body, after_body)) if ET.tostring(a) != ET.tostring(b)]
        assert changed == list(REVISIONS), changed
    print("Changed body blocks relative to section backup:", changed)
    for index, prose in REVISIONS.items():
        print(index, prose)


if __name__ == "__main__":
    main()
