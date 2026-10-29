from pathlib import Path
from shutil import copy2
from zipfile import ZipFile
import re
import xml.etree.ElementTree as ET

from edit_section_1_2_1 import DOCUMENT, W, W14, revise_paragraph


BACKUP = Path(__file__).with_name("校订前_1.5.docx")
REVISIONS = {
    352: "全文共7章。第1章梳理公交控制、强化学习及仿真到现实研究，明确研究问题，按目标训练数据的可用条件和利用方式提出研究内容与技术路线。",
    353: "第2章建立车辆到站事件下的驻站决策模型，明确状态、动作、奖励及评价指标，介绍强化学习、不确定性处理和迁移学习基础，并说明仿真平台的基础数据、构建与标定、控制响应检验及实验设置。",
    354: "第3章研究平稳随机环境下的鲁棒驻站控制，以类别嵌入支持共享车队策略，通过RE-SAC改进动作价值评价。利用与SAC的比较及组件消融，分析对车头时距和乘客时间的影响。",
    355: "第4章将研究扩展至运行规律随时间变化的非平稳环境，介绍BAPR的环境模式识别及CS-BAPR的训练调节与结构化策略。通过客流变化实验和组件对照，考察相对RE-SAC的控制效果。",
    356: "第5章在固定目标离线数据可用的条件下，将H2O+适配于公交驻站控制，结合目标经验与源仿真交互开展跨保真训练。分别比较训练方法和完成保障的作用，检验迁移策略的服务效果。",
    357: "第6章针对源仿真标定不足造成的动力学差异，研究MC-WM的世界模型选择、预测修正与模型样本加权。通过受控动力学实验分析作用机制，再通过公交实验检验不同标定偏差下的控制效果。",
    358: "第7章总结主要结论与贡献，讨论现有研究的局限及实际运营应用方向。数学附录为正文中的算子性质、动作补偿和误差分析提供证明。",
}


def mask_prose(raw):
    return re.sub(r"(<w:t(?:\s[^>]*)?>).*?(</w:t>)", r"\1\2", raw, flags=re.S)


def main():
    if BACKUP.exists():
        raise FileExistsError(BACKUP)
    with ZipFile(DOCUMENT) as source:
        original = source.read("word/document.xml").decode("utf-8")
        body = ET.fromstring(original).find(W + "body")
        assert "".join(n.text or "" for n in body[351].iter(W + "t")) == "1.5 论文结构"
        updated = original
        for index, prose in REVISIONS.items():
            paragraph_id = body[index].get(W14 + "paraId")
            pattern = r'<w:p\b[^>]*w14:paraId="' + paragraph_id + r'"[^>]*>.*?</w:p>'
            matches = list(re.finditer(pattern, updated, flags=re.S))
            assert len(matches) == 1, (index, paragraph_id)
            match = matches[0]
            revised = revise_paragraph(match.group(0), body[index], (prose,))
            assert mask_prose(match.group(0)) == mask_prose(revised)
            updated = updated[:match.start()] + revised + updated[match.end():]
        assert mask_prose(original) == mask_prose(updated)
        new_body = ET.fromstring(updated).find(W + "body")
        assert len(body) == len(new_body)
        changed = [i for i, (a, b) in enumerate(zip(body, new_body)) if ET.tostring(a) != ET.tostring(b)]
        assert changed == list(REVISIONS), changed
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
    print("Changed body blocks:", changed)
    print("Only seven prose paragraphs in section 1.5 changed; all other content and formatting XML preserved.")
    for index, prose in REVISIONS.items():
        print(index, prose)
    print("OUTPUT", DOCUMENT)


if __name__ == "__main__":
    main()
