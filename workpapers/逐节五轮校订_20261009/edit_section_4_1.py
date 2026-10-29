from pathlib import Path
from shutil import copy2
from zipfile import ZipFile
import re
import xml.etree.ElementTree as ET

from edit_section_1_2_1 import DOCUMENT, W, W14, revise_paragraph


BACKUP = Path(__file__).with_name("校订前_4.1.docx")
REVISIONS = {
    990: (
        "第3章研究给定运行规律下的随机波动，RE-SAC以保守价值评价改进驻站决策。本章进一步考虑客流到达率、OD构成及路段速度分布随时间变化的情形。大型活动散场可能使客流骤增，交通管制可能改变行驶时间，同类变化也可能持续或反复发生。此时，即使车辆的即时状态相近，驻站后的运行结果仍可能不同，需要判断历史经验是否适用于当前条件。",
    ),
    991: (
        "有限经验造成的认知不确定性在第3章已经存在；本章增加的难点是当前运行规律可能不同于历史规律。仅凭某次车头时距偏差，难以区分随机波动与持续变化，需要结合近期历史判断。新的运行条件下仍有随机客流和行驶时间波动。因此，控制既要应对当前条件下的随机波动，也要适应运行规律的变化。",
    ),
    992: (
        "第3章与本章均未预先获得目标域转移训练数据，策略学习与方法检验主要依靠可交互仿真。本章利用已有仿真交互历史判断变化、调节学习；第5、6章则在获得固定目标离线数据后，研究如何改善源仿真训练。",
    ),
    993: (
        "控制范围仍是给定线路、发车计划和车辆供给下的单智能体车队驻站决策，由共享策略输出有界附加驻站时长。本章以BAPR根据历史证据更新模式判断的原理为基础，提出CS-BAPR，将训练稳定化与算术关系学习结合。公交实验在训练期利用近期信息调节保守程度，评价时执行固定策略；受控执行器实验另行检验在线模式识别与动作补偿。后文依次构建模型、介绍求解方法，并比较BAPR、CS-BAPR与RE-SAC在需求持续放大、单次及反复冲击下的结果，分别以车头时距和人均广义时间成本评价控制效果。",
    ),
}


def main():
    with ZipFile(DOCUMENT) as source:
        original = source.read("word/document.xml").decode("utf-8")
        body = ET.fromstring(original).find(W + "body")
        assert "".join(body[989].itertext()) == "4.1 问题描述"
        assert "".join(body[994].itertext()) == "4.2 模型构建"
        updated = original
        for index, prose in REVISIONS.items():
            paragraph_id = body[index].get(W14 + "paraId")
            pattern = r'<w:p\b[^>]*w14:paraId="' + paragraph_id + r'"[^>]*>.*?</w:p>'
            matches = list(re.finditer(pattern, updated, flags=re.S))
            assert len(matches) == 1
            match = matches[0]
            revised = revise_paragraph(match.group(0), body[index], prose)
            updated = updated[:match.start()] + revised + updated[match.end():]
        mask = lambda raw: re.sub(r"(<w:t(?:\s[^>]*)?>).*?(</w:t>)", r"\1\2", raw, flags=re.S)
        assert mask(original) == mask(updated), "Non-text XML changed."
        new_body = ET.fromstring(updated).find(W + "body")
        assert len(body) == len(new_body)
        changed = [i for i, (a, b) in enumerate(zip(body, new_body)) if ET.tostring(a) != ET.tostring(b)]
        assert changed and set(changed).issubset(REVISIONS), changed
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
        changed_parts = [name for name in before.namelist() if before.read(name) != after.read(name)]
        assert changed_parts == ["word/document.xml"], changed_parts
    print({"changed_blocks": changed, "changed_parts": changed_parts, "backup": str(BACKUP)})


if __name__ == "__main__":
    main()
