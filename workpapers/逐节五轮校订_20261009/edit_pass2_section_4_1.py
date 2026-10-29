from pathlib import Path
from shutil import copy2
from zipfile import ZipFile
import re
import xml.etree.ElementTree as ET

from edit_section_1_2_1 import W
from edit_section_4_4 import replace_paragraph


ROOT = Path(__file__).resolve().parents[2]
DOCUMENT = ROOT / "output/排版完成_逐节校订_第2轮_20261010.docx"
BACKUP = Path(__file__).with_name("校订前_第2轮_4.1.docx")
REVISIONS = {
    990: (
        "第3章研究给定运行规律下的随机波动，RE-SAC以保守价值评价改进驻站决策。本章进一步考虑客流到达率、OD构成及路段速度分布发生持续、突发或反复变化的情形。与规律已知的日常峰谷不同，大型活动或交通管制可能使同一时段的客流和路况偏离既有规律。即时观测相近时，相同驻站动作也可能产生不同后果，历史经验是否仍适用便成为控制的新问题。",
    ),
    991: (
        "某次车头时距偏差既可能来自正常随机波动，也可能反映运行规律改变，不能仅凭一次异常作出判断。需要结合近期运行反馈，判断变化是否持续，并据此调节学习。规律改变后，随机波动和第3章讨论的认知不确定性仍然存在。因此，本章在保守价值评价的基础上，进一步研究变化的识别与适应。",
    ),
    992: (
        "第3章与本章均未预先获得目标域转移训练数据，策略学习与方法检验主要依靠可交互仿真。本章利用训练期间的仿真历史判断变化；第5、6章则在获得固定目标离线数据后，研究如何改善源仿真训练。",
    ),
    993: (
        "控制范围仍是给定线路、发车计划和车辆供给下的单智能体车队驻站决策，由共享策略输出有界附加驻站时长。本章以BAPR根据历史证据更新模式判断的原理为基础，提出CS-BAPR，研究训练稳定化、算术输出及符号参考约束对驻站控制的作用。公交实验在训练期利用近期奖励和价值分歧调节保守程度，评价时执行固定策略；受控执行器实验另行检验在线模式识别与动作补偿。后文依次构建模型、介绍求解方法，并比较BAPR、CS-BAPR与RE-SAC在需求持续放大、单次及反复冲击下的结果，分别以车头时距和人均广义时间成本评价控制效果。",
    ),
}


def main():
    assert not BACKUP.exists(), "Keep the existing pre-edit backup."
    with ZipFile(DOCUMENT) as source:
        original = source.read("word/document.xml").decode("utf-8")
        body = ET.fromstring(original).find(W + "body")
        assert "".join(body[989].itertext()) == "4.1 问题描述"
        assert "".join(body[994].itertext()) == "4.2 模型构建"
        updated = original
        for index, prose in REVISIONS.items():
            updated = replace_paragraph(updated, body[index], prose)
        mask = lambda xml: re.sub(r"(<w:t(?:\s[^>]*)?>).*?(</w:t>)", r"\1\2", xml, flags=re.S)
        assert mask(original) == mask(updated), "Non-text XML changed."
        final = ET.fromstring(updated).find(W + "body")
        assert len(body) == len(final) == 2924
        changed = [i for i, (a, b) in enumerate(zip(body, final)) if ET.tostring(a) != ET.tostring(b)]
        assert changed == list(REVISIONS), changed
        for index in changed:
            assert re.findall(r"\d+(?:\.\d+)?", "".join(body[index].itertext())) == re.findall(r"\d+(?:\.\d+)?", "".join(final[index].itertext()))
        copy2(DOCUMENT, BACKUP)
        temporary = DOCUMENT.with_suffix(".tmp")
        with ZipFile(temporary, "w") as target:
            for item in source.infolist():
                target.writestr(item, updated.encode("utf-8") if item.filename == "word/document.xml" else source.read(item.filename))
    temporary.replace(DOCUMENT)
    with ZipFile(BACKUP) as before, ZipFile(DOCUMENT) as after:
        assert before.namelist() == after.namelist()
        assert [n for n in before.namelist() if before.read(n) != after.read(n)] == ["word/document.xml"]
    print({"changed_blocks": changed, "backup": str(BACKUP), "output": str(DOCUMENT)})


if __name__ == "__main__":
    main()
