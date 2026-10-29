import re
import xml.etree.ElementTree as ET
from pathlib import Path
from shutil import copy2
from zipfile import ZipFile

from edit_section_1_2_1 import DOCUMENT, W
from edit_section_4_4 import replace_paragraph


BACKUP = Path(__file__).with_name("校订前_5.1.docx")
REVISIONS = {
    1224: (
        "第3、4章未预先获得目标域转移训练数据，分别研究平稳随机波动下的保守价值评价，以及运行规律变化后的识别与适应。本章进一步考虑能够获得目标离线数据的情况：已有记录包含目标环境中的状态、驻站动作及后续运行结果，但只覆盖采集策略实际经历的情形。如何利用这些目标经验，又减少在目标环境中反复试错的需要，是本章研究的问题。",
    ),
    1225: (
        "本章以多线路SUMO为高保真目标环境，以简易仿真为可反复交互的源环境，先后研究两项任务。第一项比较仅用固定SUMO离线数据学习与追加SUMO训练交互，考察已有数据能够支持何种控制，以及继续交互的作用。第二项保留固定SUMO数据，将新增训练交互放在源仿真中，基于H2O+框架进行公交适配，结合目标离线经验与源仿真新经验训练驻站策略。后一任务不追加SUMO训练转移，验证和测试轨迹不进入训练经验。",
    ),
    1226: (
        "源仿真与目标环境可能经常出现不同的状态和动作；在相同状态下执行相同驻站动作，道路速度、客流到达与停站服务的差异又可能改变后续运行结果。有限记录还会影响这些差异的估计精度。本章据此区分状态动作分布、条件转移和有限采样误差，并分别评价训练方案与执行规则的服务作用。两项任务采用不同的数据集、状态输入、奖励和运营时域，结果分别在相应任务内比较。源仿真标定不足时的预测修正，则由第6章进一步研究。",
    ),
}


def main():
    assert not BACKUP.exists(), "Keep the existing backup; do not rerun the initial edit."
    with ZipFile(DOCUMENT) as source:
        original = source.read("word/document.xml").decode("utf-8")
        body = ET.fromstring(original).find(W + "body")
        assert "".join(body[1223].itertext()) == "5.1 问题描述"
        assert "".join(body[1227].itertext()) == "5.2 模型构建"
        updated = original
        for index, prose in REVISIONS.items():
            updated = replace_paragraph(updated, body[index], prose)

        mask = lambda xml: re.sub(r"(<w:t(?:\s[^>]*)?>).*?(</w:t>)", r"\1\2", xml, flags=re.S)
        assert mask(original) == mask(updated), "Non-text XML changed."
        final_body = ET.fromstring(updated).find(W + "body")
        assert len(body) == len(final_body)
        changed = [i for i, (a, b) in enumerate(zip(body, final_body)) if ET.tostring(a) != ET.tostring(b)]
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
        assert [n for n in before.namelist() if before.read(n) != after.read(n)] == ["word/document.xml"]
    print({"changed_body_blocks": changed, "backup": str(BACKUP), "non_text_xml_preserved": True})


if __name__ == "__main__":
    main()
