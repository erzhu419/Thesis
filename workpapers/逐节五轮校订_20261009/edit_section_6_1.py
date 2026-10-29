import re
import xml.etree.ElementTree as ET
from pathlib import Path
from shutil import copy2
from zipfile import ZipFile

from edit_section_1_2_1 import DOCUMENT, W
from edit_section_4_4 import replace_paragraph


BACKUP = Path(__file__).with_name("校订前_6.1.docx")
REVISIONS = {
    1552: (
        "第5章利用目标离线数据和源仿真交互训练公交驻站策略，训练效果仍受源仿真标定的影响。道路速度、客流和停站服务标定不足时，源仿真对后续到达时间、乘客服务和前后时距的预测可能有系统偏差，使策略高估某些驻站动作的收益。本章利用目标记录学习世界模型，以改进预测和驻站训练。",
    ),
    1553: (
        "公交任务沿用固定目标离线数据与持续源仿真交互的条件，不追加SUMO训练交互。模型可先学习源仿真中的转移规律，再利用目标记录修正源预测；也可直接利用目标记录建立目标模型。目标样本覆盖和拟合预算有限，需要依据未参与拟合的目标记录比较预测误差，选择合适的模型。",
    ),
    1554: (
        "所选世界模型根据状态和动作预测后继状态与奖励，生成一步训练样本。这些样本参与价值更新时，也会将预测误差带入动作评价，因此还需确定它们的训练权重。权重过大会放大错误预测的影响，过小则难以发挥模型作用。最终仍需检验预测改进能否带来驻站控制收益。",
    ),
    1555: (
        "实验分别检验模型预测和控制效果。受控连续任务允许追加目标环境交互，用于分析残差修正、直接建模及模型选择的作用；公交任务保持目标离线数据固定，比较不同源仿真标定偏差下的事件预测、驻站控制结果和计算代价。",
    ),
}


def main():
    assert not BACKUP.exists(), "Keep the existing pre-edit backup."
    with ZipFile(DOCUMENT) as source:
        original = source.read("word/document.xml").decode("utf-8")
        body = ET.fromstring(original).find(W + "body")
        assert "".join(body[1551].itertext()) == "6.1 问题描述"
        assert "".join(body[1556].itertext()) == "6.2 模型构建"
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
