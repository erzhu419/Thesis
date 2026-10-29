import re
import xml.etree.ElementTree as ET
from pathlib import Path
from shutil import copy2
from zipfile import ZipFile

from edit_section_1_2_1 import DOCUMENT, W
from edit_section_4_4 import replace_paragraph


BACKUP = Path(__file__).with_name("校订前_4.5.docx")
REVISIONS = {
    1216: (
        "本章针对非平稳公交运行，建立隐含模式与条件响应模型，研究由BAPR到CS-BAPR的识别与适应方法。公交实现利用近期训练反馈调节保守程度，固定训练后的策略评价需求变化下的控制表现；执行器实验则检验模式识别后的动作补偿。CS-BAPR进一步结合训练稳定化与驻站输出约束。在单一驻站、乘客时间目标下，后续软符号配置相对RE-SAC在六类客流场景中均降低平均人均广义时间，降幅为0.083%至0.549%；计划时距偏差和前后时距不均衡度分别降低7.35至12.09秒和11.88至20.08秒，支持其在所测非平稳条件下的平均服务收益。",
    ),
    1217: (
        "早期稳定化感知机相对前序BAPR，在六种需求条件中使平均时距奖励惩罚下降58.82%至69.79%，计划绝对偏差下降27.10%至36.55%；改用乘客时间目标后，广义时间却增加0.068%至0.268%。后续软符号配置以时距为目标训练时，五类场景的平均奖励惩罚降低，但六类场景的计划时距偏差与时距不均衡度均增加。算术输出也未在所有条件下带来额外收益。不同配置、参照和训练目标因而需要分别解释，完整配置的改善不能全部归于符号约束。理论分析给出相应假设下策略响应与评价算子的性质，运营效果由实验判断。有限驻站能够调节车辆间隔，但不能替代供给调整。第3、4章未预先使用目标域转移训练数据；第5章进一步利用固定目标离线数据，研究跨保真驻站学习。",
    ),
}


def main():
    assert not BACKUP.exists(), "Keep the existing backup; do not rerun the initial edit."
    with ZipFile(DOCUMENT) as source:
        original = source.read("word/document.xml").decode("utf-8")
        body = ET.fromstring(original).find(W + "body")
        assert "".join(body[1215].itertext()) == "4.5 本章小结"
        assert "".join(body[1219].itertext()).startswith("第5章 ")
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
