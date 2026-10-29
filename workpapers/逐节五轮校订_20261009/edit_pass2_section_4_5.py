from pathlib import Path
from shutil import copy2
from zipfile import ZipFile
import re
import xml.etree.ElementTree as ET

from edit_section_3_2 import M, W, prose_groups
from edit_section_4_4 import replace_paragraph


ROOT = Path(__file__).resolve().parents[2]
DOCUMENT = ROOT / "output/排版完成_逐节校订_第2轮_20261010.docx"
BACKUP = Path(__file__).with_name("校订前_第2轮_4.5.docx")
REVISIONS = {
    1216: (
        "本章针对公交运行规律变化，研究BAPR的条件识别及CS-BAPR的训练稳定化与驻站输出约束。公交训练利用近期反馈调节保守程度，评价时固定策略；受控执行器实验另行检验在线模式识别与动作补偿。采用单一驻站动作、以乘客时间为目标时，后续软符号配置相对RE-SAC，在六类客流场景中均降低了人均广义时间成本的均值，降幅为0.083%至0.549%；计划时距偏差和前后时距不均衡度分别减少7.35至12.09秒、11.88至20.08秒，支持该配置在所测非平稳条件下的平均服务改善。",
    ),
    1217: (
        "早期稳定化MLP相对前序BAPR，在六种需求条件下使平均时距奖励惩罚下降58.82%至69.79%、计划绝对偏差下降27.10%至36.55%；改用乘客时间目标后，人均广义时间成本却增加0.068%至0.268%。后续软符号配置以时距为目标时，五类场景的平均奖励惩罚降低，六类场景的计划时距偏差和时距不均衡度却均增加。算术输出也并非在所有条件下都有额外收益，完整配置的改善不能全部归于符号约束。理论分析说明给定条件下的策略响应与价值更新的性质，实验则检验相应控制效果。有限驻站能够调节车辆间隔，不能替代供给调整。第3、4章未预先使用目标域转移训练数据；第5章进一步引入固定目标离线数据，研究如何结合目标经验与源仿真交互训练驻站策略。",
    ),
}


def main():
    assert not BACKUP.exists(), "Keep the existing pre-edit backup."
    with ZipFile(DOCUMENT) as source:
        original = source.read("word/document.xml").decode("utf-8")
        body = ET.fromstring(original).find(W + "body")
        assert "".join(body[1215].itertext()) == "4.5 本章小结"
        assert "第5章" in "".join(body[1219].itertext())
        updated = original
        for index, prose in REVISIONS.items():
            updated = replace_paragraph(updated, body[index], prose)

        # Any non-text difference would alter protected objects or formatting.
        mask = lambda xml: re.sub(r"(<w:t(?:\s[^>]*)?>).*?(</w:t>)", r"\1\2", xml, flags=re.S)
        assert mask(original) == mask(updated), "Non-text XML changed."
        final = ET.fromstring(updated).find(W + "body")
        assert len(body) == len(final) == 2924
        changed = [i for i, (a, b) in enumerate(zip(body, final)) if ET.tostring(a) != ET.tostring(b)]
        assert changed == list(REVISIONS), changed
        for index in changed:
            old_nodes, _, protected = prose_groups(body[index])
            new_nodes, _, _ = prose_groups(final[index])
            assert all(old_nodes[i].text == new_nodes[i].text for i in protected)
            for tag in (M + "oMath", W + "pPr", W + "hyperlink", W + "instrText", W + "fldChar"):
                assert [ET.tostring(n) for n in body[index].iter(tag)] == [ET.tostring(n) for n in final[index].iter(tag)]
            old_numbers = re.findall(r"\d+(?:\.\d+)?", "".join(body[index].itertext()))
            new_numbers = re.findall(r"\d+(?:\.\d+)?", "".join(final[index].itertext()))
            assert old_numbers == new_numbers, (index, old_numbers, new_numbers)
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
