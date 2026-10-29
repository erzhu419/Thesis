from pathlib import Path
from shutil import copy2
from zipfile import ZipFile
import re
import xml.etree.ElementTree as ET

from edit_section_1_2_1 import W
from edit_section_4_4 import replace_paragraph


ROOT = Path(__file__).resolve().parents[2]
DOCUMENT = ROOT / "output/排版完成_逐节校订_第2轮_20261010.docx"
BACKUP = Path(__file__).with_name("校订前_第2轮_2.5.docx")
REVISIONS = {
    633: (
        "本章建立了公交事件驱动的驻站决策模型，明确运行状态、附加驻站动作、事件转移与奖励的关系，区分车头时距和乘客时间两类训练目标，并说明行程完成情况等服务评价指标。在此基础上，介绍价值与策略学习、鲁棒评价、运行条件识别、源仿真经验加权和运行预测的基础方法，说明它们利用的信息和作用环节。仿真部分交代基础数据、平台构建、参数标定、控制响应检验与实验设置，为后续各章明确训练和比较条件。",
    ),
    634: (
        "在已标定的多线路场景中，简易仿真与SUMO在不额外驻站时的运行指标较为接近，施加驻站后的平均响应也较为一致。七种驻站策略与三档需求形成21个组合。各组合的广义时间归一化响应差及完成率响应差均计算95%区间，其端点绝对值分别不超过3.90%和0.574个百分点。这些结果支持以简易仿真进行源环境训练，再到SUMO中检验策略的控制效果。后续各章先在未预先获得目标域转移训练数据时，研究平稳随机运行下的鲁棒学习，以及运行规律变化后的识别与适应；再利用固定目标离线数据研究跨保真训练，并在源仿真标定不足时引入预测修正。",
    ),
}


def main():
    assert not BACKUP.exists(), "Keep the existing pre-edit backup."
    with ZipFile(DOCUMENT) as source:
        original = source.read("word/document.xml").decode("utf-8")
        body = ET.fromstring(original).find(W + "body")
        assert "".join(body[632].itertext()) == "2.5 本章小结"
        assert "".join(body[636].itertext()).startswith("第3章 ")
        updated = original
        for index, prose in REVISIONS.items():
            assert 632 < index < 635
            updated = replace_paragraph(updated, body[index], prose)

        mask = lambda xml: re.sub(r"(<w:t(?:\s[^>]*)?>).*?(</w:t>)", r"\1\2", xml, flags=re.S)
        assert mask(original) == mask(updated), "Non-text XML changed."
        final_body = ET.fromstring(updated).find(W + "body")
        assert len(body) == len(final_body)
        changed = [i for i, (a, b) in enumerate(zip(body, final_body)) if ET.tostring(a) != ET.tostring(b)]
        assert changed == list(REVISIONS), changed
        for index in changed:
            before_text = "".join(body[index].itertext())
            after_text = "".join(final_body[index].itertext())
            assert re.findall(r"\d+(?:\.\d+)?", before_text) == re.findall(r"\d+(?:\.\d+)?", after_text), index

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
           "backup": str(BACKUP), "output": str(DOCUMENT)})


if __name__ == "__main__":
    main()
