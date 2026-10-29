from pathlib import Path
from shutil import copy2
from zipfile import ZipFile
import re
import xml.etree.ElementTree as ET

from edit_section_1_2_1 import DOCUMENT, W, W14, revise_paragraph


BACKUP = Path(__file__).with_name("校订前_2.5.docx")
REVISIONS = {
    633: (
        "本章建立了公交事件驱动的驻站决策模型，明确了运行状态、附加驻站动作、事件转移和奖励之间的关系，说明车头时距与乘客时间两类训练目标的区别，以及行程完成情况等服务评价指标的含义。在此基础上，介绍了价值与策略学习、鲁棒评价、运行条件识别、源仿真经验加权及运行预测方法，说明各类方法利用的信息及其作用环节。仿真部分统一介绍基础数据、平台构建、参数标定、控制响应和实验设置，为后续各章确定训练与比较条件。",
    ),
    634: (
        "在已标定的多线路场景中，简易仿真与SUMO在不额外驻站时的运行指标较为接近，施加驻站后的平均响应也较为一致。七种驻站策略与三档需求形成的21个组合中，广义时间归一化响应差的95%区间端点绝对值均不超过3.90%，完成率响应差的对应上限为0.574个百分点。这些检验支持简易仿真承担源环境训练，所学策略的控制效果在SUMO中评价。后续研究据此从给定随机规律下的鲁棒学习，推进到运行规律变化后的识别与适应；获得固定目标离线数据后，再研究跨保真训练及源仿真标定不足时的预测修正。",
    ),
}


def main():
    with ZipFile(DOCUMENT) as source:
        original = source.read("word/document.xml").decode("utf-8")
        body = ET.fromstring(original).find(W + "body")
        assert "".join(body[632].itertext()) == "2.5 本章小结"
        assert "".join(body[636].itertext()).startswith("第3章 ")
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
        assert changed == list(REVISIONS), changed
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
