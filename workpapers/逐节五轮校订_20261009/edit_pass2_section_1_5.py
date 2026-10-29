from pathlib import Path
from shutil import copy2
from zipfile import ZipFile
import re
import xml.etree.ElementTree as ET

from edit_section_1_2_1 import W
from edit_section_4_4 import replace_paragraph


ROOT = Path(__file__).resolve().parents[2]
DOCUMENT = ROOT / "output/排版完成_逐节校订_第2轮_20261010.docx"
BACKUP = Path(__file__).with_name("校订前_第2轮_1.5.docx")
REVISIONS = {
    354: (
        "第3章研究平稳随机环境下的鲁棒控制，以类别嵌入支持车队共享驻站策略，以RE-SAC改进动作价值评价，通过SAC对照和组件消融检验时距控制与乘客时间效果。",
    ),
    355: (
        "第4章研究非平稳运行，介绍BAPR的模式识别及CS-BAPR的训练调节、驻站输出和符号参考约束，通过客流变化实验与组件对照检验相对RE-SAC的控制效果。",
    ),
    356: (
        "第5章研究固定目标离线数据的利用，先比较离线学习与目标在线微调，再将H2O+适配于公交，结合目标记录与源仿真新经验训练，分别检验混合训练和完成保障的服务效果。",
    ),
    357: (
        "第6章针对源仿真标定不足，研究MC-WM的模型选择与预测修正，以归一化置信更新确定一步模型样本的训练权重，辅助价值学习；通过受控动力学与公交实验检验机制及控制效果。",
    ),
}


def main():
    assert not BACKUP.exists(), "Keep the pre-edit backup."
    with ZipFile(DOCUMENT) as source:
        original = source.read("word/document.xml").decode("utf-8")
        body = ET.fromstring(original).find(W + "body")
        assert "".join(body[351].itertext()) == "1.5 论文结构"
        assert "".join(body[362].itertext()).startswith("第2章 ")
        updated = original
        for index, prose in REVISIONS.items():
            updated = replace_paragraph(updated, body[index], prose)

        # Text-only comparison protects equations, fields, objects and formatting.
        mask = lambda xml: re.sub(r"(<w:t(?:\s[^>]*)?>).*?(</w:t>)", r"\1\2", xml, flags=re.S)
        assert mask(original) == mask(updated), "Non-text XML changed."
        final_body = ET.fromstring(updated).find(W + "body")
        assert len(body) == len(final_body)
        changed = [i for i, (a, b) in enumerate(zip(body, final_body)) if ET.tostring(a) != ET.tostring(b)]
        assert changed == list(REVISIONS), changed
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
