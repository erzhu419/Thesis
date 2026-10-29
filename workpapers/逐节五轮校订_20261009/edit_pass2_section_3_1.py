from pathlib import Path
from shutil import copy2
from zipfile import ZipFile
import re
import xml.etree.ElementTree as ET

from edit_section_1_2_1 import W
from edit_section_3_2 import prose_groups
from edit_section_4_4 import replace_paragraph


ROOT = Path(__file__).resolve().parents[2]
DOCUMENT = ROOT / "output/排版完成_逐节校订_第2轮_20261010.docx"
BACKUP = Path(__file__).with_name("校订前_第2轮_3.1.docx")
REVISIONS = {
    641: (
        "沿用第2章的线路、发车计划和仿真设置，本章研究未预先获得目标域转移训练数据时的驻站学习。相同运行条件下，乘客到达、路段行驶与上下客时间的随机性仍会使同一驻站动作产生不同后果。因此，需要通过仿真反复采样学习策略，并检验其在不同随机运行中的服务表现，而不能只根据一条轨迹判断动作优劣。",
    ),
    643: (
        "即使随机生成规律保持不变，有限仿真经验仍可能使SAC高估某些驻站动作的长期价值，导致策略选择实际服务效果较差的动作。本章引入集成价值评价和保守正则化，改进动作价值学习，并分析其与第2章鲁棒评价的关系。",
    ),
    644: (
        "车辆的运行方向、站点位置和班次阶段不同，控制需求也不同。逐车训练时，班次与决策次数不同，使各车的转移样本量不均衡；共享策略若缺少车辆及运行位置的信息，又可能混淆这些需求。本章用一个学习器依次处理车队决策，以类别特征嵌入区分车辆及运行位置，共享策略、价值网络和经验回放。",
    ),
    645: (
        "本章将类别特征嵌入与RE-SAC结合，分别研究车队经验共享和动作价值评价。",
        "以车头时距稳定性为主要训练目标，并以广义乘客时间成本为另一训练目标，分别考察时距与乘客服务指标。实验区分类别表征、价值评价及训练目标的作用，并检验训练稳定性。RE-SAC利用固定目标离线数据的学习将在第5章另行讨论。",
    ),
    646: (
        "在给定训练预算与共同设置下，本章开展独立重复训练，观察不同训练过程的服务表现；通过独立仿真评价检验动作价值估计，并比较训练后期的性能变化。客流激增与快速切换的补充测试检验已训练策略面对运行变化时的表现，并不改变本章训练阶段的平稳假设。第4章进一步研究运行变化的识别与适应。",
    ),
}


def main():
    assert not BACKUP.exists(), "Keep the existing pre-edit backup."
    with ZipFile(DOCUMENT) as source:
        original = source.read("word/document.xml").decode("utf-8")
        body = ET.fromstring(original).find(W + "body")
        assert "".join(body[640].itertext()) == "3.1 问题描述"
        assert "".join(body[647].itertext()) == "3.2 模型构建"
        updated = original
        for index, prose in REVISIONS.items():
            assert 640 < index < 647
            updated = replace_paragraph(updated, body[index], prose)

        mask = lambda xml: re.sub(r"(<w:t(?:\s[^>]*)?>).*?(</w:t>)", r"\1\2", xml, flags=re.S)
        assert mask(original) == mask(updated), "Non-text XML changed."
        final_body = ET.fromstring(updated).find(W + "body")
        assert len(body) == len(final_body) == 2924
        changed = [i for i, (a, b) in enumerate(zip(body, final_body)) if ET.tostring(a) != ET.tostring(b)]
        assert changed == list(REVISIONS), changed
        protected_count = 0
        for index in changed:
            before_text = "".join(body[index].itertext())
            after_text = "".join(final_body[index].itertext())
            assert re.findall(r"\d+(?:\.\d+)?", before_text) == re.findall(r"\d+(?:\.\d+)?", after_text), index
            old_nodes, _, protected = prose_groups(body[index])
            new_nodes = list(final_body[index].iter(W + "t"))
            assert all(ET.tostring(old_nodes[j]) == ET.tostring(new_nodes[j]) for j in protected), index
            protected_count += len(protected)

        copy2(DOCUMENT, BACKUP)
        temporary = DOCUMENT.with_suffix(".tmp")
        with ZipFile(temporary, "w") as target:
            for item in source.infolist():
                target.writestr(item, updated.encode("utf-8") if item.filename == "word/document.xml" else source.read(item.filename))
    temporary.replace(DOCUMENT)
    with ZipFile(BACKUP) as before, ZipFile(DOCUMENT) as after:
        assert before.namelist() == after.namelist()
        assert [n for n in before.namelist() if before.read(n) != after.read(n)] == ["word/document.xml"]
    print({"changed_blocks": changed, "protected_text_nodes": protected_count,
           "backup": str(BACKUP), "output": str(DOCUMENT)})


if __name__ == "__main__":
    main()
