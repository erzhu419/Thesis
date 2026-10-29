from pathlib import Path
from shutil import copy2
from zipfile import ZipFile
import re
import xml.etree.ElementTree as ET

from edit_section_1_2_1 import DOCUMENT, W, W14, revise_paragraph


BACKUP = Path(__file__).with_name("校订前_3.1.docx")
REVISIONS = {
    641: (
        "沿用第2章的线路、发车计划和仿真设置，本章研究未预先获得目标域转移训练数据时的驻站学习。相同条件下，乘客到达、路段行驶与上下客时间的随机性仍会使同一动作产生不同后果。因此，需要利用仿真交互经验学习适用于多次运行的策略，而不是只拟合某一天的客流与路况。",
    ),
    642: (
        "本章假定，给定线路、站点和时段等条件后，随机生成规律保持不变。常规日内峰谷可由时段信息描述；同一时段的需求规律持续漂移或突然改变，则属于第4章的非平稳问题。偶然不确定性体现为动作后果的随机波动，有限经验造成的价值估计误差则属于认知不确定性。本章分别分析这两类影响。",
    ),
    643: (
        "仿真反复采样提供给定随机规律下的不同轨迹，域随机化则进一步改变仿真参数。本章在既定随机规律下改进SAC的价值学习：有限样本可能造成动作价值高估，进而使策略偏好实际服务效果较差的驻站动作。为此，引入集成价值评价和保守正则化调节动作选择，并分析其与第2章鲁棒评价的关系。",
    ),
    644: (
        "车辆的运行方向、站点位置和班次阶段不同，控制需求也不同。逐车训练时，班次与决策次数不同，使各车的转移样本量不均衡；共享策略若不保留运行身份，又可能混淆这些需求。本章用一个学习器依次处理车队决策，以类别特征嵌入区分车辆及运行位置，共享策略、价值网络和经验回放。",
    ),
    645: (
        "本章在类别特征嵌入的共享驻站策略基础上，以RE-SAC改进动作价值评价。",
        "以计划间隔为主要控制目标，并以广义乘客时间成本为另一训练目标，分别考察时距与乘客服务指标。模型与求解部分依次给出状态、动作、奖励、保守评价和策略更新方法，实验部分比较服务水平、训练稳定性及影响因素。RE-SAC利用固定目标离线数据的学习将在第5章另行讨论。",
    ),
    646: (
        "在给定训练预算与共同设置下，本章开展独立重复训练，观察不同训练过程的服务表现；通过独立仿真评价检验动作价值估计，并比较训练后期的性能变化。客流激增与快速切换的补充测试考察已训练策略面对运行变化时的表现。第4章进一步研究如何识别这种变化，并据此调整控制。",
    ),
}


def main():
    with ZipFile(DOCUMENT) as source:
        original = source.read("word/document.xml").decode("utf-8")
        body = ET.fromstring(original).find(W + "body")
        assert "".join(body[640].itertext()) == "3.1 问题描述"
        assert "".join(body[647].itertext()) == "3.2 模型构建"
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
        assert set(changed).issubset(REVISIONS), changed
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
        initial = before.read("word/document.xml").decode("utf-8")
        assert mask(initial) == mask(updated)
        initial_body = ET.fromstring(initial).find(W + "body")
        final_changed = [i for i, (a, b) in enumerate(zip(initial_body, new_body)) if ET.tostring(a) != ET.tostring(b)]
        assert final_changed == list(REVISIONS), final_changed
    print({"changed_blocks": changed, "changed_parts": changed_parts, "backup": str(BACKUP)})


if __name__ == "__main__":
    main()
