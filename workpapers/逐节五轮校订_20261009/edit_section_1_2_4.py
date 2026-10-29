from html import escape
from pathlib import Path
from shutil import copy2
from zipfile import ZipFile
import re
import xml.etree.ElementTree as ET

from edit_section_1_2_1 import DOCUMENT, W, M, W14, revise_paragraph


BACKUP = Path(__file__).with_name("校订前_1.2.4.docx")

INTRO = (
    "上述研究表明，公交驻站控制的学习困难不仅来自随机运行，还取决于运行规律是否变化，以及训练时能否获得目标环境信息。共享策略与保守评价、运行条件识别、源目标混合学习及世界模型分别作用于这些问题。本文据此按可用数据与运行条件组织研究。",
    "进一步对比离线学习、混合学习和模型学习中的代表方法，说明第5、6章采用的通用原理与本文公交适配的区别。",
)

REVISIONS = {
    330: "这些方法用于公交时，需要将学习变量与实际服务过程对应。类别信息区分线路、车辆和站点，连续运行量反映时距与负荷；驻站动作只在必要上下客结束后延缓离站，事件之间的实际时间决定候车和车内成本。本文据此构建公交事件决策模型，以驻站为共同动作，并分别考察时距控制与乘客服务。结合前述研究，本文重点研究以下四个相互衔接的问题。",
    331: "第一，在训练前尚无目标域转移数据、给定条件下的随机规律保持不变时，如何从仿真经验中学到可复用的驻站策略？不同车辆、站点和时段的经验量不均衡，共享参数还需保留对象差异；有限经验下的动作价值估计可能偏高，影响驻站选择。第3章以类别特征嵌入和RE-SAC处理这两个问题，通过表征与价值评价对照，检验其对时距、候车和车内时间的作用。",
    332: "第二，当客流或路况规律发生突变、漂移或反复切换时，如何判断当前条件并作出合适响应？这种非平稳变化不能仅用原有条件下的随机扰动表示，近期运行信息又可能因噪声和识别滞后而不充分。第4章从BAPR的模式识别与置信调节出发，进一步研究CS-BAPR的训练稳定化、策略输出结构及符号参考约束，检验需求变化下的时距控制和乘客时间表现。",
    333: "第三，获得一部分目标离线数据后，如何在不追加目标训练交互的条件下利用源仿真改进策略？目标记录反映目标环境的运行经验，但只能覆盖采集时出现的状态动作；源仿真能响应新动作，却可能产生不同的后续结果。第5章将H2O+用于公交，结合目标离线初始化与源仿真交互训练，并在共同驻站范围和执行规则下分别评价学习方案与行程完成保障的作用。",
    334: "第四，当源仿真标定不充分时，如何利用同一类目标数据修正预测，并使预测真正参与策略学习？经验重加权调整样本对更新的影响，并不直接改变源模型的预测关系。第6章以MC-WM比较保留源预测的残差修正与直接目标建模，再按预测可信程度调节模型样本的价值更新权重，通过标定偏差和模型使用方式对照，检验预测修正的乘客服务收益。",
}

TABLE_REVISIONS = {
    (0, 2): "主要方法",
    (0, 3): "与本文的关系",
    (1, 1): "固定离线数据；可另做在线微调",
    (1, 2): "数据内价值学习与优势加权BC",
    (1, 3): "用于离线初始化，并作为纯离线参照",
    (2, 1): "固定目标离线数据与源环境交互",
    (2, 2): "根据动力学差异调整保守价值学习",
    (2, 3): "提供固定目标数据辅助源训练的思路",
    (3, 1): "固定目标离线数据与源环境交互",
    (3, 2): "离线与在线价值目标混合、动力学比率修正",
    (3, 3): "第5章公交迁移学习的通用基础",
    (4, 1): "环境采样与模型生成样本",
    (4, 2): "从环境记录的状态出发，预测少量后续转移",
    (4, 3): "为模型样本使用提供参照，不要求保留源模型",
    (5, 0): "本文公交H2O+适配",
    (5, 1): "固定SUMO离线数据与简易仿真交互",
    (5, 2): "车辆事件编码、含不额外驻站选项的混合动作、行程完成保障",
    (5, 3): "在相同驻站范围下，分别评价混合训练与执行保障",
    (6, 0): "本文公交MC-WM",
    (6, 1): "沿用固定SUMO数据与源交互，增加预测模型拟合",
    (6, 2): "混合事件预测、残差与直接模型选择、模型样本置信加权",
    (6, 3): "检验模型辅助能否降低人均广义时间成本",
}


def replace_text_nodes(raw, replacements):
    position = -1

    def replace(match):
        nonlocal position
        position += 1
        if position not in replacements:
            return match.group(0)
        return match.group(1) + escape(replacements[position], quote=False) + match.group(3)

    result = re.sub(r"(<w:t(?:\s[^>]*)?>)(.*?)(</w:t>)", replace, raw, flags=re.S)
    assert max(replacements) <= position
    return result


def main():
    with ZipFile(DOCUMENT) as source:
        original = source.read("word/document.xml").decode("utf-8")
        body = ET.fromstring(original).find(W + "body")
        assert "".join(body[326].itertext()) == "1.2.4 现有研究总结"
        assert "".join(body[335].itertext()) == "1.3 研究目标与内容"
        updated = original
        edits = [(body[327], None), (body[328], None)]
        edits.extend((body[index], prose) for index, prose in REVISIONS.items())
        rows = body[329].findall(W + "tr")
        for (row, column), prose in TABLE_REVISIONS.items():
            cell = rows[row].findall(W + "tc")[column]
            paragraphs = cell.findall(W + "p")
            assert len(paragraphs) == 1
            edits.append((paragraphs[0], prose))
        for paragraph, prose in edits:
            paragraph_id = paragraph.get(W14 + "paraId")
            pattern = r'<w:p\b[^>]*w14:paraId="' + paragraph_id + r'"[^>]*>.*?</w:p>'
            matches = list(re.finditer(pattern, updated, flags=re.S))
            assert len(matches) == 1, paragraph_id
            match = matches[0]
            raw = match.group(0)
            if paragraph is body[327]:
                assert [x.text for x in paragraph.iter(W + "t")][3:5] == ["表", "1-1"]
                revised = replace_text_nodes(raw, {0: INTRO[0], 1: "", 2: "", 5: INTRO[1]})
            elif paragraph is body[328]:
                revised = replace_text_nodes(raw, {3: "代表性学习方法与本文公交适配的关系"})
            else:
                revised = revise_paragraph(raw, paragraph, (prose,))
            # Preserve fields, objects, styles and run properties byte for byte.
            mask = lambda text: re.sub(r"(<w:t(?:\s[^>]*)?>).*?(</w:t>)", r"\1\2", text, flags=re.S)
            assert mask(raw) == mask(revised)
            updated = updated[:match.start()] + revised + updated[match.end():]
        new_body = ET.fromstring(updated).find(W + "body")
        assert len(body) == len(new_body)
        changed = [i for i, (a, b) in enumerate(zip(body, new_body)) if ET.tostring(a) != ET.tostring(b)]
        assert changed == list(range(327, 335)), changed
        for index in changed:
            for tag in (W + "hyperlink", M + "oMath", W + "pPr", W + "rPr", W + "instrText", W + "fldChar"):
                assert [ET.tostring(x) for x in body[index].iter(tag)] == [ET.tostring(x) for x in new_body[index].iter(tag)]
        assert not BACKUP.exists(), "Keep the earlier backup and inspect the manuscript before rerunning."
        copy2(DOCUMENT, BACKUP)
        temporary = DOCUMENT.with_suffix(".section-edit.docx")
        with ZipFile(temporary, "w") as target:
            for item in source.infolist():
                data = updated.encode("utf-8") if item.filename == "word/document.xml" else source.read(item.filename)
                target.writestr(item, data)
    temporary.replace(DOCUMENT)
    with ZipFile(BACKUP) as before, ZipFile(DOCUMENT) as after:
        assert before.namelist() == after.namelist()
        assert [name for name in before.namelist() if before.read(name) != after.read(name)] == ["word/document.xml"]
    print("Changed body blocks:", changed)
    print("Only section 1.2.4 text changed; abstracts, citations, fields, math and formatting preserved.")
    for index in changed:
        print(index, "".join(n.text or "" for n in new_body[index].iter() if n.tag in (W + "t", M + "t")))
    print("OUTPUT", DOCUMENT)


if __name__ == "__main__":
    main()
