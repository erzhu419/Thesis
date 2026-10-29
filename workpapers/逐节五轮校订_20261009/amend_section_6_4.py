import re
import xml.etree.ElementTree as ET
from zipfile import ZipFile

from edit_section_1_2_1 import DOCUMENT, M, W
from edit_section_3_2 import prose_groups
from edit_section_4_4 import replace_paragraph


PARAGRAPHS = {
    1819: (
        "将这27次训练耗时分为不重叠阶段。模型拟合与更新包括初始源模型、目标候选及后续目标重拟合，并重新计算预测误差。常规优化耗时扣除已单列的置信更新、模型样本生成及损失计算、预测诊断；梯度范数与夹角未单独计时，仍计在常规优化中。",
    ),
    1829: (
        "取两个训练种子，复用各自50000次IQL初始化，再完成20000次策略与评论家更新、25000至25002个源事件。网络、学习率日程、目标数据及执行规则相同。源条件为正常标定、联合50%及75%偏差：50%时，速度为标定值0.5倍，OD与必要停站时间为1.5倍，额外路口比例为0.5；75%对应0.25倍、1.75倍及0.75。各路段额外路口数服从泊松分布，均值为该比例乘线路信号路口数、除以仿真路段数；新增路口沿用既有绿灯比例和信号周期生成延误。50%、75%是联合干预名称，不是统一的转移概率误差。",
    ),
    1830: (
        "每种源条件比较H2O+混合续训、混合系数0.75的无世界模型组、原MC-WM、归一化置信、固定0.5样本权重及归一化加源转移价值目标修正六组，共36次续训。两个初始化各作20000次纯离线IQL续训，总计38次；纯离线不随源条件重复。各策略在正常与1.25倍目标需求下各运行五个共同新SUMO日，共380个24000秒回合。先平均同种子五个评价日，再平均两次训练。",
    ),
    1831: (
        "列出人均广义时间成本。列名斜线前为源条件，后为目标需求倍率；“标定”指正常标定，50%、75%指联合偏差。",
    ),
}

TERMS_BY_BLOCK = {
    1795: {"正常标定": "正常"},
    1806: {"原H2O+参照": "H2O+混合续训组"},
    1808: {"0.75损失组": "混合系数0.75"},
    1811: {"原H2O+": "H2O+混合续训组"},
    1827: {
        "0.75损失组": "混合系数0.75",
        "贝尔曼置信评价更新/秒": "置信评价更新/秒",
    },
    1833: {
        "标定正常": "标定/1.00",
        "标定高需求": "标定/1.25",
        "50%正常": "50%/1.00",
        "50%高需求": "50%/1.25",
        "75%正常": "75%/1.00",
        "75%高需求": "75%/1.25",
        "0.75损失参照": "混合系数0.75",
    },
}


def main(paragraphs=PARAGRAPHS, terms_by_block=TERMS_BY_BLOCK):
    with ZipFile(DOCUMENT) as source:
        contents = {item.filename: source.read(item.filename) for item in source.infolist()}
        entries = source.infolist()
    original = contents["word/document.xml"].decode("utf-8")
    body = ET.fromstring(original).find(W + "body")
    updated = original
    for index, prose in paragraphs.items():
        updated = replace_paragraph(updated, body[index], prose)
    for index, terms in terms_by_block.items():
        for paragraph in body[index].iter(W + "p"):
            nodes, groups, _ = prose_groups(paragraph)
            old_prose = ["".join(nodes[i].text or "" for i in group) for group in groups]
            new_prose = old_prose.copy()
            for i, value in enumerate(new_prose):
                for old, new in terms.items():
                    value = value.replace(old, new)
                new_prose[i] = value
            if new_prose != old_prose:
                updated = replace_paragraph(updated, paragraph, tuple(new_prose))

    mask = lambda xml: re.sub(r"(<w:t(?:\s[^>]*)?>).*?(</w:t>)", r"\1\2", xml, flags=re.S)
    assert mask(original) == mask(updated), "Non-text XML changed."
    final = ET.fromstring(updated).find(W + "body")
    changed = [i for i, (a, b) in enumerate(zip(body, final)) if ET.tostring(a) != ET.tostring(b)]
    assert len(body) == len(final)
    assert set(changed) == set(paragraphs) | set(terms_by_block)
    assert all(1696 < i < 1916 for i in changed)
    assert [ET.tostring(n) for n in body.iter(M + "oMath")] == [ET.tostring(n) for n in final.iter(M + "oMath")]
    for index in changed:
        for before, after in zip(body[index].iter(W + "p"), final[index].iter(W + "p")):
            nodes, _, protected = prose_groups(before)
            result = list(after.iter(W + "t"))
            assert all(ET.tostring(nodes[i]) == ET.tostring(result[i]) for i in protected)
            style = before.find(W + "pPr/" + W + "pStyle")
            if style is not None and style.get(W + "val") in ("h1", "h2", "h3", "h4"):
                assert ET.tostring(before) == ET.tostring(after)
        if body[index].tag == W + "tbl":
            for row_index, (a, b) in enumerate(zip(body[index].findall(W + "tr"), final[index].findall(W + "tr"))):
                if row_index == 0:
                    continue
                before_cells = a.findall(W + "tc")[1:]
                after_cells = b.findall(W + "tc")[1:]
                assert [ET.tostring(n) for n in before_cells] == [ET.tostring(n) for n in after_cells], index

    temporary = DOCUMENT.with_suffix(".section-edit.docx")
    with ZipFile(temporary, "w") as target:
        for item in entries:
            target.writestr(item, updated.encode("utf-8") if item.filename == "word/document.xml" else contents[item.filename])
    temporary.replace(DOCUMENT)
    with ZipFile(DOCUMENT) as result:
        assert [name for name, data in contents.items() if result.read(name) != data] == ["word/document.xml"]
    print({"amended_blocks": changed, "non_text_xml_preserved": True, "table_data_unchanged": True})


if __name__ == "__main__":
    main()
