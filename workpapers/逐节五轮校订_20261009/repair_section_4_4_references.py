from zipfile import ZipFile
from xml.sax.saxutils import escape
import re
import xml.etree.ElementTree as ET

from edit_section_4_4 import BACKUP, DOCUMENT, W, prose_groups, replace_paragraph


DUPLICATES = {
    1094: "图4-2", 1107: "图4-3", 1112: "表4-5", 1118: "图4-4",
    1120: "图4-5", 1128: "图4-6", 1140: "表4-6", 1143: "图4-7",
    1172: "图4-9", 1176: "图4-10", 1191: "表4-9",
}
REVISIONS = {
    1206: "表4-11中的乘客成本增加0.068%至0.268%。后续软符号配置采用可学习基准动作与中心化神经残差，关闭NAU，以SINDy参考施加软约束；配置为3个评论家、批量大小256。两类奖励各训练100回合，用两个训练种子、每场景三个共同评价日，与采用10个评论家、批量大小2048及10⁻⁵学习率的RE-SAC比较。乘客目标下，正常、单次5倍、重复5倍、持续5倍、持续10倍及短时50倍的成本分别降低62.80、67.36、44.63、27.78、14.18、51.58秒，即0.549%、0.440%、0.333%、0.172%、0.083%、0.245%；计划偏差减少7.35至12.09秒，时距不均衡度减少11.88至20.08秒。",
    1213: "公交评价执行固定策略，不单独检验评价期间的在线参数适应。单次脉冲采用两个训练种子和成对输入，六次冲击运行均未恢复；重复冲击采用三个训练种子与三个评价日交叉，早期BAPR与RE-SAC只复评一个训练种子。后续软符号比较采用两个训练种子，训练超参数与RE-SAC不同。各批均值反映相对各自参照的变化。累计候车量受需求日程与总量影响，不能直接衡量恢复速度；训练低回报比例不代表现实事件风险。",
}
CAPTION_NODES = {
    1092: ["表", "4-2", " ", "训练分布下的累计回报"],
    1100: ["表", "4-3", " ", "全程固定需求倍率下的回报中位数，单位：千"],
    1104: ["表", "4-4", " ", "回合内需求突变下的回报中位数，单位：千"],
    1116: ["表", "4-5", " ", "反复变化需求下的回报中位数，单位：千"],
    1141: ["表", "4-6", " ", "五种共同运行条件下的人均广义时间成本，单位：秒/已出发乘客"],
    1192: ["表", "4-9", " ", "输出结构与", "BC", "交叉比较的平均回报，单位：千"],
}

with ZipFile(DOCUMENT) as source:
    original = source.read("word/document.xml").decode("utf-8")
    body = ET.fromstring(original).find(W + "body")
    updated = original
    for index, label in DUPLICATES.items():
        nodes, groups, _ = prose_groups(body[index])
        prose = ["".join(nodes[j].text or "" for j in group) for group in groups]
        assert len(prose) == 1 and prose[0].startswith(label)
        updated = replace_paragraph(updated, body[index], (prose[0][len(label):],))
    for index, text in REVISIONS.items():
        updated = replace_paragraph(updated, body[index], (text,))
    for index, texts in CAPTION_NODES.items():
        paragraph_id = body[index].get("{http://schemas.microsoft.com/office/word/2010/wordml}paraId")
        pattern = r'<w:p\b[^>]*w14:paraId="' + paragraph_id + r'"[^>]*>.*?</w:p>'
        matches = list(re.finditer(pattern, updated, re.S))
        assert len(matches) == 1
        match = matches[0]
        assert len(list(body[index].iter(W + "t"))) == len(texts)
        assert "".join(n.text or "" for n in body[index].iter(W + "t")) == "".join(texts)
        values = iter(texts)
        paragraph = re.sub(r'(<w:t(?:\s[^>]*)?>).*?(</w:t>)', lambda m: m[1] + escape(next(values)) + m[2], match.group(0), flags=re.S)
        updated = updated[:match.start()] + paragraph + updated[match.end():]

    mask = lambda xml: re.sub(r"(<w:t(?:\s[^>]*)?>).*?(</w:t>)", r"\1\2", xml, flags=re.S)
    assert mask(original) == mask(updated)
    final_body = ET.fromstring(updated).find(W + "body")
    changed = [i for i, (a, b) in enumerate(zip(body, final_body)) if ET.tostring(a) != ET.tostring(b)]
    assert set(changed) == set(DUPLICATES) | set(REVISIONS) | set(CAPTION_NODES)
    for index in DUPLICATES:
        old_nodes, _, protected = prose_groups(body[index])
        new_nodes = list(final_body[index].iter(W + "t"))
        assert all(ET.tostring(old_nodes[j]) == ET.tostring(new_nodes[j]) for j in protected)
    temporary = DOCUMENT.with_suffix(".section-edit.docx")
    with ZipFile(temporary, "w") as target:
        for item in source.infolist():
            target.writestr(item, updated.encode("utf-8") if item.filename == "word/document.xml" else source.read(item.filename))
temporary.replace(DOCUMENT)
with ZipFile(BACKUP) as before, ZipFile(DOCUMENT) as after:
    assert before.namelist() == after.namelist()
    assert [n for n in before.namelist() if before.read(n) != after.read(n)] == ["word/document.xml"]
print({"repaired_blocks": changed, "caption_number_bookmarks_preserved": len(CAPTION_NODES)})
