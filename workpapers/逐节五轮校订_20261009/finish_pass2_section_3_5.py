from zipfile import ZipFile
import re
import xml.etree.ElementTree as ET

from edit_pass2_section_3_5 import DOCUMENT, BACKUP, REVISIONS
from edit_section_1_2_1 import W
from edit_section_4_4 import replace_paragraph


with ZipFile(DOCUMENT) as source, ZipFile(BACKUP) as before:
    original = source.read("word/document.xml").decode("utf-8")
    baseline = before.read("word/document.xml").decode("utf-8")
    body = ET.fromstring(original).find(W + "body")
    prose = REVISIONS[982][0].replace(
        "另在同一集成结构内，保留评论家分歧与演员保守评价两项，相对同时移除两项，在正常运行、客流激增和快速切换中使平均计划偏差分别下降",
        "在同一集成结构内，相对同时移除评论家分歧与演员保守评价两项，保留两项使上述三种场景的平均计划偏差分别下降",
    ).replace(
        "该消融比较的时距结果为均值改善，乘客成本的配对区间则均支持下降。",
        "时距结果仅为均值改善，乘客成本区间均支持下降。",
    )
    updated = replace_paragraph(original, body[982], (prose,))
    updated = replace_paragraph(updated, body[983], (
        "集成规模需结合训练预算选择，扩大规模未带来持续收益。固定温度缓解了长期退化，但价值拟合误差更小，不代表动作排序和控制效果更好。计划间隔奖励减小计划偏差，却增加所测乘客时间；指数衰减尺度与驻站代价也影响这一取舍。驻站不能替代供给与计划调整。运行规律变化后，历史经验可能不再适用。下一章据此研究BAPR与CS-BAPR的条件识别与非平稳控制。",
    ))
    mask = lambda xml: re.sub(r"(<w:t(?:\s[^>]*)?>).*?(</w:t>)", r"\1\2", xml, flags=re.S)
    assert mask(baseline) == mask(updated), "Non-text XML changed."
    old_body = ET.fromstring(baseline).find(W + "body")
    final = ET.fromstring(updated).find(W + "body")
    assert len(old_body) == len(final) == 2924
    changed = [i for i, (a, b) in enumerate(zip(old_body, final)) if ET.tostring(a) != ET.tostring(b)]
    assert changed == [981, 982, 983], changed
    for index in changed:
        assert re.findall(r"\d+(?:\.\d+)?", "".join(old_body[index].itertext())) == re.findall(r"\d+(?:\.\d+)?", "".join(final[index].itertext()))
    assert before.namelist() == source.namelist()
    assert all(before.read(n) == source.read(n) for n in before.namelist() if n != "word/document.xml")
    temporary = DOCUMENT.with_suffix(".tmp")
    with ZipFile(temporary, "w") as target:
        for item in source.infolist():
            target.writestr(item, updated.encode("utf-8") if item.filename == "word/document.xml" else source.read(item.filename))
temporary.replace(DOCUMENT)
print({"changed_blocks": changed, "output": str(DOCUMENT)})
