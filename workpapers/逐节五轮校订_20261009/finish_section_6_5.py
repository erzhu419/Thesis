from zipfile import ZipFile
import re
import xml.etree.ElementTree as ET

from edit_section_1_2_1 import DOCUMENT, W
from edit_section_4_4 import replace_paragraph
from edit_section_6_5 import BACKUP, REVISIONS


with ZipFile(DOCUMENT) as source:
    xml = source.read("word/document.xml").decode("utf-8")
    body = ET.fromstring(xml).find(W + "body")
    updated = replace_paragraph(xml, body[1919], (
        "机制分析表明，预测误差与模型样本的使用强度需分别评价。早期公交训练的预测误差基本稳定，原置信度递推更新却使权重和梯度贡献持续衰减；归一化恢复了模型参与价值学习的作用。固定0.5样本权重也有效，说明收益不能全部归于状态相关置信排序。三个受控动力学任务各训练10次，MC-WM相对Target SAC的回报分别增加1925.7、1184.5和133.4，配对区间均为正，违例减少；相对匹配预算DARC的回报也提高。Ant执行器变化下，所选直接模型相对强制残差增加1529.9回报，表明源预测不合适时，替换模型有助于改善控制。",
    ))
    mask = lambda raw: re.sub(r"(<w:t(?:\s[^>]*)?>).*?(</w:t>)", r"\1\2", raw, flags=re.S)
    assert mask(xml) == mask(updated)
    new_body = ET.fromstring(updated).find(W + "body")
    assert [i for i, (a, b) in enumerate(zip(body, new_body)) if ET.tostring(a) != ET.tostring(b)] == [1919]
    temporary = DOCUMENT.with_suffix(".section-edit.docx")
    with ZipFile(temporary, "w") as target:
        for item in source.infolist():
            target.writestr(item, updated.encode("utf-8") if item.filename == "word/document.xml" else source.read(item.filename))
temporary.replace(DOCUMENT)
with ZipFile(BACKUP) as before, ZipFile(DOCUMENT) as after:
    assert before.namelist() == after.namelist()
    assert [n for n in before.namelist() if before.read(n) != after.read(n)] == ["word/document.xml"]
    initial = before.read("word/document.xml").decode("utf-8")
    assert mask(initial) == mask(updated)
    initial_body = ET.fromstring(initial).find(W + "body")
    changed = [i for i, (a, b) in enumerate(zip(initial_body, new_body)) if ET.tostring(a) != ET.tostring(b)]
    assert changed == list(REVISIONS)
print({"changed_blocks": changed, "non_text_XML_unchanged": True})
