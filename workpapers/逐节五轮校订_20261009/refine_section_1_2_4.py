from pathlib import Path
from zipfile import ZipFile
import re
import xml.etree.ElementTree as ET

from edit_section_1_2_1 import DOCUMENT, W


BACKUP = Path(__file__).with_name("校订前_1.2.4.docx")
OLD = "车辆事件编码、不额外驻站与正驻站混合动作、行程完成保障"
NEW = "车辆事件编码、含不额外驻站选项的混合动作、行程完成保障"


with ZipFile(DOCUMENT) as source:
    original = source.read("word/document.xml").decode("utf-8")
    pattern = r'<w:p\b[^>]*w14:paraId="20C655E9"[^>]*>.*?</w:p>'
    matches = list(re.finditer(pattern, original, re.S))
    assert len(matches) == 1
    match = matches[0]
    assert OLD in match.group(0)
    updated = original[:match.start()] + match.group(0).replace(OLD, NEW, 1) + original[match.end():]
    before_body = ET.fromstring(original).find(W + "body")
    after_body = ET.fromstring(updated).find(W + "body")
    changed = [i for i, (a, b) in enumerate(zip(before_body, after_body)) if ET.tostring(a) != ET.tostring(b)]
    assert changed == [329]
    temporary = DOCUMENT.with_suffix(".table-refinement.docx")
    with ZipFile(temporary, "w") as target:
        for item in source.infolist():
            data = updated.encode("utf-8") if item.filename == "word/document.xml" else source.read(item.filename)
            target.writestr(item, data)
temporary.replace(DOCUMENT)

with ZipFile(BACKUP) as before, ZipFile(DOCUMENT) as after:
    assert before.namelist() == after.namelist()
    assert [name for name in before.namelist() if before.read(name) != after.read(name)] == ["word/document.xml"]
    old_xml = before.read("word/document.xml").decode("utf-8")
    new_xml = after.read("word/document.xml").decode("utf-8")
    mask = lambda text: re.sub(r"(<w:t(?:\s[^>]*)?>).*?(</w:t>)", r"\1\2", text, flags=re.S)
    assert mask(old_xml) == mask(new_xml)
    old_body = ET.fromstring(old_xml).find(W + "body")
    new_body = ET.fromstring(new_xml).find(W + "body")
    assert len(old_body) == len(new_body)
    changed = [i for i, (a, b) in enumerate(zip(old_body, new_body)) if ET.tostring(a) != ET.tostring(b)]
    assert changed == list(range(327, 335))
print("Final changes confined to body blocks 327-334 in section 1.2.4.")
print("Only text changed; abstracts, fields, equations, objects and formatting preserved.")
