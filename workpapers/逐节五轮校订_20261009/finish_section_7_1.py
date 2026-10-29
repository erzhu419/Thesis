from zipfile import ZipFile
import re
import xml.etree.ElementTree as ET

from edit_section_1_2_1 import DOCUMENT, W
from edit_section_4_4 import replace_paragraph
from edit_section_7_1 import BACKUP, REVISIONS


with ZipFile(DOCUMENT) as source:
    xml = source.read("word/document.xml").decode("utf-8")
    body = ET.fromstring(xml).find(W + "body")
    updated = xml
    for index in (1927, 1934):
        updated = replace_paragraph(updated, body[index], REVISIONS[index])
    mask = lambda raw: re.sub(r"(<w:t(?:\s[^>]*)?>).*?(</w:t>)", r"\1\2", raw, flags=re.S)
    assert mask(xml) == mask(updated)
    final_body = ET.fromstring(updated).find(W + "body")
    assert [i for i, (a, b) in enumerate(zip(body, final_body)) if ET.tostring(a) != ET.tostring(b)] == [1927, 1934]
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
    changed = [i for i, (a, b) in enumerate(zip(initial_body, final_body)) if ET.tostring(a) != ET.tostring(b)]
    assert changed == list(REVISIONS)
print({"changed_blocks": changed, "non_text_XML_unchanged": True})
