from zipfile import ZipFile
import re
import xml.etree.ElementTree as ET

from edit_pass2_section_1_2 import BACKUP, DOCUMENT, REVISIONS, W
from edit_section_4_4 import replace_paragraph


with ZipFile(DOCUMENT) as source:
    original = source.read("word/document.xml").decode("utf-8")
    body = ET.fromstring(original).find(W + "body")
    updated = original
    for index in (304,):
        updated = replace_paragraph(updated, body[index], REVISIONS[index])
    mask = lambda xml: re.sub(r"(<w:t(?:\s[^>]*)?>).*?(</w:t>)", r"\1\2", xml, flags=re.S)
    assert mask(original) == mask(updated)
    revised = ET.fromstring(updated).find(W + "body")
    assert len(body) == len(revised)
    changed = [i for i, (a, b) in enumerate(zip(body, revised)) if ET.tostring(a) != ET.tostring(b)]
    assert changed == [304], changed
    with ZipFile(DOCUMENT.with_suffix(".tmp"), "w") as target:
        for item in source.infolist():
            target.writestr(item, updated.encode("utf-8") if item.filename == "word/document.xml" else source.read(item.filename))
DOCUMENT.with_suffix(".tmp").replace(DOCUMENT)
with ZipFile(BACKUP) as before, ZipFile(DOCUMENT) as after:
    assert before.namelist() == after.namelist()
    assert [n for n in before.namelist() if before.read(n) != after.read(n)] == ["word/document.xml"]
    old = before.read("word/document.xml").decode("utf-8")
    assert mask(old) == mask(updated)
    old_body = ET.fromstring(old).find(W + "body")
    cumulative = [i for i, (a, b) in enumerate(zip(old_body, revised)) if ET.tostring(a) != ET.tostring(b)]
    assert cumulative == list(REVISIONS), cumulative
print({"refined_blocks": changed, "cumulative_changed_blocks": cumulative})
