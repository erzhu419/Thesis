from pathlib import Path
from shutil import copy2
from zipfile import ZipFile
import re
import xml.etree.ElementTree as ET

from edit_section_2_1 import DOCUMENT, BACKUP, W, M, W14, REVISIONS, TEXT_REVISIONS, revise_paragraph


with ZipFile(DOCUMENT) as source:
    original = source.read("word/document.xml").decode("utf-8")
    body = ET.fromstring(original).find(W + "body")
    paragraph = body[396]
    assert "这一共同控制任务下" in "".join(paragraph.itertext())
    pattern = r'<w:p\b[^>]*w14:paraId="' + paragraph.get(W14 + "paraId") + r'"[^>]*>.*?</w:p>'
    matches = list(re.finditer(pattern, original, flags=re.S))
    assert len(matches) == 1
    match = matches[0]
    revised = revise_paragraph(match.group(0), paragraph, REVISIONS[396])
    mask = lambda text: re.sub(r"(<w:t(?:\s[^>]*)?>).*?(</w:t>)", r"\1\2", text, flags=re.S)
    assert mask(match.group(0)) == mask(revised)
    updated = original[:match.start()] + revised + original[match.end():]
    preview_backup = Path(__file__).with_name("2.1文字初改.docx")
    assert not preview_backup.exists()
    copy2(DOCUMENT, preview_backup)
    temporary = DOCUMENT.with_suffix(".section-edit.docx")
    with ZipFile(temporary, "w") as target:
        for item in source.infolist():
            data = updated.encode("utf-8") if item.filename == "word/document.xml" else source.read(item.filename)
            target.writestr(item, data)
temporary.replace(DOCUMENT)

with ZipFile(BACKUP) as before, ZipFile(DOCUMENT) as after:
    assert before.namelist() == after.namelist()
    assert [n for n in before.namelist() if before.read(n) != after.read(n)] == ["word/document.xml"]
    old_xml = before.read("word/document.xml").decode("utf-8")
    new_xml = after.read("word/document.xml").decode("utf-8")
    assert mask(old_xml) == mask(new_xml)
    old_body = ET.fromstring(old_xml).find(W + "body")
    new_body = ET.fromstring(new_xml).find(W + "body")
    assert len(old_body) == len(new_body)
    changed = [i for i, (a, b) in enumerate(zip(old_body, new_body)) if ET.tostring(a) != ET.tostring(b)]
    assert changed == sorted(REVISIONS.keys() | TEXT_REVISIONS.keys())
    for i in changed:
        for tag in (W + "hyperlink", M + "oMath", W + "pPr", W + "rPr", W + "instrText", W + "fldChar"):
            assert [ET.tostring(x) for x in old_body[i].iter(tag)] == [ET.tostring(x) for x in new_body[i].iter(tag)]
print("Final section 2.1 scope verified:", changed)
print("Shortened body block 396 without modifying the following section.")
