from zipfile import ZipFile
import re
import xml.etree.ElementTree as ET

from edit_pass2_section_1_3 import BACKUP, DOCUMENT, REVISIONS, W, replace_paragraph


with ZipFile(DOCUMENT) as source, ZipFile(BACKUP) as backup:
    original = source.read("word/document.xml").decode("utf-8")
    body = ET.fromstring(original).find(W + "body")
    updated = replace_paragraph(original, body[337], REVISIONS[337])
    mask = lambda xml: re.sub(r"(<w:t(?:\s[^>]*)?>).*?(</w:t>)", r"\1\2", xml, flags=re.S)
    assert mask(original) == mask(updated)
    initial = ET.fromstring(backup.read("word/document.xml")).find(W + "body")
    final = ET.fromstring(updated).find(W + "body")
    assert len(initial) == len(final)
    changed = [i for i, (a, b) in enumerate(zip(initial, final)) if ET.tostring(a) != ET.tostring(b)]
    assert changed == list(REVISIONS), changed
    assert all(source.read(n) == backup.read(n) for n in source.namelist() if n != "word/document.xml")
    temporary = DOCUMENT.with_suffix(".tmp")
    with ZipFile(temporary, "w") as target:
        for item in source.infolist():
            target.writestr(item, updated.encode("utf-8") if item.filename == "word/document.xml" else source.read(item.filename))
temporary.replace(DOCUMENT)
print({"changed_blocks_from_backup": changed, "refined_block": 337})
