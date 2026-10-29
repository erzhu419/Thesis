from zipfile import ZipFile
import re
import xml.etree.ElementTree as ET

from edit_pass2_section_4_1 import BACKUP, DOCUMENT, REVISIONS, W, replace_paragraph


with ZipFile(DOCUMENT) as source:
    original = source.read("word/document.xml").decode("utf-8")
    body = ET.fromstring(original).find(W + "body")
    updated = replace_paragraph(original, body[992], REVISIONS[992])
    temporary = DOCUMENT.with_suffix(".tmp")
    with ZipFile(temporary, "w") as target:
        for item in source.infolist():
            target.writestr(item, updated.encode("utf-8") if item.filename == "word/document.xml" else source.read(item.filename))

with ZipFile(BACKUP) as before, ZipFile(temporary) as after:
    assert before.namelist() == after.namelist()
    assert [n for n in before.namelist() if before.read(n) != after.read(n)] == ["word/document.xml"]
    old = before.read("word/document.xml").decode("utf-8")
    new = after.read("word/document.xml").decode("utf-8")
    mask = lambda xml: re.sub(r"(<w:t(?:\s[^>]*)?>).*?(</w:t>)", r"\1\2", xml, flags=re.S)
    assert mask(old) == mask(new)
    a = ET.fromstring(old).find(W + "body")
    b = ET.fromstring(new).find(W + "body")
    assert len(a) == len(b) == 2924
    changed = [i for i, (x, y) in enumerate(zip(a, b)) if ET.tostring(x) != ET.tostring(y)]
    assert changed == list(REVISIONS), changed
    for i in changed:
        assert "".join(b[i].itertext()) == REVISIONS[i][0]
        assert re.findall(r"\d+(?:\.\d+)?", "".join(a[i].itertext())) == re.findall(r"\d+(?:\.\d+)?", "".join(b[i].itertext()))

temporary.replace(DOCUMENT)
print({"changed_blocks": changed, "other_content_and_format_preserved": True})
