from zipfile import ZipFile
import re
import xml.etree.ElementTree as ET

from edit_section_1_2_3 import BACKUP, DOCUMENT, W, M, W14, REVISIONS, revise_paragraph


with ZipFile(DOCUMENT) as source:
    original = source.read("word/document.xml").decode("utf-8")
    body = ET.fromstring(original).find(W + "body")
    paragraph_id = body[318].get(W14 + "paraId")
    pattern = r'<w:p\b[^>]*w14:paraId="' + paragraph_id + r'"[^>]*>.*?</w:p>'
    matches = list(re.finditer(pattern, original, flags=re.S))
    assert len(matches) == 1
    match = matches[0]
    revised = revise_paragraph(match.group(0), body[318], REVISIONS[318])
    mask = lambda raw: re.sub(r"(<w:t(?:\s[^>]*)?>).*?(</w:t>)", r"\1\2", raw, flags=re.S)
    assert mask(match.group(0)) == mask(revised)
    updated = original[:match.start()] + revised + original[match.end():]
    temporary = DOCUMENT.with_suffix(".section-edit.docx")
    with ZipFile(temporary, "w") as target:
        for item in source.infolist():
            data = updated.encode("utf-8") if item.filename == "word/document.xml" else source.read(item.filename)
            target.writestr(item, data)
temporary.replace(DOCUMENT)
with ZipFile(BACKUP) as before, ZipFile(DOCUMENT) as after:
    assert before.namelist() == after.namelist()
    assert [name for name in before.namelist() if before.read(name) != after.read(name)] == ["word/document.xml"]
    old = ET.fromstring(before.read("word/document.xml")).find(W + "body")
    new = ET.fromstring(after.read("word/document.xml")).find(W + "body")
    assert len(old) == len(new)
    assert [i for i, (a, b) in enumerate(zip(old, new)) if ET.tostring(a) != ET.tostring(b)] == list(REVISIONS)
    for i in REVISIONS:
        for tag in (W + "hyperlink", M + "oMath", W + "pPr"):
            assert [ET.tostring(x) for x in old[i].iter(tag)] == [ET.tostring(x) for x in new[i].iter(tag)]
print("Reflow repair completed; only the section 1.2.3 prose differs from its original backup.")
