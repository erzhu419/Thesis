import re
import xml.etree.ElementTree as ET
from zipfile import ZipFile

from edit_section_5_4 import DOCUMENT, M, REVISIONS, W, replace_paragraph


INDICES = (1384, 1451, 1468, 1484, 1492, 1521, 1538)
with ZipFile(DOCUMENT) as source:
    original = source.read("word/document.xml").decode("utf-8")
    body = ET.fromstring(original).find(W + "body")
    updated = original
    for index in INDICES:
        updated = replace_paragraph(updated, body[index], REVISIONS[index])
    paragraphs = list(body[1515].iter(W + "p"))
    updated = replace_paragraph(updated, paragraphs[24], ("不额外驻站",))
    mask = lambda xml: re.sub(r"(<w:t(?:\s[^>]*)?>).*?(</w:t>)", r"\1\2", xml, flags=re.S)
    assert mask(original) == mask(updated)
    final_body = ET.fromstring(updated).find(W + "body")
    changed = [i for i, (a, b) in enumerate(zip(body, final_body)) if ET.tostring(a) != ET.tostring(b)]
    assert set(changed) == set(INDICES) | {1515}
    for index in changed:
        for tag in (M + "oMath", W + "instrText", W + "fldChar", W + "pPr"):
            assert [ET.tostring(n) for n in body[index].iter(tag)] == [ET.tostring(n) for n in final_body[index].iter(tag)]
    temporary = DOCUMENT.with_suffix(".section-edit.docx")
    with ZipFile(temporary, "w") as target:
        for item in source.infolist():
            target.writestr(item, updated.encode("utf-8") if item.filename == "word/document.xml" else source.read(item.filename))
temporary.replace(DOCUMENT)
print({"refined_body_blocks": changed, "non_text_xml_preserved": True})
