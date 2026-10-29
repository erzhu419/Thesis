import re
import xml.etree.ElementTree as ET
from zipfile import ZipFile

from edit_section_5_4 import BACKUP, CAPTIONS, DOCUMENT, W, replace_caption


with ZipFile(DOCUMENT) as source:
    original = source.read("word/document.xml").decode("utf-8")
    body = ET.fromstring(original).find(W + "body")
    updated = original
    for index, text in CAPTIONS.items():
        assert "".join(n.text or "" for n in body[index].iter(W + "t")) == text
        updated = replace_caption(updated, body[index], text)
    mask = lambda xml: re.sub(r"(<w:t(?:\s[^>]*)?>).*?(</w:t>)", r"\1\2", xml, flags=re.S)
    assert mask(original) == mask(updated)
    final_body = ET.fromstring(updated).find(W + "body")
    changed = [i for i, (a, b) in enumerate(zip(body, final_body)) if ET.tostring(a) != ET.tostring(b)]
    assert set(changed) == set(CAPTIONS)
    with ZipFile(BACKUP) as before:
        previous = ET.fromstring(before.read("word/document.xml")).find(W + "body")
        for index in CAPTIONS:
            assert [n.text for n in previous[index].iter(W + "t")][:2] == [n.text for n in final_body[index].iter(W + "t")][:2]
    temporary = DOCUMENT.with_suffix(".section-edit.docx")
    with ZipFile(temporary, "w") as target:
        for item in source.infolist():
            target.writestr(item, updated.encode("utf-8") if item.filename == "word/document.xml" else source.read(item.filename))
temporary.replace(DOCUMENT)
print({"caption_number_ranges_restored": len(changed), "non_text_xml_preserved": True})
