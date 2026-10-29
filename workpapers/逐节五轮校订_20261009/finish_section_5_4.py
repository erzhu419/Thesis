import re
import xml.etree.ElementTree as ET
from zipfile import ZipFile

from edit_section_5_4 import CAPTIONS, DOCUMENT, W, replace_caption


with ZipFile(DOCUMENT) as source:
    original = source.read("word/document.xml").decode("utf-8")
    body = ET.fromstring(original).find(W + "body")
    updated = replace_caption(original, body[1446], CAPTIONS[1446])
    mask = lambda xml: re.sub(r"(<w:t(?:\s[^>]*)?>).*?(</w:t>)", r"\1\2", xml, flags=re.S)
    assert mask(original) == mask(updated)
    final_body = ET.fromstring(updated).find(W + "body")
    assert [i for i, (a, b) in enumerate(zip(body, final_body)) if ET.tostring(a) != ET.tostring(b)] == [1446]
    temporary = DOCUMENT.with_suffix(".section-edit.docx")
    with ZipFile(temporary, "w") as target:
        for item in source.infolist():
            target.writestr(item, updated.encode("utf-8") if item.filename == "word/document.xml" else source.read(item.filename))
temporary.replace(DOCUMENT)
print("Updated figure 5-8 caption; number bookmark and non-text XML preserved.")
