from zipfile import ZipFile
import re
import xml.etree.ElementTree as ET

from edit_section_3_3 import DOCUMENT, REVISIONS, W, paragraph_pattern, revise


with ZipFile(DOCUMENT) as source:
    original = source.read("word/document.xml").decode("utf-8")
    body = ET.fromstring(original).find(W + "body")
    updated = original
    for index in (793,):
        match = re.search(paragraph_pattern(body[index]), updated, re.S)
        revised = revise(match.group(0), body[index], REVISIONS[index])
        updated = updated[:match.start()] + revised + updated[match.end():]
    mask = lambda value: re.sub(r"(<w:t(?:\s[^>]*)?>).*?(</w:t>)", r"\1\2", value, flags=re.S)
    assert mask(original) == mask(updated)
    final = ET.fromstring(updated).find(W + "body")
    changed = [i for i, (a, b) in enumerate(zip(body, final)) if ET.tostring(a) != ET.tostring(b)]
    assert changed == [793], changed
    temporary = DOCUMENT.with_suffix(".section-refine.docx")
    with ZipFile(temporary, "w") as target:
        for item in source.infolist():
            target.writestr(item, updated.encode("utf-8") if item.filename == "word/document.xml" else source.read(item.filename))
temporary.replace(DOCUMENT)
print({"refined_paragraphs": changed, "non_text_xml_unchanged": True})
