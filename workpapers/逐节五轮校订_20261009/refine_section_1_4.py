from zipfile import ZipFile
import re
import xml.etree.ElementTree as ET

from edit_section_1_4 import DOCUMENT, W, W14, PARAGRAPH_REVISIONS, TABLE_REVISIONS
from edit_section_1_2_4 import replace_text_nodes


with ZipFile(DOCUMENT) as source:
    original = source.read("word/document.xml").decode("utf-8")
    body = ET.fromstring(original).find(W + "body")
    edits = [(body[i], PARAGRAPH_REVISIONS[i]) for i in (344, 347, 350)]
    rows = body[349].findall(W + "tr")
    for row, column in ((1, 2), (2, 2), (4, 1), (4, 3)):
        paragraph = rows[row].findall(W + "tc")[column].find(W + "p")
        nodes = list(paragraph.iter(W + "t"))
        text = TABLE_REVISIONS[(row, column)]
        edits.append((paragraph, {i: text if i == 0 else "" for i in range(len(nodes))}))
    updated = original
    mask = lambda raw: re.sub(r"(<w:t(?:\s[^>]*)?>).*?(</w:t>)", r"\1\2", raw, flags=re.S)
    for paragraph, replacements in edits:
        pattern = r'<w:p\b[^>]*w14:paraId="' + paragraph.get(W14 + "paraId") + r'"[^>]*>.*?</w:p>'
        matches = list(re.finditer(pattern, updated, flags=re.S))
        assert len(matches) == 1
        match = matches[0]
        revised = replace_text_nodes(match.group(0), replacements)
        assert mask(match.group(0)) == mask(revised)
        updated = updated[:match.start()] + revised + updated[match.end():]
    new_body = ET.fromstring(updated).find(W + "body")
    changed = [i for i, (a, b) in enumerate(zip(body, new_body)) if ET.tostring(a) != ET.tostring(b)]
    assert changed == [344, 347, 349, 350], changed
    temporary = DOCUMENT.with_suffix(".section-edit.docx")
    with ZipFile(temporary, "w") as target:
        for item in source.infolist():
            target.writestr(item, updated.encode("utf-8") if item.filename == "word/document.xml" else source.read(item.filename))
temporary.replace(DOCUMENT)
print("Refined section 1.4 only:", changed)
