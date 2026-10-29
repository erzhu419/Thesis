from zipfile import ZipFile
import xml.etree.ElementTree as ET

from edit_section_4_4 import DOCUMENT, W, prose_groups, replace_paragraph


with ZipFile(DOCUMENT) as source:
    original = source.read("word/document.xml").decode("utf-8")
    body = ET.fromstring(original).find(W + "body")
    nodes, groups, _ = prose_groups(body[1206])
    assert len(groups) == 1
    text = "".join(nodes[j].text or "" for j in groups[0])
    prefix = "表4-11中的乘客成本增加0.068%至0.268%。"
    assert text.startswith(prefix)
    updated = replace_paragraph(original, body[1206], (text[len(prefix):],))
    final_body = ET.fromstring(updated).find(W + "body")
    assert [i for i, (a, b) in enumerate(zip(body, final_body)) if ET.tostring(a) != ET.tostring(b)] == [1206]
    temporary = DOCUMENT.with_suffix(".section-edit.docx")
    with ZipFile(temporary, "w") as target:
        for item in source.infolist():
            target.writestr(item, updated.encode("utf-8") if item.filename == "word/document.xml" else source.read(item.filename))
temporary.replace(DOCUMENT)
print("Removed the repeated table 4-11 summary from block 1206.")
