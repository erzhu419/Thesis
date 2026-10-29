from zipfile import ZipFile
import re
import xml.etree.ElementTree as ET

from edit_section_1_2_1 import DOCUMENT, W, M, W14, revise_paragraph
from edit_section_1_2_4 import replace_text_nodes
from edit_section_2_4 import BACKUP, REVISIONS, FIELD_REVISIONS, TABLE_REVISIONS, field_results


INDICES = [521, 532, 546, 553, 560, 563, 578, 580, 583, 596, 618, 626, 629]
CELLS = {
    527: [(2, 1), (3, 1)],
    542: [(5, 0)],
    584: [(0, 3), (0, 4), (4, 0), (5, 0), (6, 0)],
    613: [(3, 2)],
    619: [(1, 3), (3, 2), (3, 3), (5, 3)],
}


def main():
    with ZipFile(DOCUMENT) as source:
        original = source.read("word/document.xml").decode("utf-8")
        body = ET.fromstring(original).find(W + "body")
        updated = original
        edits = [(body[index], FIELD_REVISIONS.get(index), REVISIONS.get(index)) for index in INDICES]
        for index, cells in CELLS.items():
            rows = body[index].findall(W + "tr")
            for row, column in cells:
                paragraphs = rows[row].findall(W + "tc")[column].findall(W + "p")
                assert len(paragraphs) == 1
                edits.append((paragraphs[0], None, (TABLE_REVISIONS[index][(row, column)],)))
        for paragraph, targeted, prose in edits:
            paragraph_id = paragraph.get(W14 + "paraId")
            pattern = r'<w:p\b[^>]*w14:paraId="' + paragraph_id + r'"[^>]*>.*?</w:p>'
            matches = list(re.finditer(pattern, updated, flags=re.S))
            assert len(matches) == 1, paragraph_id
            match = matches[0]
            raw = match.group(0)
            revised = replace_text_nodes(raw, targeted) if targeted is not None else revise_paragraph(raw, paragraph, prose)
            updated = updated[:match.start()] + revised + updated[match.end():]
        mask = lambda raw: re.sub(r"(<w:t(?:\s[^>]*)?>).*?(</w:t>)", r"\1\2", raw, flags=re.S)
        assert mask(original) == mask(updated)
        new_body = ET.fromstring(updated).find(W + "body")
        changed = [i for i, (a, b) in enumerate(zip(body, new_body)) if ET.tostring(a) != ET.tostring(b)]
        assert set(changed) <= set(INDICES) | set(CELLS)
        temporary = DOCUMENT.with_suffix(".section-edit.docx")
        with ZipFile(temporary, "w") as target:
            for item in source.infolist():
                target.writestr(item, updated.encode("utf-8") if item.filename == "word/document.xml" else source.read(item.filename))
    temporary.replace(DOCUMENT)
    with ZipFile(BACKUP) as before, ZipFile(DOCUMENT) as after:
        assert before.namelist() == after.namelist()
        assert [name for name in before.namelist() if before.read(name) != after.read(name)] == ["word/document.xml"]
        before_raw = before.read("word/document.xml").decode("utf-8")
        after_raw = after.read("word/document.xml").decode("utf-8")
        assert mask(before_raw) == mask(after_raw)
        before_body = ET.fromstring(before_raw).find(W + "body")
        after_body = ET.fromstring(after_raw).find(W + "body")
        assert len(before_body) == len(after_body)
        for index, (a, b) in enumerate(zip(before_body, after_body)):
            if ET.tostring(a) == ET.tostring(b):
                continue
            assert 515 < index < 632, index
            assert field_results(a) == field_results(b), index
            for tag in (M + "oMath", W + "instrText", W + "fldChar", W + "hyperlink"):
                assert [ET.tostring(n) for n in a.iter(tag)] == [ET.tostring(n) for n in b.iter(tag)]
        for index in [532, 546, 553, 560, 580, 583, 618, 626]:
            assert [n.text for n in before_body[index].iter(W + "t")][:3] == [n.text for n in after_body[index].iter(W + "t")][:3]
    print("Refined section 2.4 blocks:", changed)
    print("Number bookmark contents restored; native objects and formatting unchanged.")


if __name__ == "__main__":
    main()
