"""Check the numbered, Word-rendered thesis rather than source labels alone."""

import json
import re
import sys
from collections import Counter
from pathlib import Path
from zipfile import ZipFile

import fitz
from lxml import etree

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "output"
W = "http://schemas.openxmlformats.org/wordprocessingml/2006/main"


def main():
    pdf_path = Path(sys.argv[1]) if len(sys.argv) > 1 else OUTPUT / "博士论文_正文与数学附录_阶段稿_20260929_第二轮评审修订.pdf"
    pdf = fitz.open(pdf_path)
    with ZipFile(OUTPUT / "博士论文_正文与数学附录_阶段稿.docx") as archive:
        tree = etree.fromstring(archive.read("word/document.xml"))
    stack, instructions = [], []
    for node in tree.iter():
        if node.tag == f"{{{W}}}fldSimple":
            instructions.append(node.get(f"{{{W}}}instr"))
        elif node.tag == f"{{{W}}}fldChar":
            kind = node.get(f"{{{W}}}fldCharType")
            if kind == "begin":
                stack.append(dict(parts=[], result=False))
            elif kind == "separate":
                stack[-1]["result"] = True
            elif kind == "end":
                instructions.append("".join(stack.pop()["parts"]))
        elif node.tag == f"{{{W}}}instrText" and stack and not stack[-1]["result"]:
            stack[-1]["parts"].append(node.text or "")
    assert not stack, "Unclosed Word field"
    names = tree.xpath("//w:bookmarkStart/@w:name", namespaces={"w": W})
    assert len(names) == len(set(names)), "Duplicate bookmarks"
    refs = [i.split()[1] for i in instructions if i.strip().startswith("REF ")]
    assert not set(refs)-set(names), "Undefined Word REF target"
    sequences = [i for i in instructions if i.strip().startswith("SEQ ")]
    expected = [e["number"] for e in json.loads((OUTPUT / "equation_index.json").read_text())]
    assert len(sequences) == len(expected)
    actual, overlap = [], []
    for page in pdf:
        text = page.get_text()
        assert not re.search(r"错误[!！]|Error!|Reference source not found|Bookmark not defined", text), page.number+1
        if "关键词：" in text:
            assert len(re.sub(r"\s", "", text)) > 180, "Abstract keywords on an otherwise empty page"
        words = page.get_text("words")
        for x0, y0, x1, y1, value, *_ in words:
            if x0 > 460 and re.fullmatch(r"\([2-6A-E]-\d+\)", value):
                actual.append(value[1:-1])
                collisions = [w[4] for w in words if w[0] < x0-1 and w[2] > x0-3
                              and min(y1, w[3])-max(y0, w[1]) > 3]
                if collisions:
                    overlap.append(dict(page=page.number+1, equation=value, words=collisions))
    assert actual == expected, "Missing, duplicate or reordered rendered equation numbers"
    assert not overlap, overlap
    # Every bibliography number must remain present after field update and export.
    bibliography = []
    for p in tree.xpath("//w:p[w:bookmarkStart[starts-with(@w:name,'ref-')]]", namespaces={"w": W}):
        text = "".join(p.xpath(".//w:t/text()", namespaces={"w": W}))
        bibliography.append(int(re.match(r"\[(\d+)\]", text)[1]))
    build = json.loads((OUTPUT / "word_build.json").read_text())
    assert bibliography == list(range(1, build["cited_references"] + 1)), bibliography
    report = dict(pdf_pages=len(pdf), equations=len(actual), numbered_equation_fields=len(sequences),
                  equation_counts=dict(Counter(n.split("-")[0] for n in actual)),
                  body_cross_reference_fields=len(refs),
                  cross_reference_targets=dict(Counter(re.match(r"[a-z]+", n)[0] for n in refs)),
                  reference_entries=len(bibliography),
                  missing_targets=[], field_errors=[], formula_number_overlap_candidates=[])
    (OUTPUT / "thesis_crossrefs_check.json").write_text(json.dumps(report, ensure_ascii=False, indent=2)+"\n")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
