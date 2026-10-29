"""Compare the finished chapter sequence with the seven supplied Markdown files."""

from collections import Counter
from pathlib import Path
from zipfile import ZipFile
import json
import re

from lxml import etree as ET
import pypandoc

from integrate import ROOT, WORK, OUT, NS, W, style, text, inline_text, normalized_prose


def table_cells(block):
    value = block["c"]
    rows = value[3][1] + [row for body in value[4] for row in body[2] + body[3]] + value[5][1]
    return [["".join(inline_text(paragraph["c"]) for paragraph in cell[4]) for cell in row[1]] for row in rows]


def main():
    with ZipFile(OUT) as package:
        root = ET.fromstring(package.read("word/document.xml"))
        body = root.find("w:body", NS)
    first = next(i for i, node in enumerate(body) if style(node) == "chapter" and text(node).startswith("第1章"))
    last = next(i for i, node in enumerate(body) if style(node) == "bibliographytitle")
    actual = []
    for node in body[first:last]:
        if style(node) == "figure" or node.find("w:pPr/w:sectPr", NS) is not None:
            continue
        if node.tag == W + "tbl":
            if node.xpath('.//w:bookmarkStart[starts-with(@w:name,"eq")]', namespaces=NS):
                actual.append(("equation", re.search(r"\(([1-7]-\d+)\)\s*$", text(node)).group(1)))
            else:
                actual.append(("table", [["".join(normalized_prose(p) for p in cell.findall("w:p", NS)) for cell in row.findall("w:tc", NS)] for row in node.findall("w:tr", NS)]))
        elif node.tag == W + "p":
            actual.append(("paragraph", normalized_prose(node)))
        else:
            raise ValueError(ET.QName(node).localname)
    expected = []
    for path in sorted((ROOT / "markdown").glob("*润色重写稿_20261010.md")):
        ast = json.loads(pypandoc.convert_text(path.read_text(), "json", format="markdown+tex_math_dollars-auto_identifiers-smart"))
        for block in ast["blocks"]:
            if block["t"] == "Table":
                expected.append(("table", table_cells(block)))
            elif block["t"] == "Header":
                expected.append(("paragraph", inline_text(block["c"][2])))
            else:
                items = block["c"]
                if len(items) == 1 and items[0]["t"] == "Math" and items[0]["c"][0]["t"] == "DisplayMath":
                    expected.append(("equation", re.search(r"\\tag\{([1-7]-\d+)\}", items[0]["c"][1]).group(1)))
                else:
                    expected.append(("paragraph", inline_text(items)))
    assert len(actual) == len(expected), (len(actual), len(expected))
    for index, (received, required) in enumerate(zip(actual, expected)):
        assert received == required, (index, received, required)
    names = set(root.xpath(".//w:bookmarkStart/@w:name", namespaces=NS))
    unresolved = [anchor for anchor in root.xpath(".//w:hyperlink/@w:anchor", namespaces=NS) if anchor not in names]
    fields = root.xpath(".//w:instrText/text()", namespaces=NS)
    unresolved += [match.group(1) for value in fields if (match := re.match(r"\s*(?:REF|PAGEREF)\s+(\S+)", value)) and match.group(1) not in names]
    assert not unresolved, unresolved
    # A moved bookmark end must not precede its start after paragraphs are merged.
    positions = {}
    invalid_ranges = []
    for position, item in enumerate(root.iter()):
        if item.tag == W + "bookmarkStart":
            positions[item.get(W + "id")] = position
        elif item.tag == W + "bookmarkEnd" and item.get(W + "id") not in positions:
            invalid_ranges.append(item.get(W + "id"))
    assert not invalid_ranges, invalid_ranges
    result = {"ordered_markdown_elements_exact": len(expected), "element_counts": dict(Counter(item[0] for item in actual)), "unresolved_reference_targets": unresolved, "invalid_bookmark_ranges": invalid_ranges}
    (WORK / "正文逐项一致性核对.json").write_text(json.dumps(result, ensure_ascii=False, indent=2))
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
