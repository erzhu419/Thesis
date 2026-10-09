"""Read the current Word document without rebuilding its source or layout."""

import json
from pathlib import Path
from zipfile import ZipFile

from lxml import etree as E

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
NS = {"w": "http://schemas.openxmlformats.org/wordprocessingml/2006/main",
      "m": "http://schemas.openxmlformats.org/officeDocument/2006/math",
      "a": "http://schemas.openxmlformats.org/drawingml/2006/main"}


def text(element):
    return "".join(element.xpath(".//w:t/text() | .//m:t/text()", namespaces=NS))


def inspect():
    with ZipFile(ROOT / "output/排版完成.docx") as archive:
        document = E.fromstring(archive.read("word/document.xml"))
        body = document.find("w:body", NS)
        blocks = []
        for index, element in enumerate(body):
            styles = element.xpath("./w:pPr/w:pStyle/@w:val", namespaces=NS)
            row = dict(index=index, type=E.QName(element).localname,
                style=styles[0] if styles else "", text=text(element),
                bookmarks=element.xpath(".//w:bookmarkStart/@w:name", namespaces=NS))
            if row["type"] == "tbl":
                row["rows"] = [[text(cell) for cell in line.findall("w:tc", NS)]
                    for line in element.findall("w:tr", NS)]
            blocks.append(row)
        (HERE / "word_blocks.json").write_text(json.dumps(blocks, ensure_ascii=False, indent=2)+"\n")
        lines = []
        for row in blocks:
            lines.append(f"[{row['index']} {row['style'] or row['type']}] {row['text']}")
            for cells in row.get("rows", []):
                lines.append("  | "+" | ".join(cells))
        (HERE / "word_body.txt").write_text("\n".join(lines)+"\n")
        print(dict(blocks=len(blocks), equations=len(document.xpath(".//m:oMath", namespaces=NS)),
            images=len(document.xpath(".//a:blip", namespaces=NS))))


if __name__ == "__main__":
    inspect()
