"""Resolve the report's western glyph selection and native math h formatting."""
from copy import deepcopy
import json
import unicodedata
from pathlib import Path
from zipfile import ZipFile, ZIP_DEFLATED
from lxml import etree

NS = {
    "w": "http://schemas.openxmlformats.org/wordprocessingml/2006/main",
    "m": "http://schemas.openxmlformats.org/officeDocument/2006/math",
}
W, M = NS["w"], NS["m"]
SIZE_PARAGRAPH_IDS = {
    "04A2BF31",  # report page 83: omega_h and g_h
    "18A1E436",  # report page 85: tau_h
    "15BCB8CC",  # report page 85: h_f = h_b = h
    "3CC78692",  # report page 85: exponential derivative
    "13582728",  # report page 88: Delta h
    "1C2AC8F1",  # report page 110: (tau_h, w_hold)
}


def _western_problem(c):
    return (ord(c) > 127 and "LATIN" in unicodedata.name(c, "")) or c in "–±"


def _rpr(run):
    pr = run.find(f"{{{W}}}rPr")
    if pr is None:
        pr = etree.Element(f"{{{W}}}rPr")
        run.insert(1 if run.find(f"{{{M}}}rPr") is not None else 0, pr)
    return pr


def _font(run, family):
    pr = _rpr(run)
    fonts = pr.find(f"{{{W}}}rFonts")
    if fonts is None:
        fonts = etree.Element(f"{{{W}}}rFonts")
        pr.insert(0, fonts)
    for theme in ("asciiTheme", "hAnsiTheme", "eastAsiaTheme", "cstheme", "csTheme"):
        fonts.attrib.pop(f"{{{W}}}{theme}", None)
    for slot in ("ascii", "hAnsi", "eastAsia", "cs"):
        fonts.set(f"{{{W}}}{slot}", family)
    fonts.set(f"{{{W}}}hint", "default")


def _size(run, halfpoints):
    pr = _rpr(run)
    for name in ("sz", "szCs"):
        node = pr.find(f"{{{W}}}{name}")
        if node is None:
            node = etree.SubElement(pr, f"{{{W}}}{name}")
        node.set(f"{{{W}}}val", str(halfpoints))


def apply_report_fonts(document):
    """Mutate only real problem glyphs; preserve semantic run props and OMML trees.

    Bibliography western glyphs are handled by the bibliography change. Native
    math h uses Cambria Math; only the six marked body paragraphs receive an
    explicit 12-point parent size. Subscript/superscript nodes remain intact.
    """
    changes = {"western": [], "native_math_h": []}
    for p_index, p in enumerate(document.findall(".//w:p", NS)):
        style = p.xpath("string(./w:pPr/w:pStyle/@w:val)", namespaces=NS)
        para_id = p.get("{http://schemas.microsoft.com/office/word/2010/wordml}paraId")
        if style != "bibliography":
            for r_index, run in enumerate(list(p.findall(".//w:r", NS))):
                text_nodes = run.findall("w:t", NS)
                text = "".join(n.text or "" for n in text_nodes)
                bad = [i for i, c in enumerate(text) if _western_problem(c)]
                if not bad:
                    continue
                assert len(text_nodes) == 1
                assert all(n.tag in (f"{{{W}}}rPr", f"{{{W}}}t") for n in run)
                # Western-only strings (author names and numerical table cells)
                # can retain their existing run. Mixed Chinese text is split at
                # the glyph, with every original semantic rPr property copied.
                if all(ord(c) < 128 or _western_problem(c) for c in text):
                    _font(run, "Times New Roman")
                else:
                    parent = run.getparent()
                    insert_at = parent.index(run)
                    pieces = []
                    start = 0
                    for char_index in bad:
                        if start < char_index:
                            pieces.append((text[start:char_index], False))
                        pieces.append((text[char_index], True))
                        start = char_index + 1
                    if start < len(text):
                        pieces.append((text[start:], False))
                    for offset, (piece, change_font) in enumerate(pieces):
                        new_run = deepcopy(run)
                        node = new_run.find("w:t", NS)
                        node.text = piece
                        if piece[:1].isspace() or piece[-1:].isspace():
                            node.set("{http://www.w3.org/XML/1998/namespace}space", "preserve")
                        if change_font:
                            _font(new_run, "Times New Roman")
                        parent.insert(insert_at + offset, new_run)
                    parent.remove(run)
                changes["western"].append({
                    "paragraph_index": p_index, "para_id": para_id,
                    "ordinary_run_index_before_patch": r_index,
                    "run_text_before": text, "character_offsets": bad,
                })
        for r_index, run in enumerate(p.findall(".//m:r", NS)):
            text = "".join(run.xpath("./m:t/text()", namespaces=NS))
            if text not in ("h", "ℎ"):
                continue
            _font(run, "Cambria Math")
            explicit_size = para_id in SIZE_PARAGRAPH_IDS
            if explicit_size:
                _size(run, 24)
            changes["native_math_h"].append({
                "paragraph_index": p_index, "para_id": para_id,
                "math_run_index": r_index, "text": text,
                "explicit_parent_size_points": 12 if explicit_size else None,
            })
    return changes


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("source")
    ap.add_argument("output")
    ap.add_argument("--log", required=True)
    args = ap.parse_args()
    with ZipFile(args.source) as source:
        doc = etree.fromstring(source.read("word/document.xml"))
        changes = apply_report_fonts(doc)
        with ZipFile(args.output, "w", ZIP_DEFLATED) as target:
            for item in source.infolist():
                data = etree.tostring(doc, encoding="UTF-8", xml_declaration=True, standalone=True) if item.filename == "word/document.xml" else source.read(item.filename)
                target.writestr(item, data)
    Path(args.log).write_text(json.dumps(changes, ensure_ascii=False, indent=2), encoding="utf-8")
