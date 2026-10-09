"""Extract the authoritative Word, retaining math structure and reference fields."""
import json
import re
import shutil
from pathlib import Path
from zipfile import ZipFile

from lxml import etree as E

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
NS = {"w": "http://schemas.openxmlformats.org/wordprocessingml/2006/main",
      "m": "http://schemas.openxmlformats.org/officeDocument/2006/math"}


def content(node):
    tag = E.QName(node).localname
    if tag in ("t", "instrText"):
        return node.text or "" if tag == "t" else ""
    children = list(node)
    parts = {E.QName(c).localname: c for c in children}
    if tag in ("sSub", "sSup", "sSubSup"):
        out = content(parts["e"])
        if "sub" in parts:
            out += "_{" + content(parts["sub"]) + "}"
        if "sup" in parts:
            out += "^{" + content(parts["sup"]) + "}"
        return out
    if tag == "f":
        return "(" + content(parts["num"]) + ")/(" + content(parts["den"]) + ")"
    if tag == "acc":
        chars = node.xpath("./m:accPr/m:chr/@m:val", namespaces=NS)
        return (chars[0] if chars else "hat") + "{" + content(parts["e"]) + "}"
    if tag == "bar":
        return "bar{" + content(parts["e"]) + "}"
    if tag == "rad":
        return "sqrt(" + content(parts["e"]) + ")"
    if tag == "d":
        pr = parts.get("dPr")
        def delimiter(name, default):
            child = pr.find("m:" + name, NS) if pr is not None else None
            return child.get("{" + NS["m"] + "}val", default) if child is not None else default
        return delimiter("begChr", "(") + ",".join(content(c) for c in children if E.QName(c).localname == "e") + delimiter("endChr", ")")
    if tag == "nary":
        pr = parts.get("naryPr")
        ch = pr.find("m:chr", NS) if pr is not None else None
        sign = ch.get("{" + NS["m"] + "}val", "∫") if ch is not None else "∫"
        return sign + "_{" + content(parts["sub"]) + "}^{" + content(parts["sup"]) + "}(" + content(parts["e"]) + ")"
    if tag == "tab":
        return "\t"
    if tag == "br":
        return "\n"
    return "".join(content(c) for c in children)


def run():
    baseline = ROOT / "output/排版完成.docx"
    backup = HERE / "排版完成_校审前.docx"
    if not backup.exists():
        shutil.copy2(baseline, backup)
    with ZipFile(baseline) as z:
        doc = E.fromstring(z.read("word/document.xml"))
        body = doc.find("w:body", NS)
        blocks = []
        for i, b in enumerate(body):
            styles = b.xpath("./w:pPr/w:pStyle/@w:val", namespaces=NS)
            row = {"i": i, "type": E.QName(b).localname,
                   "style": styles[0] if styles else "", "text": content(b),
                   "bookmarks": b.xpath(".//w:bookmarkStart/@w:name", namespaces=NS),
                   "fields": b.xpath(".//w:instrText/text()", namespaces=NS)}
            if row["type"] == "tbl":
                row["rows"] = [[content(c) for c in r.findall("w:tc", NS)]
                               for r in b.findall("w:tr", NS)]
            blocks.append(row)
        (HERE / "blocks.json").write_text(json.dumps(blocks, ensure_ascii=False, indent=2))
        chapters = {}
        section = "前置"
        for b in blocks:
            if b["style"] == "chapter":
                section = b["text"].strip()
            if b["style"] == "bibliographytitle":
                section = "参考文献"
            chapters.setdefault(section, []).append(f"[{b['i']} {b['style'] or b['type']}] {b['text']}")
            if "rows" in b:
                chapters[section].extend("  | " + " | ".join(r) for r in b["rows"])
        for name, lines in chapters.items():
            safe = re.sub(r'[\\/:*?"<>|\s]', "_", name)
            (HERE / (safe + ".txt")).write_text("\n".join(lines) + "\n")
        summary = {"blocks": len(blocks), "native_math": len(doc.xpath(".//m:oMath", namespaces=NS)),
                   "chapters": {name: len(lines) for name, lines in chapters.items()}}
        (HERE / "inventory.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2))
        print(json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    run()
