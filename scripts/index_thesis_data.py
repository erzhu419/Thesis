"""Read the chosen Word baseline and index its figures and result tables."""

from collections import defaultdict
import json
from pathlib import Path
import re
from xml.etree import ElementTree as ET
from zipfile import ZipFile

ROOT = Path(__file__).resolve().parents[1]
BASELINE = ROOT / "output/排版完成_参考文献复核_20261007_导师批改.docx"
OUT = ROOT / "workpapers/论文数据对照_20261009"
NS = {
    "w": "http://schemas.openxmlformats.org/wordprocessingml/2006/main",
    "m": "http://schemas.openxmlformats.org/officeDocument/2006/math",
    "a": "http://schemas.openxmlformats.org/drawingml/2006/main",
    "r": "http://schemas.openxmlformats.org/officeDocument/2006/relationships",
    "v": "urn:schemas-microsoft-com:vml",
    "o": "urn:schemas-microsoft-com:office:office",
}
TEXT_TAGS = {"{" + NS[key] + "}t" for key in ["w", "m"]}


def text(element):
    return "".join(node.text or "" for node in element.iter() if node.tag in TEXT_TAGS)


def image_relations(block):
    return [node.get("{" + NS["r"] + "}embed") for node in block.findall(".//a:blip", NS)] + [
        node.get("{" + NS["r"] + "}id") for node in block.findall(".//v:imagedata", NS)]


def main():
    before = BASELINE.stat()
    sources = json.loads((ROOT / "manuscript/figures/figure_sources.json").read_text())
    source_by_name = {row["figure"]: row for row in sources}
    pictures = defaultdict(list)
    for folder in ["manuscript/figures", "output/figure_redesign", "output/imagegen", "workpapers"]:
        for path in (ROOT / folder).rglob("*"):
            if path.is_file() and path.suffix.lower() in [".png", ".jpg", ".jpeg"]:
                pictures[path.stat().st_size].append(path)
    with ZipFile(BASELINE) as archive:
        document = ET.fromstring(archive.read("word/document.xml"))
        body = list(document.find("w:body", NS))
        relations = {node.get("Id"): node.get("Target") for node in
                     ET.fromstring(archive.read("word/_rels/document.xml.rels"))}
        matches = {}
        for media in {relations[rid] for block in body for rid in image_relations(block)
                      if rid in relations}:
            data = archive.read("word/" + media)
            matches[media] = [str(path.relative_to(ROOT)) for path in pictures[len(data)]
                              if path.read_bytes() == data]
        start = next(i for i, block in enumerate(body)
                     if text(block).strip() == "第1章 绪论")
        section, chapter = "", ""
        headings, figures, tables = [], [], []
        for index, block in enumerate(body):
            if index < start:
                continue
            value = text(block).strip()
            if block.tag == "{" + NS["w"] + "}p":
                if re.match(r"^第[1-7]章\s", value) or re.match(r"^附录[A-F]\s", value):
                    chapter = section = value
                    headings.append(dict(block=index, title=value))
                elif re.match(r"^[1-7](?:\.\d+){1,2}\s", value) or re.match(r"^[A-F]\.\d+\s", value):
                    section = value
                    headings.append(dict(block=index, title=value))
                caption = re.fullmatch(r"(图|表)([1-7A-F])-([0-9]+)\s+(.+)", value)
                if not caption:
                    continue
                item = dict(number=caption[1] + caption[2] + "-" + caption[3],
                            title=caption[4], block=index, chapter=chapter, section=section)
                if caption[1] == "图":
                    picture = None
                    for previous in reversed(body[max(start, index - 3):index]):
                        if re.match(r"^(图|表)[1-7A-F]-\d+\s", text(previous).strip()):
                            break
                        if image_relations(previous):
                            picture = previous
                            break
                    media = [] if picture is None else list(dict.fromkeys(
                        relations[rid] for rid in image_relations(picture) if rid in relations))
                    item["embedded_media"] = media
                    item["exact_image_matches"] = list(dict.fromkeys(p for key in media for p in matches[key]))
                    item["image_metadata_paths"] = [] if picture is None else [
                        node.get("descr") for node in picture.iter()
                        if node.tag.endswith("}cNvPr") and node.get("descr", "").startswith(str(ROOT))]
                    item["figure_source_records"] = [source_by_name[name] for name in
                        dict.fromkeys(Path(p).stem for p in
                                      item["exact_image_matches"] + item["image_metadata_paths"])
                        if name in source_by_name]
                    item["editable_diagrams"] = []
                    if picture is not None:
                        for node in picture.findall(".//o:OLEObject", NS):
                            target = relations[node.get("{" + NS["r"] + "}id")]
                            if target.endswith(".vsdx"):
                                destination = OUT / "diagram_sources" / (item["number"] + ".vsdx")
                                destination.parent.mkdir(parents=True, exist_ok=True)
                                destination.write_bytes(archive.read("word/" + target))
                                item["editable_diagrams"].append(dict(
                                    embedded=target, local=str(destination.relative_to(ROOT)),
                                    application=node.get("ProgID")))
                    figures.append(item)
                else:
                    candidate = next((b for b in body[index + 1:index + 4]
                                      if b.tag == "{" + NS["w"] + "}tbl"), None)
                    item["rows"] = [] if candidate is None else [
                        [text(cell) for cell in row.findall("w:tc", NS)]
                        for row in candidate.findall("w:tr", NS)]
                    tables.append(item)
        result = dict(baseline=str(BASELINE.relative_to(ROOT)), bytes=before.st_size,
                      modification_time_ns=before.st_mtime_ns, headings=headings,
                      figures=figures, tables=tables,
                      tracked_insertions=len(document.findall(".//w:ins", NS)),
                      tracked_deletions=len(document.findall(".//w:del", NS)))
    assert (BASELINE.stat().st_size, BASELINE.stat().st_mtime_ns) == (before.st_size, before.st_mtime_ns)
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "正文图表索引.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(dict(figures=len(figures), tables=len(tables), headings=len(headings),
                         figures_with_exact_matches=sum(bool(r["exact_image_matches"]) for r in figures),
                         editable_diagrams=sum(len(r["editable_diagrams"]) for r in figures),
                         baseline_unmodified=True), ensure_ascii=False))


if __name__ == "__main__":
    main()
