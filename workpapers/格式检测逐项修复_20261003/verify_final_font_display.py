"""Verify the report's font locations in a newly exported native Word PDF."""
import argparse
import json
from pathlib import Path
from zipfile import ZipFile
import xml.etree.ElementTree as ET
import fitz

NS = {"w": "http://schemas.openxmlformats.org/wordprocessingml/2006/main", "m": "http://schemas.openxmlformats.org/officeDocument/2006/math"}
WESTERN = [(2, "Sánchez-Martínez"), (3, "Hüllermeier"), (34, "Fréchet"), (46, "Kolmogorov–Smirnov"), (99, "均值±标准差")]
MATH = {
    11: ["本节的", "为计划目标间隔", "为单侧计划偏差得分"],
    13: ["在另一截面上", "斜率绝对值"],
    14: ["为时距衰减尺度", "都给出相同的时距评价"],
    16: ["可行停站若超过乘客成本", "便会使服务更均衡"],
    29: ["三种指数配置的", "依次为", "时距均衡与停站动作奖励"],
}


def verify(docx, pdf):
    document = fitz.open(pdf)
    pages = []
    all_h = []
    nonblank_mincho = []
    for page in document:
        spans = [s for b in page.get_text("dict")["blocks"] for l in b.get("lines", []) for s in l["spans"]]
        row = {"page": page.number + 1, "text": "".join(page.get_text().split()), "spans": spans}
        pages.append(row)
        all_h.extend({"page": row["page"], "text": s["text"], "glyph_count": s["text"].count("ℎ"), "font": s["font"], "pdf_size_points": s["size"], "bbox": s["bbox"]} for s in spans if "ℎ" in s["text"])
        nonblank_mincho.extend({"page": row["page"], "text": s["text"], "font": s["font"], "pdf_size_points": s["size"]} for s in spans if "Mincho" in s["font"] and s["text"].strip())
    cases = []
    for issue, text in WESTERN:
        matches = [p for p in pages if text in p["text"]]
        p = matches[0]
        token = "±" if issue == 99 else text
        selected = [s for s in p["spans"] if token in s["text"] and (issue != 99 or abs(s["size"] - 12) < 0.01)]
        ok = bool(selected) and all(s["font"].startswith("TimesNewRoman") and abs(s["size"] - 12) < 0.01 for s in selected)
        cases.append({"issue_id": issue, "final_page": p["page"], "target": text, "expected_font": "Times New Roman", "expected_size_points": 12, "actual_pdf_spans": [{k: s[k] for k in ("text", "font", "size", "bbox")} for s in selected], "passed": ok})
    for issue, anchors in MATH.items():
        p = next(p for p in pages if all(anchor in p["text"] for anchor in anchors))
        selected = [h for h in all_h if h["page"] == p["page"]]
        ok = bool(selected) and all(h["font"] == "CambriaMath" and (abs(h["pdf_size_points"] - 12) < 0.01 or abs(h["pdf_size_points"] - 8.52) < 0.01) for h in selected)
        cases.append({"issue_id": issue, "final_page": p["page"], "paragraph_anchors": anchors, "expected_native_math_font": "Cambria Math", "expected_parent_size_points": 12, "automatic_subscript_pdf_size_points": 8.52, "actual_page_native_h_spans": selected, "passed": ok})
    table_page = next(p for p in pages if "2678.11±40.93" in p["text"])
    tables = [{"page": table_page["page"], "text": s["text"], "font": s["font"], "pdf_size_points": s["size"]} for s in table_page["spans"] if "±" in s["text"] and abs(s["size"] - 10.56) < 0.01]
    source = ET.fromstring(ZipFile(docx).read("word/document.xml"))
    native_h_count = sum((m.findtext("m:t", default="", namespaces=NS) in ("h", "ℎ")) for m in source.findall(".//m:r", NS))
    xml_tables = []
    for p in source.findall(".//w:p", NS):
        t = "".join(n.text or "" for n in p.findall(".//w:t", NS))
        if "±" in t and "均值" not in t:
            sizes = [n.get("{" + NS["w"] + "}val") for n in p.findall(".//w:r/w:rPr/w:sz", NS)]
            xml_tables.append({"text": t, "run_size_halfpoints": sizes, "actual_word_size_points": 10.5, "passed": bool(sizes) and all(v == "21" for v in sizes)})
    appendix = [{"page": p["page"], "text": s["text"], "font": s["font"], "pdf_size_points": s["size"]} for p in pages for s in p["spans"] if "Fréchet" in s["text"]]
    total_h = sum(x["glyph_count"] for x in all_h)
    h_pass = total_h == native_h_count == 206 and all(x["font"] == "CambriaMath" for x in all_h)
    table_pass = len(tables) == len(xml_tables) == 12 and all(x["font"].startswith("TimesNewRoman") for x in tables) and all(x["passed"] for x in xml_tables)
    appendix_pass = len(appendix) == 2 and all(x["font"].startswith("TimesNewRoman") and abs(x["pdf_size_points"] - 12) < 0.01 for x in appendix)
    return {"docx": str(docx), "word_pdf": str(pdf), "pages": len(document), "annotation_cases": sorted(cases, key=lambda c: c["issue_id"]), "all_206_native_h": {"source_count": native_h_count, "rendered_count": total_h, "passed": h_pass, "actual_spans": all_h}, "table_12_plus_minus": {"passed": table_pass, "word_xml": xml_tables, "pdf_spans": tables, "note": "Word run size remains 10.5pt; PDF exported font matrix records 10.56pt for the entire numerical string, including the plus-minus glyph."}, "body_and_appendix_frechet": {"passed": appendix_pass, "spans": appendix}, "remaining_nonblank_MS_Mincho": nonblank_mincho, "all_10_marked_groups_passed": all(c["passed"] for c in cases), "passed": all(c["passed"] for c in cases) and h_pass and table_pass and appendix_pass and not nonblank_mincho, "visual_review_completed": False}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("docx", type=Path)
    parser.add_argument("pdf", type=Path)
    parser.add_argument("out", type=Path)
    args = parser.parse_args()
    result = verify(args.docx, args.pdf)
    args.out.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"passed": result["passed"], "all_10_marked_groups_passed": result["all_10_marked_groups_passed"], "all_206_h_passed": result["all_206_native_h"]["passed"], "table_12_plus_minus_passed": result["table_12_plus_minus"]["passed"], "remaining_nonblank_MS_Mincho": len(result["remaining_nonblank_MS_Mincho"])}, ensure_ascii=False))
