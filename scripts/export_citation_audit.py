"""Export the current bibliography-to-body correspondence for editorial review."""

import json
import re
import csv
from pathlib import Path

from docx import Document
from docx.oxml.ns import qn

ROOT = Path(__file__).resolve().parents[1]
DEST = ROOT / "workpapers/manuscript_audit"


def main():
    inventory = json.loads((DEST / "inventory.json").read_text())
    checked = json.loads((DEST / "reference_verification.json").read_text())
    references = {r["id"]: r for r in json.loads((ROOT / "manuscript/references.json").read_text())}
    doc = Document(ROOT / "output/博士论文_正文与数学附录_阶段稿.docx")
    notes_path = DEST / "引文语义逐项判断_20260929.tsv"
    notes = {int(row["编号"]): row["判断"] for row in
             csv.DictReader(notes_path.open(), delimiter="\t")} if notes_path.exists() else {}
    ordered = []
    for bookmark in doc.element.xpath(".//w:bookmarkStart"):
        name = bookmark.get(qn("w:name"))
        if name.startswith("ref-"):
            ordered.append(name[4:])
    assert set(ordered) == set(references)
    lines = ["# 参考文献逐条落点对照（2026-09-29）", "",
             "按本次Word参考文献编号排列，逐条列出核对判断、实际引用章节、引用句群和原始来源。一般方法概述核对作者摘要，具体公式另查相关正文；第38、43项仅作研究对象概述，未获得全文。此表不表示通读了154篇全文。", ""]
    for number, key in enumerate(ordered, 1):
        ref, record = references[key], checked[key]
        url = ref.get("URL") or ("https://doi.org/"+ref["DOI"] if ref.get("DOI") else "")
        title = f"[{ref['title']}]({url})" if url else ref["title"]
        uses = inventory["citations"][key]
        sections = list(dict.fromkeys(u["section"] for u in uses))
        lines.extend([f"## [{number}] {title}", "", "正文位置：" + "；".join(sections) + "。", ""])
        if ref.get("archive") == "arXiv":
            basis = "arXiv题名、作者、年份及摘要；采用该预印本版本。"
        elif key == "liu2025buspriority":
            basis = "所提供学位论文的封面及场景说明；采用封面正式题名。"
        elif key in ("yu2024hierarchical", "xu2025systematic"):
            basis = "作者机构成果页及DOI书目；本轮未获得全文，正文只概述研究对象，不据此声称具体机制或定量效果。"
        elif key in ("zhao2021overlapping", "li2023crowdholding", "he2019transferholding"):
            basis = "期刊官方记录；中文题名、作者、年卷期与页码按原记录保留。"
        else:
            basis = "正式出版或作者机构来源及DOI登记；保留原文优先于登记差异的著录。"
        lines.extend(["核验依据："+basis, ""])
        if number in notes:
            lines.extend(["逐项判断："+notes[number], ""])
        seen = set()
        for use in uses:
            for match in re.finditer(r"\[@[^\]]+\]", use["text"]):
                keys = re.findall(r"@([A-Za-z0-9_-]+)", match[0])
                if key not in keys:
                    continue
                before = use["text"][:match.start()]
                preceding = list(re.finditer(r"\[@[^\]]+\]", before))
                excerpt = (use["text"] if use["text"].lstrip().startswith("|") else
                           before[preceding[-1].end():].strip() if preceding else before.strip())
                if excerpt and excerpt not in seen:
                    lines.extend(["> " + excerpt, ""])
                    seen.add(excerpt)
        if record.get("same_work_verified"):
            note = ("预印本与正式题名存在措辞差别，已另核对同篇关系，不以字面近似直接替换。"
                    if ref.get("archive") == "arXiv" else
                    "登记题名省略副题，已按出版方完整题名补全。")
            lines.extend(["版本说明：" + note, ""])
    (DEST / "参考文献逐条落点对照_20260929.md").write_text("\n".join(lines))
    print(f"Exported {len(ordered)} bibliography entries and their actual citation passages")


if __name__ == "__main__":
    main()
