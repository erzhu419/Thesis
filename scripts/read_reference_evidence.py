"""Show each cited passage beside its retrieved primary-source abstract."""

import argparse
import json
import re
from pathlib import Path
from docx import Document
from docx.oxml.ns import qn

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("start", type=int)
    parser.add_argument("stop", type=int)
    args = parser.parse_args()
    refs = {r["id"]: r for r in json.loads((ROOT / "manuscript/references.json").read_text())}
    evidence = json.loads((ROOT / "workpapers/manuscript_audit/reference_verification.json").read_text())
    uses = json.loads((ROOT / "workpapers/manuscript_audit/inventory.json").read_text())["citations"]
    doc = Document(ROOT / "output/博士论文_正文与数学附录_阶段稿.docx")
    ordered = [b.get(qn("w:name"))[4:] for b in doc.element.xpath(".//w:bookmarkStart")
               if b.get(qn("w:name"), "").startswith("ref-")]
    for n in range(args.start, min(args.stop, len(ordered))+1):
        key = ordered[n-1]
        ref, record = refs[key], evidence[key]
        print(f"\n[{n}] {key}: {ref['title']}")
        print("Authors/year:", ref.get("author"), ref.get("issued"))
        print("Primary:", record.get("source"))
        print("Abstract:", re.sub(r"<[^>]+>", "", record.get("abstract", "")))
        seen = set()
        for use in uses[key]:
            for match in re.finditer(r"\[@[^\]]+\]", use["text"]):
                if key not in re.findall(r"@([A-Za-z0-9_-]+)", match[0]):
                    continue
                before = use["text"][:match.start()]
                preceding = list(re.finditer(r"\[@[^\]]+\]", before))
                excerpt = before[preceding[-1].end():].strip() if preceding else before.strip()
                if excerpt not in seen:
                    print("Use", use["section"]+":", excerpt)
                    seen.add(excerpt)


if __name__ == "__main__":
    main()
