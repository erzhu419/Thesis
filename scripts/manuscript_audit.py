"""Inventory the actual Word sources for the thesis editorial review."""

import json
import re
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEST = ROOT / "workpapers/manuscript_audit"


def sources():
    base = ROOT / "manuscript"
    return (sorted((base / "frontmatter").glob("*.md"))
            + sorted((base / "chapters").glob("*.md"))
            + sorted(p for p in (base / "appendices").glob("*.md")
                     if p.name[0] in "ABCDE" or p.name == "F_公交世界模型的探索性比较.md"))


def audit_sections(documents):
    headings, mentions, errors = [], [], []
    for file, text in documents:
        root = re.search(r"(?m)^# (?:第([1-7])章|附录([A-F]))", text)
        chapter = (root[1] or root[2]) if root else None
        parents, counters = {chapter} if chapter else set(), Counter()
        for line_number, line in enumerate(text.splitlines(), 1):
            heading = re.match(r"^(#{2,3}) ([1-7A-F](?:\.\d+){1,2})\s+", line)
            if heading:
                label = heading[2]
                parts = label.split(".")
                parent = ".".join(parts[:-1])
                counters[parent] += 1
                expected = f"{parent}.{counters[parent]}"
                item = dict(file=file, line=line_number, label=label)
                headings.append(item)
                if parts[0] != chapter or len(parts) != len(heading[1]):
                    errors.append(dict(**item, error="heading_level_or_chapter"))
                if parent not in parents:
                    errors.append(dict(**item, error="missing_parent"))
                if label != expected:
                    errors.append(dict(**item, error="section_sequence", expected=expected))
                parents.add(label)
            else:
                for group in re.finditer(
                    r"第[1-7A-F](?:\.\d+){1,2}"
                    r"(?:(?:至|及|、|与)(?:第)?[1-7A-F](?:\.\d+){1,2})*节", line
                ):
                    for label in re.findall(r"[1-7A-F](?:\.\d+){1,2}", group[0]):
                        mentions.append(dict(file=file, line=line_number, label=label))
    labels = Counter(h["label"] for h in headings)
    for label, count in labels.items():
        if count > 1:
            errors.append(dict(label=label, error="duplicate_section", count=count))
    errors.extend(dict(**m, error="undefined_section_reference")
                  for m in mentions if m["label"] not in labels)
    return dict(headings=headings, mentions=mentions, errors=errors)


def main():
    DEST.mkdir(exist_ok=True)
    citations = defaultdict(list)
    equations, math_uses, captions, mentions, abbreviations, appendix_mentions = [], [], [], [], [], []
    for path in sources():
        text = path.read_text()
        file = str(path.relative_to(ROOT))
        section = ""
        chapter = path.name[:1] if path.parent.name == "appendices" else str(int(path.name[:2]))
        equation_count = 0
        for match in re.finditer(r"(?ms)^\$\$[ \t]*\n(.*?)^\$\$[ \t]*$", text):
            equation_count += 1
            prefix = text[:match.start()]
            headings = re.findall(r"(?m)^#{1,3} (.+)$", prefix)
            equations.append(dict(file=file, line=prefix.count("\n") + 1,
                                  number=f"{chapter}-{equation_count}",
                                  section=headings[-1] if headings else "", latex=match[1].strip(),
                                  before=prefix[-330:], after=text[match.end():match.end()+330]))
        for match in re.finditer(r"\$\$(.*?)\$\$|(?<!\$)\$(?!\$)(.*?)(?<!\$)\$(?!\$)", text, re.S):
            math_uses.append(dict(file=file, line=text[:match.start()].count("\n")+1,
                                  latex=match[1] if match[1] is not None else match[2]))
        for line_number, line in enumerate(text.splitlines(), 1):
            if line.startswith("#"):
                section = line.lstrip("# ")
            for group in re.findall(r"\[@([^\]]+)\]", line):
                for key in re.findall(r"(?:^|;\s*@)([A-Za-z0-9_-]+)", group):
                    citations[key].append(dict(file=file, line=line_number, section=section, text=line))
            caption = re.match(r"^([图表][1-7A-F]-\d+)\s+(.+)", line)
            if caption:
                captions.append(dict(label=caption[1], title=caption[2], file=file, line=line_number))
            else:
                for label in re.findall(r"[图表][1-7A-F]-\d+", line):
                    mentions.append(dict(label=label, file=file, line=line_number))
            for token in re.findall(r"(?<![A-Za-z])[A-Z][A-Za-z]*(?:[+/-][A-Z]+|[0-9]+)?(?:-[A-Z]+)*(?![A-Za-z])", line):
                if len(token) > 1:
                    abbreviations.append(dict(token=token, file=file, line=line_number, context=line))
            if re.search(r"附录[A-F]|(?:定理|引理|命题|推论)[A-E]\.\d+", line):
                appendix_mentions.append(dict(file=file, line=line_number, text=line))
    refs = json.loads((ROOT / "manuscript/references.json").read_text())
    known = {r["id"] for r in refs}
    labels = Counter(c["label"] for c in captions)
    cited_labels = {m["label"] for m in mentions}
    sections = audit_sections([(str(p.relative_to(ROOT)), p.read_text()) for p in sources()])
    result = dict(citations=dict(citations), equations=equations, math_uses=math_uses,
                  captions=captions, mentions=mentions, abbreviations=abbreviations,
                  appendix_mentions=appendix_mentions, sections=sections)
    (DEST / "inventory.json").write_text(json.dumps(result, ensure_ascii=False, indent=2)+"\n")
    print(json.dumps(dict(references=len(refs), cited=len(citations), missing_refs=sorted(set(citations)-known),
                         uncited=sorted(known-set(citations)), equations=len(equations),
                         captions=len(captions), duplicate_captions={k:v for k,v in labels.items() if v>1},
                         caption_without_mention=sorted(set(labels)-cited_labels),
                         undefined_figure_table_mentions=sorted(cited_labels-set(labels)),
                         sections=len(sections["headings"]), section_references=len(sections["mentions"]),
                         section_errors=sections["errors"]), ensure_ascii=False, indent=2))
    if sections["errors"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
