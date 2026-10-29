"""Copy only Word's updated TOC page results, not its document-wide rewrites."""

from copy import deepcopy
from zipfile import ZipFile
import json
import re

from lxml import etree as ET

from integrate import OUT, WORK, NS, style, body_spans


STAGE = "/mnt/c/Users/erzhu/AppData/Local/Temp/thesis_md_merge_20261010/toc_stage.docx"


def main():
    with ZipFile(STAGE) as stage:
        saved = ET.fromstring(stage.read("word/document.xml")).find("w:body", NS)
        pages = [x.xpath(".//w:t/text()", namespaces=NS)[-1] for x in saved if style(x) in ("TOC1", "TOC2", "TOC3")]
    with ZipFile(OUT) as package:
        original = package.read("word/document.xml")
        body = ET.fromstring(original).find("w:body", NS)
        spans = body_spans(original)
        toc_indices = [i for i, x in enumerate(body) if style(x) in ("TOC1", "TOC2", "TOC3")]
        assert len(pages) == len(toc_indices) == 165
        replacements = []
        changed = 0
        for index, page in zip(toc_indices, pages):
            assert page.isdigit(), page
            start, end = spans[index]
            paragraph = original[start:end]
            cache = list(re.finditer(rb"(<w:t(?:\s[^>]*)?>)(.*?)(</w:t>)", paragraph))[-1]
            assert cache.group(2).decode().isdigit()
            changed += cache.group(2).decode() != page
            paragraph = paragraph[:cache.start(2)] + page.encode() + paragraph[cache.end(2):]
            replacements.append((start, end, paragraph))
        updated = original
        for start, end, paragraph in reversed(replacements):
            updated = updated[:start] + paragraph + updated[end:]
        parts = [(deepcopy(info), updated if info.filename == "word/document.xml" else package.read(info.filename)) for info in package.infolist()]
    with ZipFile(OUT, "w") as package:
        for info, data in parts:
            package.writestr(info, data)
    result = {"toc_entries": len(pages), "changed_page_results": changed, "only_existing_toc_page_cache_text_updated": True}
    (WORK / "目录页码更新.json").write_text(json.dumps(result, ensure_ascii=False, indent=2))
    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()
