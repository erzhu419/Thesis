"""Correct only the 43 mathematical glyphs still exported using MS Mincho."""
import json
from pathlib import Path
from fix_report_fonts import NS, _font


def apply_native_letterlike_fonts(document):
    targets = json.loads(Path(__file__).with_name("native_letterlike_exact_targets.json").read_text(encoding="utf-8"))
    paragraphs = {
        p.get("{http://schemas.microsoft.com/office/word/2010/wordml}paraId"): p
        for p in document.findall(".//w:p", NS)
    }
    seen = set()
    changes = []
    for target in targets:
        key = target["paraId"], target["math_run_index_0based"]
        if key in seen:
            continue
        seen.add(key)
        paragraph = paragraphs[key[0]]
        run = paragraph.findall(".//m:r", NS)[key[1]]
        text = run.xpath("string(./m:t)", namespaces=NS)
        assert text == target["run_text"], (key, text, target["run_text"])
        # Font family selection only. Native m:scr, m:sty, size, language,
        # operators, superscripts and subscripts remain exactly as saved.
        _font(run, "Cambria Math")
        changes.append({
            "para_id": key[0], "math_run_index_0based": key[1],
            "run_text": text,
            "glyphs": [x for x in targets if (x["paraId"], x["math_run_index_0based"]) == key],
        })
    assert len(changes) == 42
    assert sum(len(c["glyphs"]) for c in changes) == 43
    return changes
