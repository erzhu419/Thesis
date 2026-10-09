"""Apply narrowly scoped heading and punctuation repairs to existing OOXML trees.

The caller owns package I/O and preservation of the authoritative Word document.
"""
from copy import deepcopy
import re
from lxml import etree

W = "{http://schemas.openxmlformats.org/wordprocessingml/2006/main}"
M = "{http://schemas.openxmlformats.org/officeDocument/2006/math}"


def text_of(p):
    return "".join(n.text or "" for n in p.iter() if n.tag in (W + "t", M + "t"))


def ensure_child(parent, name):
    node = parent.find(W + name)
    if node is None:
        node = etree.SubElement(parent, W + name)
    return node


def set_terminal(p, old, new):
    nodes = [n for n in p.iter() if n.tag in (W + "t", M + "t") and n.text]
    node = nodes[-1]
    assert node.text.endswith(old), text_of(p)
    node.text = node.text[:-len(old)] + new


def apply_semantics(document, styles):
    """Return exact changes; do not remove or reconstruct paragraphs or objects."""
    changes = []
    paragraphs = list(document.iter(W + "p"))
    h4 = []
    caption_changes = []
    for p in paragraphs:
        ppr = p.find(W + "pPr")
        sty = None if ppr is None else ppr.find(W + "pStyle")
        sid = None if sty is None else sty.get(W + "val")
        text = text_of(p)
        if sid == "h4":
            ensure_child(ppr, "outlineLvl").set(W + "val", "3")
            h4.append(text)
        if sid == "tablecaption" and re.match(r"^表(?:3-1|5-[1-6])\s", text) and text.endswith("。"): 
            set_terminal(p, "。", "")
            caption_changes.append({"before": text, "after": text_of(p), "style": sid})
    assert len(h4) == 53, f"Expected 53 existing h4 headings, found {len(h4)}"
    assert len(caption_changes) == 7, f"Expected seven caption punctuation repairs, found {len(caption_changes)}"
    changes.append({"type": "table_caption_terminal_punctuation", "items": caption_changes})

    for prefix, suffix in [
        ("正则化通过偏好特定策略或函数形状影响学习。", "时，"),
        ("再从第一次样本形成的经验分布独立有放回采样", "由全方差公式，"),
    ]:
        found = [p for p in paragraphs if text_of(p).startswith(prefix) and text_of(p).endswith(suffix)]
        assert len(found) == 1, (prefix, len(found))
        p = found[0]
        before = text_of(p)
        set_terminal(p, "，", "：")
        changes.append({"type": "formula_introduction_terminal_colon", "before": before, "after": text_of(p)})

    # Both displayed equalities end a sentence, followed by a separate explanatory
    # sentence. Only their terminal punctuation changes; all mathematical nodes stay.
    for eq_no in ["2-15", "3-13"]:
        tables = [t for t in document.iter(W + "tbl") if f"({eq_no})" in text_of(t)]
        assert len(tables) == 1, (eq_no, len(tables))
        found = [p for p in tables[0].iter(W + "p") if p.find(".//" + M + "oMath") is not None and text_of(p).endswith(",")]
        assert len(found) == 1, (eq_no, len(found))
        p = found[0]
        before = text_of(p)
        set_terminal(p, ",", ".")
        changes.append({"type": "display_formula_terminal_period", "equation": eq_no, "before": before, "after": text_of(p)})

    toc3 = styles.find(f"{W}style[@{W}styleId='TOC3']")
    toc4 = styles.find(f"{W}style[@{W}styleId='TOC4']")
    assert toc3 is not None and toc4 is not None
    # There is no effective fourth-level TOC example in the target template.
    # Infer one indentation step from the existing second and third levels.
    for name in ["pPr", "rPr"]:
        old = toc4.find(W + name)
        if old is not None:
            toc4.remove(old)
        toc4.append(deepcopy(toc3.find(W + name)))
    ppr = toc4.find(W + "pPr")
    ind = ensure_child(ppr, "ind")
    ind.attrib.clear()
    ind.set(W + "left", "630")
    ind.set(W + "firstLine", "0")
    for n in list(toc4):
        if n.tag in (W + "basedOn", W + "autoRedefine"):
            toc4.remove(n)
    instr_changes = []
    for n in document.iter(W + "instrText"):
        if re.search(r"\bTOC\s", n.text or ""):
            old = n.text
            n.text = re.sub(r'(\\o\s+")1-3(")', r'\g<1>1-4\2', n.text)
            assert n.text != old
            instr_changes.append({"before": old, "after": n.text})
    assert len(instr_changes) == 1
    changes.append({"type": "automatic_toc_level_four", "heading_count": len(h4), "headings": h4,
                    "toc_code_changes": instr_changes, "toc4_rule": "宋体/Times New Roman 12 pt; exact 20 pt; left 31.5 pt; right dotted tab 414.8 pt",
                    "rule_basis": "TOC4 has no effective template example; indentation inferred by continuing TOC2/TOC3 increments"})
    return changes
