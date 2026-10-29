"""Repair bibliography layout without changing bibliographic metadata or links."""

from copy import deepcopy
from pathlib import Path
from zipfile import ZipFile
import json
import re
import shutil
import sys

from lxml import etree as E

ROOT = Path(__file__).resolve().parents[2]
WORK = Path(__file__).resolve().parent
SOURCE = ROOT / 'output/排版完成_摘要与附录终校_20261011.docx'
OUT = ROOT / 'output/排版完成_参考文献排版修复_20261011.docx'
sys.path.insert(0, str(ROOT / 'workpapers/七章重写稿整合_20261010'))
from integrate import body_spans

W = '{http://schemas.openxmlformats.org/wordprocessingml/2006/main}'
ZWS = '\u200b'


def text(p):
    return ''.join(t.text or '' for t in p.iter(W + 't'))


def canonical_text(p):
    return re.sub(r'^(\[\d+\])\s*', r'\1', text(p).replace(ZWS, ''))


def url_breaks(value):
    # Word can otherwise move an entire URL to the next line. Keep its target intact.
    match = re.match(r'(https?://)(.*)', value)
    if not match:
        return value
    if match[2].startswith('arxiv.org/'):
        return match[1] + 'arxiv.org/' + ZWS + match[2][len('arxiv.org/'):]
    return match[1] + ZWS + re.sub(r'/(?=.)', '/' + ZWS, match[2])


def main():
    shutil.copy2(SOURCE, WORK / '修复前_摘要与附录终校.docx')
    with ZipFile(SOURCE) as package:
        raw = package.read('word/document.xml')
        body = E.fromstring(raw).find(W + 'body')
        spans = body_spans(raw)
        heading = [i for i, p in enumerate(body) if text(p).strip() == '参考文献'][-1]
        indices = []
        for i in range(heading + 1, len(body)):
            if not re.match(r'^\[\d+\]', text(body[i])):
                break
            indices.append(i)
        assert len(indices) == 214
        replacements = []
        url_count = 0
        for i in indices:
            p = deepcopy(body[i])
            pp = p.find(W + 'pPr')
            ind = pp.find(W + 'ind')
            ind.attrib.clear()
            ind.attrib.update({W + 'left': '600', W + 'hangingChars': '250', W + 'hanging': '600'})
            pp.find(W + 'jc').set(W + 'val', 'both')
            tabs = pp.find(W + 'tabs')
            if tabs is not None:
                pp.remove(tabs)
            tabs = E.Element(W + 'tabs')
            E.SubElement(tabs, W + 'tab', {W + 'val': 'left', W + 'pos': '600'})
            pp.insert(list(pp).index(ind), tabs)
            first = next(p.iter(W + 't'))
            first.text = re.sub(r'^(\[\d+\]) +$', r'\1', first.text or '')
            for hyperlink in p.iter(W + 'hyperlink'):
                for t in hyperlink.iter(W + 't'):
                    updated = url_breaks(t.text or '')
                    url_count += updated != t.text
                    t.text = updated
            assert canonical_text(p) == canonical_text(body[i]), i
            start, end = spans[i]
            replacements.append((start, end, E.tostring(p, encoding='utf-8', with_tail=False)))
        updated = raw
        for start, end, replacement in reversed(replacements):
            updated = updated[:start] + replacement + updated[end:]
        parts = [(deepcopy(info), updated if info.filename == 'word/document.xml'
                  else package.read(info.filename)) for info in package.infolist()]
    with ZipFile(OUT, 'w') as package:
        for info, data in parts:
            package.writestr(info, data)
    with ZipFile(OUT) as package:
        repaired = package.read('word/document.xml')
        new_spans = body_spans(repaired)
        assert len(new_spans) == len(spans)
        for i, ((start, end), (a, b)) in enumerate(zip(spans, new_spans)):
            if i not in indices:
                assert raw[start:end] == repaired[a:b], i
        new_body = E.fromstring(repaired).find(W + 'body')
        for i in indices:
            assert canonical_text(body[i]) == canonical_text(new_body[i]), i
            assert [h.attrib for h in body[i].iter(W + 'hyperlink')] == [h.attrib for h in new_body[i].iter(W + 'hyperlink')]
    result = {'references': len(indices), 'references_with_url_break_points': url_count,
              'bibliography_indices': [indices[0], indices[-1]],
              'other_document_blocks_unchanged': True,
              'metadata_order_and_hyperlink_targets_preserved': True,
              'alignment': 'both', 'left_indent_pt': 30, 'hanging_indent_pt': 30,
              'label_tab_pt': 30, 'font_and_line_spacing_unchanged': True,
              'url_display_only': 'arXiv breaks only after domain; abs/identifier kept intact; no forced newlines.'}
    (WORK / '修复与内容核对.json').write_text(json.dumps(result, ensure_ascii=False, indent=2))
    print(json.dumps(result, ensure_ascii=False))


if __name__ == '__main__':
    main()
