"""Read-only final regression after the two authorized REF/TOC cache updates."""
from pathlib import Path
from zipfile import ZipFile
from copy import deepcopy
from lxml import etree as E
import importlib.util, json, re

A = Path(__file__).resolve().parent
R = A.parents[1]
BASE = R / '调整完格式的_全公式显示校订_20261019.docx'
FINAL = R / '调整完格式的_全公式显示终校_20261019.docx'
W = '{http://schemas.openxmlformats.org/wordprocessingml/2006/main}'
M = '{http://schemas.openxmlformats.org/officeDocument/2006/math}'
NS = {'w': W[1:-1], 'm': M[1:-1]}
spec = importlib.util.spec_from_file_location('math_content', A.parent / 'exact_library_v2_20261016/prepare_math_verification.py')
h = importlib.util.module_from_spec(spec)
spec.loader.exec_module(h)


def sig(n):
    return (n.tag, tuple(sorted(n.attrib.items())), n.text or '', tuple(sig(c) for c in n))


def mtext(n):
    return ''.join(t.text or '' for t in n.iter(M + 't'))


def mask_authorized(root, native, final):
    root = deepcopy(root)
    refs = {i['bookmark']: i for i in native['ref_updates']}
    seen_refs = []
    for p in root.iter(W + 'p'):
        stack = []
        for n in p.iter():
            if n.tag == W + 'fldChar':
                kind = n.get(W + 'fldCharType')
                if kind == 'begin': stack.append({'bookmark': None, 'result': False, 'cache': ''})
                elif kind == 'separate' and stack: stack[-1]['result'] = True
                elif kind == 'end' and stack:
                    field = stack.pop()
                    if field['bookmark']:
                        expected_result = refs[field['bookmark']]['new_result' if final else 'old_result']
                        assert field['cache'] == expected_result, (field['bookmark'], field['cache'], expected_result)
            elif n.tag == W + 'instrText' and stack:
                match = re.search(r'\bREF\s+(eqA_00[45])\b', n.text or '')
                if match:
                    key = match[1]
                    expect = refs[key]['new_code' if final else 'old_code']
                    assert (n.text or '').strip() == expect.strip(), (key, n.text, expect)
                    stack[-1]['bookmark'] = key
                    seen_refs.append(key)
                    n.text = 'AUTHORIZED_REF_CODE'
            elif n.tag == W + 't' and stack and stack[-1]['result'] and stack[-1]['bookmark']:
                stack[-1]['cache'] += n.text or ''
                n.text = ''
    assert sorted(seen_refs) == sorted(refs), seen_refs
    pages = {i['bookmark'].lower(): i['page'] for i in native['toc_pages']}
    seen_toc = []
    for link in root.iter(W + 'hyperlink'):
        key = (link.get(W + 'anchor') or '').lower()
        if key not in pages: continue
        texts = list(link.iter(W + 't'))
        assert texts and re.fullmatch(r'\d+', texts[-1].text or ''), (key, [t.text for t in texts])
        if final: assert texts[-1].text == pages[key], (key, texts[-1].text, pages[key])
        texts[-1].text = 'AUTHORIZED_TOC_PAGE'
        seen_toc.append(key)
    assert set(seen_toc) == set(pages), (len(seen_toc), len(pages))
    for math in list(root.iter(M + 'oMath')):
        math.getparent().replace(math, E.Element(M + 'oMath'))
    return root, len(seen_toc)


def main():
    native = json.loads((A / 'native_final_field_results.json').read_text(encoding='utf-8-sig'))
    with ZipFile(BASE) as oldzip, ZipFile(FINAL) as newzip:
        assert set(oldzip.namelist()) == set(newzip.namelist())
        changed = [n for n in oldzip.namelist() if oldzip.read(n) != newzip.read(n)]
        assert changed == ['word/document.xml'], changed
        old = E.fromstring(oldzip.read('word/document.xml'))
        new = E.fromstring(newzip.read('word/document.xml'))
        media_count = sum(n.startswith('word/media/') for n in oldzip.namelist())
    om, nm = list(old.iter(M + 'oMath')), list(new.iter(M + 'oMath'))
    assert len(om) == len(nm) == 4738
    text_diffs, semantic_diffs, xml_diffs = [], [], []
    for i, (a, b) in enumerate(zip(om, nm), 1):
        if mtext(a) != mtext(b): text_diffs.append(i)
        if h.content([a]) != h.content([b]): semantic_diffs.append(i)
        if sig(a) != sig(b): xml_diffs.append(i)
    assert not semantic_diffs, semantic_diffs
    assert text_diffs == [3992] and mtext(om[3991]).replace('&', '') == mtext(nm[3991]), text_diffs
    expected_layout = [1357, 1361, 1365, 1369, 1373, 1377, 1762, 3992, 3999]
    assert xml_diffs == expected_layout, xml_diffs
    a, toc_count = mask_authorized(old, native, False)
    b, _ = mask_authorized(new, native, True)
    assert sig(a) == sig(b), 'Nonmath XML differs beyond the two REF updates and TOC page caches'
    runs = list(new.iter(M + 'r'))
    ctrls = list(new.iter(M + 'ctrlPr'))
    def proof(node):
        p = node.find(W + 'rPr/' + W + 'noProof')
        return p is not None and p.get(W + 'val', '1') in ('1', 'true', 'on')
    assert all(proof(n) for n in runs + ctrls), 'Math noProof missing'
    bookmarks = {n.get(W + 'name').lower() for n in new.iter(W + 'bookmarkStart')}
    unresolved = []
    for n in new.iter(W + 'hyperlink'):
        target = n.get(W + 'anchor')
        if target and target.lower() not in bookmarks: unresolved.append(target)
    for n in new.iter(W + 'instrText'):
        match = re.search(r'\b(?:REF|PAGEREF)\s+(\S+)', n.text or '')
        if match and match[1].lower() not in bookmarks: unresolved.append(match[1])
    assert not unresolved, unresolved
    report = {'passed': True, 'read_only': True, 'baseline': str(BASE), 'final': str(FINAL),
              'math_objects': len(nm), 'math_semantics_all_identical': True,
              'visible_math_characters_unchanged': True,
              'only_layout_alignment_markers_removed_in': 'M003992 (B-53)',
              'only_math_layout_changes': expected_layout,
              'nonmath_XML_exact_except_authorized_REF_and_TOC_caches': True,
              'authorized_REF_updates': native['ref_updates'], 'TOC_page_caches_verified': toc_count,
              'unchanged_media_parts': media_count, 'other_ZIP_parts_identical': True,
              'all_math_noProof': True, 'internal_targets_valid': True, 'actual_errors': []}
    (A / 'final_independent_qa.json').write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps({k: report[k] for k in ('passed', 'math_objects', 'TOC_page_caches_verified', 'unchanged_media_parts', 'actual_errors')}, ensure_ascii=False))


if __name__ == '__main__':
    main()
