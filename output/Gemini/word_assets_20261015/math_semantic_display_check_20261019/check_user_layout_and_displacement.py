"""Read-only layout provenance and the explicitly requested displacement rename."""
from pathlib import Path
from zipfile import ZipFile
from collections import Counter
from copy import deepcopy
from lxml import etree as E
import importlib.util
import json
import re

A = Path(__file__).resolve().parent
R = A.parents[1]
OLD = A.parent / 'full_math_display_audit_20261019/before_final_field_cache_sync.docx'
SOURCE = A / 'source_user_layout_snapshot.docx'
FINAL = R / '调整完格式的_公式符号复核_20261019.docx'
W = '{http://schemas.openxmlformats.org/wordprocessingml/2006/main}'
M = '{http://schemas.openxmlformats.org/officeDocument/2006/math}'
PID = '{http://schemas.microsoft.com/office/word/2010/wordml}paraId'
spec = importlib.util.spec_from_file_location('math_content', A.parent / 'exact_library_v2_20261016/prepare_math_verification.py')
h = importlib.util.module_from_spec(spec)
spec.loader.exec_module(h)


def load(path):
    with ZipFile(path) as z:
        return E.fromstring(z.read('word/document.xml'))


def sig(n):
    if n is None:
        return None
    return (n.tag, tuple(sorted(n.attrib.items())), n.text or '', n.tail or '', tuple(sig(c) for c in n))


def text(n):
    return ''.join(t.text or '' for t in n.iter() if t.tag in (W + 't', M + 't'))


def label(n):
    row = next((a for a in n.iterancestors() if a.tag == W + 'tr'), None)
    if row is None:
        return None
    val = text(row.findall(W + 'tc')[-1])
    return val[1:-1] if re.fullmatch(r'[（(][1-7A-F]-\d+[a-z]*[）)]', val) else None


def math_style(run):
    out = {}
    for path, key, ns in [(M + 'rPr/' + M + 'sty', 'math_style', M),
                          (M + 'rPr/' + M + 'nor', 'normal_text', M),
                          (W + 'rPr/' + W + 'i', 'direct_italic', W)]:
        e = run.find(path)
        out[key] = e.get(ns + 'val', '1') if e is not None else None
    return out


old, source = load(OLD), load(SOURCE)
op, sp = list(old.iter(W + 'p')), list(source.iter(W + 'p'))
old_by_id, src_by_id = {p.get(PID): p for p in op}, {p.get(PID): p for p in sp}
old_rows, src_rows = h.word_rows(OLD), h.word_rows(SOURCE)
assert [x['label'] for x in old_rows] == [x['label'] for x in src_rows]
splits = []
for a, b in zip(old_rows, src_rows):
    assert h.content(a['maths']) == h.content(b['maths']), b['label']
    if len(a['maths']) != len(b['maths']):
        splits.append({'label': b['label'], 'old_oMath_count': len(a['maths']),
                       'new_oMath_count': len(b['maths']),
                       'new_paragraphs': [{'paraId': p.get(PID), 'text': text(p)}
                                          for p in sp if p.get(PID) not in old_by_id and label(p) == b['label']],
                       'math_content_identical_when_aggregated': True})
split_labels = {x['label'] for x in splits}
unexplained_semantic = []
for key, p in old_by_id.items():
    q = src_by_id[key]
    if h.content(list(p.iter(M + 'oMath'))) != h.content(list(q.iter(M + 'oMath'))) and label(q) not in split_labels:
        unexplained_semantic.append(key)
empty = lambda p: not text(p).strip() and not list(p.iter(W + 'drawing'))
integrals = []
for i, math in enumerate(source.iter(M + 'oMath'), 1):
    has_integral = any((n.find(M + 'naryPr/' + M + 'chr') is None or
                        n.find(M + 'naryPr/' + M + 'chr').get(M + 'val') in '∫∬∭∮∯∰')
                       for n in math.iter(M + 'nary')) or any(c in text(math) for c in '∫∬∭∮∯∰')
    if not has_integral:
        continue
    runs = list(math.iter(M + 'r'))
    ds = []
    for j, run in enumerate(runs, 1):
        if text(run).strip() == 'd':
            role = 'displacement_vector' if (i, j) == (1812, 32) else 'differential_operator'
            ds.append({'run': j, 'role': role, **math_style(run),
                       'runXPath': source.getroottree().getpath(run)})
    integrals.append({'math_index': i, 'label': label(math), 'text': text(math), 'd_runs': ds})
layout_report = {
    'read_only': True, 'baseline': str(OLD), 'user_source_snapshot': str(SOURCE),
    'snapshot_byte_identical_to_current_user_named_final': SOURCE.read_bytes() == (R / '调整完格式的_全公式显示终校_20261019.docx').read_bytes(),
    'old_math_objects': len(list(old.iter(M + 'oMath'))), 'new_math_objects': len(list(source.iter(M + 'oMath'))),
    'old_paragraphs': len(op), 'new_paragraphs': len(sp),
    'all_old_paraIds_retained': set(old_by_id).issubset(src_by_id),
    'original_paragraph_pPr_changes': [key for key, p in old_by_id.items() if sig(p.find(W + 'pPr')) != sig(src_by_id[key].find(W + 'pPr'))],
    'old_empty_paragraphs': sum(empty(p) for p in op), 'new_empty_paragraphs': sum(empty(p) for p in sp),
    'new_empty_paragraphs_added': [p.get(PID) for p in sp if p.get(PID) not in old_by_id and empty(p)],
    'manual_w_br_count_old_new': [len(list(x.iter(W + 'br'))) for x in (old, source)],
    'm_brk_count_old_new': [len(list(x.iter(M + 'brk'))) for x in (old, source)],
    'numbered_formula_slots': len(src_rows), 'all_376_aggregated_math_content_identical': True,
    'user_formula_splits': splits, 'unexplained_inline_math_content_changes': unexplained_semantic,
    'integral_math_objects': integrals,
    'differential_d_count': sum(d['role'] == 'differential_operator' for x in integrals for d in x['d_runs']),
    'differential_d_all_explicit_upright': all(d['math_style'] == 'p' for x in integrals for d in x['d_runs'] if d['role'] == 'differential_operator'),
    'displacement_d_in_integral_kept_default_math_italic': next(d for x in integrals for d in x['d_runs'] if d['role'] == 'displacement_vector')['math_style'] is None,
    'notes': ['空段数量不变；本次增加的是17个带公式的段落，原有段落属性未改。',
              '旧基准在最终REF与目录域缓存同步之前，缓存差异不应误报为用户数学改动。',
              'XML样式审查确认d的用途与正斜体设置；实际字形由根代理原生Word复看。']}
(A / 'user_layout_diff_readonly.json').write_text(json.dumps(layout_report, ensure_ascii=False, indent=2) + '\n')


def normalize_runs(node):
    node = deepcopy(node)
    for parent in reversed(list(node.iter())):
        previous = None
        for child in list(parent):
            if child.tag != M + 'r':
                previous = None
                continue
            props = tuple(sig(e) for e in child if e.tag != M + 't')
            if previous is not None and props == tuple(sig(e) for e in previous if e.tag != M + 't'):
                pts, cts = previous.findall(M + 't'), child.findall(M + 't')
                if len(pts) == len(cts) == 1:
                    pts[0].text = (pts[0].text or '') + (cts[0].text or '')
                    parent.remove(child)
                    continue
            previous = child
    return node


final = load(FINAL)
sm, fm = list(source.iter(M + 'oMath')), list(final.iter(M + 'oMath'))
expected = {1794: 1, 1799: 1, 1811: 1, 1812: 1, 1813: 1, 1814: 1, 1817: 2, 1822: 1, 1823: 1, 1826: 1}
changes, unexpected = [], []
assert len(sm) == len(fm) == 4755
for i, (a, b) in enumerate(zip(sm, fm), 1):
    if sig(a) == sig(b):
        continue
    if i not in expected:
        unexpected.append(i)
        continue
    masked = deepcopy(b)
    delta_runs = [r for r in masked.iter(M + 'r') if text(r) == 'Δ']
    changes.append({'math_index': i, 'label': label(b), 'old_text': text(a), 'new_text': text(b),
                    'inserted_Delta_count': len(delta_runs),
                    'all_inserted_Delta_explicit_upright': all(math_style(r)['math_style'] == 'p' for r in delta_runs)})
    assert len(delta_runs) == expected[i], (i, len(delta_runs))
    for r in delta_runs:
        r.getparent().remove(r)
    assert sig(normalize_runs(a)) == sig(normalize_runs(masked)), i
assert [x['math_index'] for x in changes] == list(expected)
masked_source, masked_final = deepcopy(source), deepcopy(final)
for root in (masked_source, masked_final):
    for math in list(root.iter(M + 'oMath')):
        math.getparent().replace(math, E.Element(M + 'oMath'))
with ZipFile(SOURCE) as za, ZipFile(FINAL) as zb:
    changed_parts = [n for n in za.namelist() if za.read(n) != zb.read(n)]
    same_part_names = set(za.namelist()) == set(zb.namelist())
final_ps = {p.get(PID): p for p in final.iter(W + 'p')}
result = {
    'passed': not unexpected and same_part_names and changed_parts == ['word/document.xml'] and sig(masked_source) == sig(masked_final),
    'read_only': True, 'source': str(SOURCE), 'checked': str(FINAL), 'math_objects': len(fm),
    'changed_parts': changed_parts, 'changed_math_objects': len(changes),
    'inserted_Delta_total': sum(x['inserted_Delta_count'] for x in changes),
    'only_authorized_displacement_d_renames': True, 'all_original_d_characters_and_properties_preserved': True,
    'differential_du_unchanged': sig(list(sm[1811].iter(M + 'r'))[32]) == sig(list(fm[1811].iter(M + 'r'))[34]),
    'nonmath_XML_exactly_unchanged': sig(masked_source) == sig(masked_final),
    'all_paragraph_paraIds_order_and_pPr_unchanged': [p.get(PID) for p in sp] == [p.get(PID) for p in final.iter(W + 'p')] and all(sig(p.find(W + 'pPr')) == sig(final_ps[p.get(PID)].find(W + 'pPr')) for p in sp),
    'all_17_user_split_paragraphs_preserved': all(key in final_ps and sig(p.find(W + 'pPr')) == sig(final_ps[key].find(W + 'pPr')) for key, p in src_by_id.items() if key not in old_by_id),
    'changes': changes, 'unexpected_math_changes': unexpected,
    'notes': ['为插入Δ而拆分相邻同属性run属于容器分段；合并同属性相邻run后，移除11个新增Δ即与源公式逐节点一致。',
              '积分里的微分d及后续u未改；位移d保留原数学斜体，新增Δ为正体。']}
assert result['differential_du_unchanged'] and result['passed']
(A / 'displacement_independent_qa.json').write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n')
print(json.dumps({'source_user_formula_splits': len(splits), 'source_all_376_math_identical': True,
                  'integral_differential_d_count': layout_report['differential_d_count'],
                  'Delta_patch_passed': result['passed'], 'changed_math_objects': len(changes),
                  'inserted_Delta_total': result['inserted_Delta_total']}, ensure_ascii=False))
