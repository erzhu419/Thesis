"""Read-only literal/structural bracket and math proofing inventory."""
from pathlib import Path
from zipfile import ZipFile
from collections import Counter
from lxml import etree as E
import json
import re

A = Path(__file__).resolve().parent
SOURCE = A / 'source_snapshot.docx'
W = '{http://schemas.openxmlformats.org/wordprocessingml/2006/main}'
M = '{http://schemas.openxmlformats.org/officeDocument/2006/math}'
BRACKETS = set('()[]{}（）［］｛｝⌈⌉⌊⌋⟨⟩')
with ZipFile(SOURCE) as z:
    root = E.fromstring(z.read('word/document.xml'))
tree = root.getroottree()

def text(e):
    return ''.join(t.text or '' for t in e.iter() if t.tag in (W+'t', M+'t'))

def on(e, name):
    node = e.find(name) if e is not None else None
    return None if node is None else node.get(W+'val', node.get(M+'val', '1')) not in ('0', 'false', 'off')

def properties(run):
    wp, mp = run.find(W+'rPr'), run.find(M+'rPr')
    sty = mp.find(M+'sty') if mp is not None else None
    scr = mp.find(M+'scr') if mp is not None else None
    return {'m_nor': on(mp, M+'nor'), 'm_sty': sty.get(M+'val') if sty is not None else None,
            'm_scr': scr.get(M+'val') if scr is not None else None,
            'w_i': on(wp, W+'i'), 'w_iCs': on(wp, W+'iCs')}

def context(e):
    row = next((x for x in e.iterancestors() if x.tag == W+'tr'), None)
    p = next((x for x in e.iterancestors() if x.tag == W+'p'), None)
    cells = row.findall(W+'tc') if row is not None else []
    label = text(cells[-1]).strip() if cells else ''
    numbered = bool(re.fullmatch(r'[（(][1-7A-F]-\d+[a-z]*[）)]', label))
    return {'display_label': label[1:-1] if numbered else None,
            'paragraph_excerpt': text(p)[:100] if p is not None else None}

literal = []
for ordinal, math in enumerate(root.iter(M+'oMath'), 1):
    for run in math.iter(M+'r'):
        value = text(run)
        positions = [{'char': c, 'charposition_zero_based': i} for i, c in enumerate(value) if c in BRACKETS]
        if not positions:
            continue
        props = properties(run)
        explicitly_italic = ((props['m_nor'] is True and props['w_i'] is True)
                             or (props['m_nor'] is not True and props['m_sty'] in ('i', 'bi')))
        if explicitly_italic:
            state = 'explicit_italic'
        elif props['m_nor'] is True and props['w_i'] is not True:
            state = 'normal_text_without_direct_italic'
        elif props['m_sty'] in ('p', 'b'):
            state = 'explicit_upright_math_style'
        else:
            state = 'math_default_style_requires_glyph_interpretation'
        literal.append({'math_ordinal': ordinal, 'runXPath': tree.getpath(run), 'text': value,
                        'positions': positions, 'bracket_count': len(positions), 'state': state,
                        'mixed_with_other_characters': any(c not in BRACKETS and not c.isspace() for c in value),
                        **props, **context(run)})

delimiters = []
for delim in root.iter(M+'d'):
    pr = delim.find(M+'dPr')
    boundaries = []
    for field, default in [('begChr', '('), ('endChr', ')')]:
        prop = pr.find(M+field) if pr is not None else None
        value = prop.get(M+'val', '') if prop is not None else default
        if value in BRACKETS:
            boundaries.append({'property': field, 'char': value, 'explicit_property': prop is not None})
    if not boundaries:
        continue
    ctrl = pr.find(M+'ctrlPr/'+W+'rPr') if pr is not None else None
    delimiters.append({'delimiterXPath': tree.getpath(delim), 'boundaries': boundaries,
                       'control_w_i': on(ctrl, W+'i'), 'control_w_iCs': on(ctrl, W+'iCs'), **context(delim)})

proofs = list(root.iter(W+'proofErr'))
inside = [p for p in proofs if any(a.tag == M+'oMath' for a in p.iterancestors())]
active = {}; ranges = []
for node in root.iter():
    if node.tag == W+'proofErr':
        kind = node.get(W+'type', '')
        if kind.endswith('Start'):
            active[kind[:-5]] = {'start': node, 'math_characters': 0, 'ordinary_characters': 0}
        elif kind.endswith('End') and kind[:-3] in active:
            span = active.pop(kind[:-3])
            if span['math_characters']:
                start = span.pop('start')
                ranges.append({**span, 'type': kind[:-3], 'startXPath': tree.getpath(start),
                               'endXPath': tree.getpath(node),
                               'start_inside_math': any(a.tag == M+'oMath' for a in start.iterancestors()),
                               'end_inside_math': any(a.tag == M+'oMath' for a in node.iterancestors())})
    elif node.tag in (W+'t', M+'t'):
        key = 'math_characters' if any(a.tag == M+'oMath' for a in node.iterancestors()) else 'ordinary_characters'
        for span in active.values():
            span[key] += len(node.text or '')

focus = {label: {'literal_runs': [x for x in literal if x['display_label'] == label],
                 'native_delimiters': [x for x in delimiters if x['display_label'] == label]}
         for label in ('C-31', '5-3', '5-9', '5-10')}
summary = {'source': str(SOURCE.resolve()), 'math_objects': len(list(root.iter(M+'oMath'))),
           'literal_bracket_runs': len(literal), 'literal_bracket_characters': sum(x['bracket_count'] for x in literal),
           'literal_character_counts': dict(Counter(p['char'] for x in literal for p in x['positions'])),
           'literal_run_state_counts': dict(Counter(x['state'] for x in literal)),
           'explicit_italic_runs': len([x for x in literal if x['state'] == 'explicit_italic']),
           'explicit_italic_characters': sum(x['bracket_count'] for x in literal if x['state'] == 'explicit_italic'),
           'mixed_runs': sum(x['mixed_with_other_characters'] for x in literal),
           'native_delimiter_objects_with_target_brackets': len(delimiters),
           'native_delimiter_boundaries': sum(len(x['boundaries']) for x in delimiters),
           'native_delimiter_control_explicit_italic': sum(x['control_w_i'] is True for x in delimiters),
           'proofErr_total': len(proofs), 'proofErr_inside_math': len(inside),
           'proof_ranges_touching_math': len(ranges),
           'proof_ranges_with_both_boundary_markers_outside_math': sum(not x['start_inside_math'] and not x['end_inside_math'] for x in ranges),
           'scope_note': 'ASCII < > excluded; m:d properties counted separately from literal m:t. XML italic flags are separate from native glyph appearance.',
           'proofing_note': 'Set noProof within math runs. Existing math-internal proofErr caches can be removed; ranges whose outside markers also cover ordinary text should not be deleted globally.',
           'source_modified': False}
for name, value in [('literal_bracket_runs.json', literal), ('native_delimiter_brackets.json', delimiters),
                    ('focus_brackets.json', focus), ('math_proofing_ranges.json', ranges), ('bracket_inventory_summary.json', summary)]:
    (A/name).write_text(json.dumps(value, ensure_ascii=False, indent=2)+'\n')
print(json.dumps(summary, ensure_ascii=False))
