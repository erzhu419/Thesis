"""Check the live manuscript, not an old report or style-name-only snapshot."""
from collections import Counter, defaultdict
import json
from pathlib import Path
import re
import sys
from zipfile import ZipFile

from lxml import etree as E

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / 'workpapers/模板迁移_20261002'))
from verify_format import Doc, NS, W, on, val, text
sys.path.insert(0, str(ROOT / 'workpapers/全文深度校审_20261003'))
from extract_word import content

path = ROOT / 'output/排版完成.docx'
doc = Doc(path)
root = E.fromstring(doc.z.read('word/document.xml'))
body = root.find('w:body', NS)
values = json.loads((ROOT / 'workpapers/模板迁移_20261002/template_analysis_values.json').read_text())

def role(p):
    sid = val(p.find('w:pPr/w:pStyle', NS), doc.default)
    name = val(doc.styles[sid].find('w:name', NS), '') if sid in doc.styles else ''
    return name.removeprefix('模板迁移 ') if name.startswith('模板迁移 ') else name

def pagination(pp):
    return {k: (on(pp, k) if k in pp else k in ('widowControl', 'snapToGrid'))
            for k in ('keepNext', 'keepLines', 'pageBreakBefore', 'widowControl', 'snapToGrid')}

additional = defaultdict(lambda: {'template': {}, 'actual_forms': Counter(), 'differences': Counter()})
color_problems = []
for p in doc.ps:
    cat = role(p)
    if cat not in values:
        continue
    _, pp, _, runs, _ = doc.resolve(p)
    expected = pagination(values[cat]['effective_pPr'])
    actual = pagination(pp)
    group = additional[cat]
    group['template'] = expected
    group['actual_forms'][json.dumps(actual, sort_keys=True)] += 1
    for k in expected:
        if actual[k] != expected[k]:
            group['differences'][k] += 1
    for txt, rp in runs:
        color = rp.get('color', {}).get('val', 'auto')
        if color not in ('auto', '000000'):
            color_problems.append({'role': cat, 'text': txt, 'color': rp.get('color')})

references, captions, equations = {}, {}, {}
uses = defaultdict(list)
errors = []
starts = root.xpath('.//w:bookmarkStart/@w:name', namespaces=NS)
names = set(starts)
sequence = defaultdict(list)
for i, n in enumerate(body):
    txt = content(n)
    cat = role(n) if n.tag == W+'p' else 'table'
    if cat == 'bibliography':
        m = re.match(r'\[(\d+)\]', txt)
        keys = n.xpath('.//w:bookmarkStart/@w:name', namespaces=NS)
        key = next((k for k in keys if k.startswith('ref-')), None)
        if not m or not key:
            errors.append({'malformed_reference': i, 'text': txt[:80]})
        else:
            references[int(m[1])] = {'anchor': key, 'text': txt, 'block': i}
    if cat in ('figure_caption', 'table_caption'):
        m = re.match(r'([图表])\s*([A-F\d]+)-(\d+)', txt)
        if m:
            key = ''.join(m.groups()[:1]) + m[2] + '-' + m[3]
            if key in captions:
                errors.append({'duplicate_caption': key})
            captions[key] = i
    if any(re.search(r'\bSEQ\s+Eq', c) for c in n.xpath('.//w:instrText/text()', namespaces=NS)):
        for ch, number in re.findall(r'\(([A-F\d]+)-(\d+)\)', txt):
            key = ch+'-'+number
            if key in equations:
                errors.append({'duplicate_equation': key})
            equations[key] = i
            sequence[ch].append(int(number))

by_anchor = {v['anchor']: k for k, v in references.items()}
for i, n in enumerate(body):
    txt = content(n)
    cat = role(n) if n.tag == W+'p' else 'table'
    for link in n.xpath('.//w:hyperlink[@w:anchor]', namespaces=NS):
        anchor = link.get(W+'anchor')
        if anchor not in names:
            errors.append({'missing_link_target': anchor, 'block': i})
        if anchor in by_anchor and cat != 'bibliography':
            shown = content(link).strip('[] ')
            if shown.isdigit() and int(shown) != by_anchor[anchor]:
                errors.append({'wrong_citation_number': shown, 'target': by_anchor[anchor], 'block': i})
            uses[by_anchor[anchor]].append(i)
    if cat != 'bibliography' and not cat.lower().startswith('toc'):
        # Mathematical intervals have no reference hyperlinks and are not cited literature.
        for group in re.findall(r'\[(\d+(?:[,，、–—-]\d+)*)\]', txt):
            if not re.search('[,，、–—-]', group):
                nums = [int(group)]
            else:
                nums = []
                for token in re.split('[,，、]', group):
                    if re.search('[–—-]', token):
                        lo, hi = map(int, re.split('[–—-]', token))
                        nums.extend(range(lo, hi+1))
                    else:
                        nums.append(int(token))
            if all(k in references for k in nums):
                for k in nums:
                    if i not in uses[k]:
                        uses[k].append(i)
        if cat not in ('figure_caption', 'table_caption'):
            for a, b, c in re.findall(r'(图|表)\s*([A-F\d]+)-(\d+)', txt):
                key = a+b+'-'+c
                if key not in captions:
                    errors.append({'missing_caption_reference': key, 'block': i})
            for a, b in re.findall(r'式[（(]([A-F\d]+)-(\d+)[）)]', txt):
                if a+'-'+b not in equations:
                    errors.append({'missing_equation_reference': a+'-'+b, 'block': i})
    for code in n.xpath('.//w:instrText/text()', namespaces=NS):
        m = re.search(r'\b(?:REF|PAGEREF)\s+([^\s\\]+)', code)
        if m and m[1] not in names:
            errors.append({'missing_field_target': m[1], 'block': i})
for ch, nums in sequence.items():
    if nums != list(range(1, len(nums)+1)):
        errors.append({'equation_sequence': ch, 'numbers': nums})
for k in references:
    if not uses[k]:
        errors.append({'uncited_reference': k})

result = {
    'document': str(path), 'references': len(references), 'equations': len(equations),
    'figures': sum(k.startswith('图') for k in captions), 'tables': sum(k.startswith('表') for k in captions),
    'reference_use_counts': {k: len(uses[k]) for k in references},
    'structural_errors': errors,
    'duplicate_bookmarks': {k: v for k, v in Counter(starts).items() if v > 1},
    'color_problems': color_problems,
    'pagination_properties': {k: {'template': v['template'],
        'actual_forms': [{'count': count, **json.loads(form)} for form, count in v['actual_forms'].items()],
        'differences': dict(v['differences'])} for k, v in additional.items()},
    'scope': 'Structural/actual-format checks do not certify every citation interpretation or every mathematical definition.',
}
(HERE/'current_acceptance.json').write_text(json.dumps(result, ensure_ascii=False, indent=2)+'\n')
print(json.dumps({k: v for k, v in result.items() if k not in ('reference_use_counts', 'pagination_properties')}, ensure_ascii=False))
