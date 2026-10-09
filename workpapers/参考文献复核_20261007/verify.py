"""Check lost content, citation destinations and editable objects after Word saves."""
import importlib.util
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from zipfile import ZipFile

from lxml import etree as E

ROOT = Path('/home/erzhu419/mine_code/Thesis')
WORK = Path(__file__).resolve().parent
SOURCE = WORK / '修改前.docx'
OUTPUT = ROOT / 'output/排版完成_参考文献复核_20261007.docx'
assert SOURCE.read_bytes() == (ROOT/'output/排版完成_最终通读校订_20261007.docx').read_bytes()
spec = importlib.util.spec_from_file_location('helper', ROOT/'workpapers/全文深度校审_20261003/extract_word.py')
helper = importlib.util.module_from_spec(spec)
spec.loader.exec_module(helper)
NS = helper.NS
NS['r'] = 'http://schemas.openxmlformats.org/officeDocument/2006/relationships'
old_content = helper.content
helper.content = lambda n: old_content(n) if E.QName(n).namespace in (NS['w'], NS['m']) else ''
W = '{' + NS['w'] + '}'


def shape(node):
    return (node.tag, tuple(sorted(node.attrib.items())), node.text,
            tuple(shape(c) for c in node if c.tag != W+'lastRenderedPageBreak'))


def load(path):
    with ZipFile(path) as z:
        return {n: z.read(n) for n in z.namelist()}


before_parts, after_parts = load(SOURCE), load(OUTPUT)
before, after = [E.fromstring(parts['word/document.xml']) for parts in (before_parts, after_parts)]
old_body, body = [d.find('w:body', NS) for d in (before, after)]
changed = set(json.loads((WORK/'修改位置.json').read_text()))
issues, blocks, refs, captions, equations = [], [], {}, {}, {}
eqnums, caption_uses, citation_uses = defaultdict(list), defaultdict(list), defaultdict(list)
bookmarks = after.xpath('//w:bookmarkStart/@w:name', namespaces=NS)
targets = set(bookmarks)
for i, node in enumerate(body):
    styles = node.xpath('./w:pPr/w:pStyle/@w:val', namespaces=NS)
    style = styles[0] if styles else ''
    value = helper.content(node)
    fields = node.xpath('.//w:instrText/text()', namespaces=NS)
    blocks.append({'index': i, 'style': style, 'text': value})
    if i not in changed and not any('PAGEREF' in f for f in fields):
        if value != helper.content(old_body[i]):
            issues.append({'unplanned_content_difference': i})
    if style == 'bibliography':
        number = int(re.match(r'\[(\d+)\]', value)[1])
        anchors = node.xpath('.//w:bookmarkStart/@w:name', namespaces=NS)
        key = next(k for k in anchors if k.startswith('ref-'))
        refs[key] = {'number': number, 'block': i, 'text': value}
        if not re.match(r'^\['+str(number)+r'\] ?\t', value):
            issues.append({'reference_indent': number})
    if style in ('figurecaption', 'tablecaption'):
        m = re.match(r'([图表])\s*([A-F\d]+-\d+)', value)
        if m:
            name = m[1]+m[2]
            if name in captions:
                issues.append({'duplicate_caption': name})
            captions[name] = i
    for row in node.findall('w:tr', NS):
        cells = row.findall('w:tc', NS)
        number = helper.content(cells[-1]).strip()
        if re.fullmatch(r'\([A-F\d]+-\d+\)', number):
            key = number[1:-1]
            if key in equations:
                issues.append({'duplicate_equation': key})
            equations[key] = i
            chapter, n = key.split('-')
            eqnums[chapter].append(int(n))
    for link in node.xpath('.//w:hyperlink[@w:anchor]', namespaces=NS):
        anchor = link.get(W+'anchor')
        if anchor not in targets:
            issues.append({'missing_hyperlink_target': anchor, 'block': i})
        if anchor.startswith('ref-') and style != 'bibliography':
            citation_uses[anchor].append({'block': i, 'text': value,
                                        'displayed': helper.content(link).strip('[] ')})
    for f in fields:
        m = re.search(r'\b(?:REF|PAGEREF)\s+([^\s\\]+)', f)
        if m and m[1] not in targets:
            issues.append({'missing_field_target': m[1], 'block': i})
    if '未找到引用源' in value or 'Error!' in value:
        issues.append({'field_error': i})

for key, ref in refs.items():
    if not citation_uses[key]:
        issues.append({'uncited_reference': ref['number']})
    for use in citation_uses[key]:
        if not use['displayed'].isdigit() or int(use['displayed']) != ref['number']:
            issues.append({'citation_number_mismatch': use['block'], 'reference': ref['number']})
for item in blocks:
    if item['style'] not in ('figurecaption', 'tablecaption', 'bibliography'):
        for m in re.finditer(r'(图|表)\s*([A-F\d]+-\d+)', item['text']):
            key = m[1]+m[2]
            caption_uses[key].append(item['index'])
            if key not in captions:
                issues.append({'missing_caption': key, 'block': item['index']})
        for m in re.finditer(r'式[（(]([A-F\d]+-\d+)[）)]', item['text']):
            if m[1] not in equations:
                issues.append({'missing_equation': m[1], 'block': item['index']})
for chapter, nums in eqnums.items():
    if nums != list(range(1, len(nums)+1)):
        issues.append({'equation_sequence': chapter})
for key in captions:
    if key not in caption_uses:
        issues.append({'unmentioned_caption': key})
for name, count in Counter(bookmarks).items():
    if count > 1:
        issues.append({'duplicate_bookmark': name})

math_before = before.xpath('//m:oMath', namespaces=NS)
math_after = after.xpath('//m:oMath', namespaces=NS)
assert [shape(n) for n in math_before] == [shape(n) for n in math_after], 'Native equation structure changed.'
assert [helper.content(n) for n in before.xpath('//w:tbl', namespaces=NS)] == [
    helper.content(n) for n in after.xpath('//w:tbl', namespaces=NS)], 'Table content changed.'
section_shapes = []
for document, parts in [(before, before_parts), (after, after_parts)]:
    relationships = E.fromstring(parts['word/_rels/document.xml.rels'])
    mapping = {n.get('Id'): n.get('Target') for n in relationships}
    sections = []
    for section in document.xpath('//w:sectPr', namespaces=NS):
        section = E.fromstring(E.tostring(section))
        for reference in section.xpath('./w:headerReference|./w:footerReference', namespaces=NS):
            attribute = '{'+NS['r']+'}id'
            reference.set(attribute, mapping[reference.get(attribute)])
        sections.append(shape(section))
    section_shapes.append(sections)
assert section_shapes[0] == section_shapes[1], 'Section layout changed.'
for name in before_parts:
    if name.startswith(('word/media/', 'word/embeddings/')):
        assert before_parts[name] == after_parts[name], name
assert helper.content(body[64]) == helper.content(old_body[64])
assert len(refs) == 214
assert sorted(r['number'] for r in refs.values()) == list(range(1, 215))
rels = E.fromstring(after_parts['word/_rels/document.xml.rels'])
rel_targets = {n.get('Id'): n.get('Target') for n in rels}
for n, url in [(5, 'https://arxiv.org/abs/2105.00376v2'),
               (43, 'https://arxiv.org/abs/2508.20784v2')]:
    p = body[1961+n]
    link = next(l for l in p.findall('w:hyperlink', NS) if url in helper.content(l))
    assert rel_targets[link.get('{'+NS['r']+'}id')] == url

result = {'output': str(OUTPUT), 'body_blocks': len(body), 'references': len(refs),
          'arxiv_references': sum('arxiv.org/abs/' in r['text'] for r in refs.values()),
          'native_math': len(math_after), 'numbered_equations': len(equations),
          'captions': len(captions), 'sections': len(after.xpath('//w:sectPr', namespaces=NS)),
          'math_structure_preserved': True, 'tables_and_media_preserved': True,
          'author_opening_preserved': True, 'issues': issues}
(WORK/'成品检查.json').write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')
(WORK/'成品通读文本.txt').write_text('\n'.join(f'[{b["index"]}] {b["text"]}' for b in blocks), encoding='utf-8')
(WORK/'成品文献与引用.json').write_text(json.dumps(
    [{'key': k, **r, 'uses': citation_uses[k]} for k, r in refs.items()], ensure_ascii=False, indent=2), encoding='utf-8')
print(json.dumps(result, ensure_ascii=False, indent=2))
assert not issues, issues
