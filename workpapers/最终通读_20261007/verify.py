"""Check preservation and actionable reference/numbering failures after Word saves."""
import importlib.util
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from zipfile import ZipFile

from lxml import etree as E

ROOT = Path('/home/erzhu419/mine_code/Thesis')
WORK = Path(__file__).resolve().parent
SOURCE = WORK/'修改前.docx'
OUTPUT = ROOT/'output/排版完成_最终通读校订_20261007.docx'
assert SOURCE.read_bytes() == (ROOT/'output/排版完成_自有论文与仓库补齐_20261007.docx').read_bytes(), 'The manuscript baseline changed during review.'
spec = importlib.util.spec_from_file_location('extract', ROOT/'workpapers/全文深度校审_20261003/extract_word.py')
helper = importlib.util.module_from_spec(spec)
spec.loader.exec_module(helper)
NS = helper.NS
old_content = helper.content
helper.content = lambda n: old_content(n) if E.QName(n).namespace in (NS['w'], NS['m']) else ''
W = '{'+NS['w']+'}'
with ZipFile(SOURCE) as z:
    before_parts = {n: z.read(n) for n in z.namelist()}
with ZipFile(OUTPUT) as z:
    after_parts = {n: z.read(n) for n in z.namelist()}
before = E.fromstring(before_parts['word/document.xml'])
after = E.fromstring(after_parts['word/document.xml'])
old_body, body = before.find('w:body', NS), after.find('w:body', NS)
changed = set(json.loads((WORK/'修改位置.json').read_text()))
issues, blocks, captions, equations = [], [], {}, {}
eqnums, caption_uses = defaultdict(list), defaultdict(list)
bookmarks = after.xpath('//w:bookmarkStart/@w:name', namespaces=NS)
targets, citations = set(bookmarks), set()
for i, node in enumerate(body):
    s = node.xpath('./w:pPr/w:pStyle/@w:val', namespaces=NS)
    style = s[0] if s else ''
    value = helper.content(node)
    fields = node.xpath('.//w:instrText/text()', namespaces=NS)
    blocks.append({'index': i, 'style': style, 'text': value})
    if i not in changed and not any('PAGEREF' in f for f in fields):
        if value != helper.content(old_body[i]):
            issues.append({'unplanned_content_difference': i})
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
            citations.add(anchor)
    for f in fields:
        m = re.search(r'\b(?:REF|PAGEREF)\s+([^\s\\]+)', f)
        if m and m[1] not in targets:
            issues.append({'missing_field_target': m[1], 'block': i})
    for term in ['零与正停', '低载尾部', '未找到引用源', 'Error!', '公交世界模型的补充实验.']:
        if term in value:
            issues.append({'unrepaired_text': term, 'block': i})
for item, node in zip(blocks, body):
    if item['style'] == 'bibliography':
        if not any(a in citations for a in node.xpath('.//w:bookmarkStart/@w:name', namespaces=NS) if a.startswith('ref-')):
            issues.append({'uncited_reference': item['index']})
    elif item['style'] not in ('figurecaption', 'tablecaption'):
        for m in re.finditer(r'(图|表)\s*([A-F\d]+-\d+)', item['text']):
            key = m[1]+m[2]
            caption_uses[key].append(item['index'])
            if key not in captions:
                issues.append({'missing_caption': key, 'block': item['index']})
        for m in re.finditer(r'式[（(]([A-F\d]+-\d+)[）)]', item['text']):
            if m[1] not in equations:
                issues.append({'missing_equation': m[1], 'block': item['index']})
reference_numbers = {}
for item, node in zip(blocks, body):
    if item['style'] == 'bibliography':
        number = int(re.match(r'\[(\d+)\]', item['text'])[1])
        for anchor in node.xpath('.//w:bookmarkStart/@w:name', namespaces=NS):
            if anchor.startswith('ref-'):
                reference_numbers[anchor] = number
for item, node in zip(blocks, body):
    if item['style'] != 'bibliography':
        for link in node.xpath('.//w:hyperlink[@w:anchor]', namespaces=NS):
            anchor = link.get(W+'anchor')
            shown = helper.content(link).strip('[] ')
            if anchor in reference_numbers and shown.isdigit() and int(shown) != reference_numbers[anchor]:
                issues.append({'citation_number_mismatch': item['index'], 'shown': shown, 'target': reference_numbers[anchor]})
for chapter, nums in eqnums.items():
    if nums != list(range(1, len(nums)+1)):
        issues.append({'equation_sequence': chapter, 'numbers': nums})
for name, count in Counter(bookmarks).items():
    if count > 1:
        issues.append({'duplicate_bookmark': name})
for key in captions:
    if key not in caption_uses:
        issues.append({'unmentioned_caption': key})
media_changed = [n for n in before_parts if n.startswith('word/media/') and before_parts[n] != after_parts[n]]
native_math = len(after.xpath('//m:oMath', namespaces=NS))
sections = len(after.xpath('//w:sectPr', namespaces=NS))
assert native_math == 2086 and sections == 16
assert helper.content(body[64]) == helper.content(old_body[64])
assert media_changed == ['word/media/image44.emf'], media_changed
assert all(before_parts[n] == after_parts[n] for n in before_parts if n.startswith('word/embeddings/') and n != 'word/embeddings/Microsoft_Visio_Drawing5.vsdx')
result = {'output': str(OUTPUT), 'body_blocks': len(body), 'native_math': native_math,
          'sections': sections, 'references': sum(b['style']=='bibliography' for b in blocks),
          'numbered_equations': len(equations), 'caption_count': len(captions),
          'media_changed': media_changed, 'author_opening_preserved': True, 'issues': issues}
(WORK/'成品检查.json').write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')
(WORK/'成品通读文本.txt').write_text('\n'.join(f'[{b["index"]}] {b["text"]}' for b in blocks), encoding='utf-8')
print(json.dumps(result, ensure_ascii=False, indent=2))
assert not issues, issues
