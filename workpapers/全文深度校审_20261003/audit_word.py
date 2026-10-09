"""Inspect reference targets, numbering, and metadata against the current Word."""
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from zipfile import ZipFile

from lxml import etree as E
from extract_word import NS, content

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
W = '{' + NS['w'] + '}'
R = '{http://schemas.openxmlformats.org/officeDocument/2006/relationships}'


def run(path):
    with ZipFile(path) as z:
        doc = E.fromstring(z.read('word/document.xml'))
        body = doc.find('w:body', NS)
        rels = {n.get('Id'): n.get('Target') for n in E.fromstring(z.read('word/_rels/document.xml.rels'))}
    cache = json.loads((ROOT / 'workpapers/manuscript_audit/reference_verification.json').read_text())
    cache.update(json.loads((HERE/'reference_updates.json').read_text()))
    starts = doc.xpath('.//w:bookmarkStart/@w:name', namespaces=NS)
    names = set(starts)
    references = {}
    captions = {}
    equations = {}
    uses = defaultdict(list)
    errors = []
    seqs = defaultdict(list)
    refs_by_key = {}
    for i, node in enumerate(body):
        text = content(node)
        style = node.xpath('./w:pPr/w:pStyle/@w:val', namespaces=NS)
        style = style[0] if style else ''
        if style == 'bibliography':
            number = int(re.match(r'\[(\d+)\]', text).group(1))
            anchor = next(n for n in node.xpath('.//w:bookmarkStart/@w:name', namespaces=NS) if n.startswith('ref-'))
            key = anchor[4:]
            c = cache.get(key, {})
            md = c.get('metadata') or c.get('entry', {})
            title = md.get('title', '')
            if isinstance(title, list):
                title = ' / '.join(title)
            urls = [rels[n.get(R+'id')] for n in node.xpath('.//w:hyperlink', namespaces=NS) if n.get(R+'id') in rels]
            references[number] = dict(block=i, key=key, text=text, urls=urls,
                                      primary_source=c.get('source'), verified_title=title,
                                      metadata_checked=c.get('checked'), cached_title_match=c.get('title_match'))
            refs_by_key[key] = number
        if style in ('figurecaption', 'tablecaption'):
            match = re.match(r'([图表])\s*([A-F\d]+)-(\d+)', text)
            if match:
                name = match[1] + match[2] + '-' + match[3]
                if name in captions:
                    errors.append({'duplicate_caption': name, 'blocks': [captions[name], i]})
                captions[name] = i
        codes = node.xpath('.//w:instrText/text()', namespaces=NS)
        if any(re.search(r'\bSEQ\s+Eq', c) for c in codes):
            numbers = re.findall(r'\(([A-F\d]+)-(\d+)\)', text)
            for chapter, n in numbers:
                key = f'{chapter}-{n}'
                if key in equations:
                    errors.append({'duplicate_equation': key, 'blocks': [equations[key], i]})
                equations[key] = i
                seqs[chapter].append(int(n))
        if style != 'bibliography':
            for link in node.xpath('.//w:hyperlink[@w:anchor]', namespaces=NS):
                anchor = link.get(W+'anchor')
                if anchor.startswith('ref-'):
                    uses[anchor[4:]].append(dict(block=i, text=text, shown=content(link)))
    for i, node in enumerate(body):
        style = node.xpath('./w:pPr/w:pStyle/@w:val', namespaces=NS)
        if not style or style[0] not in ('bibliography', 'bibliographytitle'):
            text = content(node)
            for group in re.findall(r'\[(\d+(?:[,，、–—-]\d+)*)\]', text):
                numbers = []
                for item in re.split('[,，、]', group):
                    if re.search('[–—-]', item):
                        lo, hi = map(int, re.split('[–—-]', item))
                        numbers.extend(range(lo, hi+1))
                    else:
                        numbers.append(int(item))
                if any(number not in references for number in numbers):
                    continue
                for number in numbers:
                    if number in references:
                        key = references[number]['key']
                        if not any(u['block'] == i for u in uses[key]):
                            uses[key].append(dict(block=i, text=text, shown='['+group+']', plain_text=True))
        for link in node.xpath('.//w:hyperlink[@w:anchor]', namespaces=NS):
            anchor = link.get(W+'anchor')
            if anchor not in names:
                errors.append({'missing_link_target': anchor, 'block': i})
            if anchor.startswith('ref-') and anchor[4:] in refs_by_key:
                shown = content(link).strip('[] ')
                if shown.isdigit() and int(shown) != refs_by_key[anchor[4:]]:
                    errors.append({'wrong_citation_number': shown, 'target': refs_by_key[anchor[4:]], 'block': i})
        for code in node.xpath('.//w:instrText/text()', namespaces=NS):
            match = re.search(r'\b(?:REF|PAGEREF)\s+([^\s\\]+)', code)
            if match and match[1] not in names:
                errors.append({'missing_field_target': match[1], 'block': i})
        if style and style[0] not in ('bibliography', 'figurecaption', 'tablecaption'):
            text = content(node)
            for match in re.finditer(r'(图|表)\s*([A-F\d]+)-(\d+)', text):
                name = match[1] + match[2] + '-' + match[3]
                if name not in captions:
                    errors.append({'missing_caption_reference': name, 'block': i})
            for match in re.finditer(r'式[（(]([A-F\d]+)-(\d+)[）)]', text):
                name = match[1] + '-' + match[2]
                if name not in equations:
                    errors.append({'missing_equation_reference': name, 'block': i})
    for chapter, nums in seqs.items():
        if nums != list(range(1, len(nums)+1)):
            errors.append({'equation_sequence': chapter, 'numbers': nums})
    for number, ref in references.items():
        ref['citations'] = uses[ref['key']]
        if not ref['citations']:
            errors.append({'uncited_reference': number, 'key': ref['key']})
    duplicates = {k:v for k,v in Counter(starts).items() if v > 1}
    result = dict(manuscript=str(path), references=references, captions=captions, equations=equations,
                  errors=errors, duplicate_bookmark_names=duplicates, sections=len(doc.xpath('.//w:sectPr', namespaces=NS)),
                  native_math=len(doc.xpath('.//m:oMath', namespaces=NS)))
    (HERE/'structural_audit.json').write_text(json.dumps(result, ensure_ascii=False, indent=2))
    lines = ['# 现行Word引文逐项落点', '以下记录只用于校审，不写入论文正文。', '']
    for number, ref in references.items():
        lines.extend([f'## [{number}] {ref["key"]}', ref['text'],
                      f'原始元数据：{ref["primary_source"]}；日期：{ref["metadata_checked"]}；题名：{ref["verified_title"]}'])
        for u in ref['citations']:
            lines.append(f'- 正文块{u["block"]}：{u["text"]}')
        lines.append('')
    (HERE/'现行Word引文逐项落点.md').write_text('\n'.join(lines))
    print(json.dumps(dict(references=len(references), equations=len(equations), captions=len(captions),
                          errors=errors, duplicate_bookmarks=duplicates), ensure_ascii=False))


if __name__ == '__main__':
    run(ROOT/'output/排版完成.docx')
