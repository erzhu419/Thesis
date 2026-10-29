"""Extract numbered objects, references, symbol contexts, and abbreviation uses."""
from collections import Counter, defaultdict
from pathlib import Path
from zipfile import ZipFile
import json
import re
import sys

from lxml import etree as E

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
SOURCE = ROOT / 'output/排版完成_参考文献排版修复_20261011.docx'
sys.path.insert(0, str(ROOT / 'workpapers/全文深度校审_20261003'))
import extract_word as extractor
NS = extractor.NS
W = '{' + NS['w'] + '}'
M = '{' + NS['m'] + '}'
math_content = extractor.content


def content(node):
    if node.tag in (W + 't', M + 't'):
        return node.text or ''
    if node.tag == W + 'instrText':
        return ''
    if node.tag.startswith(M):
        return math_content(node)
    return ''.join(content(c) for c in node)


extractor.content = content
NUMBER = r'([1-7A-F])-([0-9]+[a-z]?)'
ABBREVIATION = re.compile(r'(?<![A-Za-z0-9-])(?:CVaR|SINDy(?:-RL)?|MoCo|Sim2Real|RE-SAC|CS-BAPR|MC-WM|H2O\+?|TD3|[A-Z]{2,}(?:[-+][A-Za-z0-9]+)*\+?)(?![A-Za-z0-9])')


def save(name, value):
    (HERE / name).write_text(json.dumps(value, ensure_ascii=False, indent=2))


def run(path=SOURCE):
    with ZipFile(path) as z:
        doc = E.fromstring(z.read('word/document.xml'))
    body = doc.find(W + 'body')
    blocks, equations, captions, references = [], {}, {}, {}
    bookmarks = doc.xpath('.//w:bookmarkStart/@w:name', namespaces=NS)
    names = set(bookmarks)
    errors = []
    chapter = '前置'
    abbrevs = defaultdict(list)
    symbols = defaultdict(list)
    uses = defaultdict(list)
    for i, p in enumerate(body):
        t = content(p)
        s = p.find(W + 'pPr/' + W + 'pStyle')
        s = s.get(W + 'val') if s is not None else 'table' if p.tag == W + 'tbl' else ''
        if s in ('chapter', 'bibliographytitle', 'symboltitle', 'abstractcntitle', 'abstractentitle'):
            chapter = t.strip()
        math = p.findall('.//' + M + 'oMath')
        row = {'i': i, 'style': s, 'chapter': chapter, 'text': t,
               'math': [content(x) for x in math],
               'fields': p.xpath('.//w:instrText/text()', namespaces=NS),
               'bookmarks': p.xpath('.//w:bookmarkStart/@w:name', namespaces=NS)}
        blocks.append(row)
        if s == 'bibliography':
            n = int(re.match(r'\[(\d+)\]', t)[1])
            references[n] = row
        if s in ('figurecaption', 'tablecaption'):
            match = re.match(r'([图表])\s*' + NUMBER, t)
            if match:
                key = match[1] + match[2] + '-' + match[3]
                if key in captions:
                    errors.append({'duplicate_caption': key})
                captions[key] = row
        if math and p.tag == W + 'tbl':
            for tr in p.findall(W + 'tr'):
                cells = tr.findall(W + 'tc')
                label = ''.join(x.text or '' for x in cells[-1].iter(W + 't')).strip()
                match = re.fullmatch(r'[（(]' + NUMBER + r'[）)]', label)
                if match:
                    key = match[1] + '-' + match[2]
                    if key in equations:
                        errors.append({'duplicate_equation': key})
                    equations[key] = row
        if not s.startswith('TOC') and s != 'bibliography':
            for a in ABBREVIATION.finditer(t):
                abbrevs[a[0]].append({'i': i, 'chapter': chapter, 'style': s,
                                      'context': t[max(0, a.start()-120):a.end()+150]})
            for x in math:
                for v in x.iter(M + 't'):
                    for symbol in re.findall(r'[A-Za-zα-ωΑ-Ωℙℚℝℕℱℋℐℰℒℳℛℙℓℐ]', v.text or ''):
                        if not symbols[symbol] or symbols[symbol][-1]['i'] != i:
                            symbols[symbol].append({'i': i, 'chapter': chapter})
    eq_uses, cap_uses = defaultdict(list), defaultdict(list)
    for row, p in zip(blocks, body):
        i, s, t = row['i'], row['style'], row['text']
        for link in p.findall('.//' + W + 'hyperlink'):
            anchor = link.get(W + 'anchor')
            if anchor and anchor not in names:
                errors.append({'missing_bookmark': anchor, 'i': i})
            if anchor and anchor.startswith('ref-'):
                uses[anchor].append(i)
        for code in row['fields']:
            m = re.search(r'\b(?:REF|PAGEREF)\s+([^\s\\]+)', code)
            if m and m[1] not in names:
                errors.append({'missing_field_target': m[1], 'i': i})
        if s.startswith('TOC') or s in ('bibliography', 'figurecaption', 'tablecaption'):
            continue
        for match in re.finditer(r'([图表])\s*' + NUMBER, t):
            key = match[1] + match[2] + '-' + match[3]
            cap_uses[key].append(i)
            if key not in captions:
                errors.append({'missing_caption': key, 'i': i})
        for match in re.finditer(r'([图表])\s*([1-7A-F])-(\d+)\s*至\s*\1\s*\2-(\d+)', t):
            for number in range(int(match[3]), int(match[4]) + 1):
                key = f'{match[1]}{match[2]}-{number}'
                if i not in cap_uses[key]:
                    cap_uses[key].append(i)
                if key not in captions:
                    errors.append({'missing_caption_range_member': key, 'i': i})
        if i in {v['i'] for v in equations.values()}:
            continue
        for match in re.finditer(r'式\s*[（(]' + NUMBER + r'[）)]', t):
            key = match[1] + '-' + match[2]
            eq_uses[key].append(i)
            if key not in equations:
                errors.append({'missing_equation': key, 'i': i})
    reference_numbers = {}
    for n, row in references.items():
        for name in row['bookmarks']:
            if name.startswith('ref-'):
                reference_numbers[name] = n
                if not uses[name]:
                    errors.append({'uncited_reference': n, 'i': row['i']})
    for row, p in zip(blocks, body):
        for link in p.findall('.//' + W + 'hyperlink'):
            anchor = link.get(W + 'anchor')
            if anchor in reference_numbers:
                label = content(link).strip('[] ')
                if label.isdigit() and int(label) != reference_numbers[anchor]:
                    errors.append({'wrong_citation_cache': label, 'expected': reference_numbers[anchor], 'i': row['i']})
    save('全文块.json', blocks)
    save('简称出现位置.json', abbrevs)
    save('数学符号出现位置.json', symbols)
    save('编号与引用检查.json', {'equations': equations, 'captions': captions,
          'equation_uses': eq_uses, 'caption_uses': cap_uses, 'errors': errors,
          'duplicate_bookmarks': {k:v for k,v in Counter(bookmarks).items() if v>1}})
    groups = defaultdict(list)
    for row in blocks:
        if not row['style'].startswith('TOC') and row['style'] != 'bibliography':
            groups[row['chapter']].append(f"[{row['i']} {row['style']}] {row['text']}")
    for n, (chapter, lines) in enumerate(groups.items()):
        (HERE / f'{n:02d}_{chapter[:14]}.txt').write_text('\n'.join(lines))
    contexts = []
    for key, row in equations.items():
        i = row['i']
        contexts.append(f'## ({key}) 块{i}\n' + '\n'.join(f"[{b['i']}] {b['text']}" for b in blocks[max(0,i-1):i+3]))
    (HERE / '公式与释义上下文.txt').write_text('\n\n'.join(contexts))
    print(json.dumps({'blocks': len(blocks), 'equations': len(equations), 'figures_tables': len(captions),
                      'references': len(references), 'abbreviation_types': len(abbrevs), 'errors': errors}, ensure_ascii=False))


if __name__ == '__main__':
    run(Path(sys.argv[1]) if len(sys.argv)>1 else SOURCE)
