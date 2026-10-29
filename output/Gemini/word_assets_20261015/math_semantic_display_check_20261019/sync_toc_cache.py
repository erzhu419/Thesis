"""Synchronize only Word's automatic TOC page-number caches after reflow."""
from pathlib import Path
from zipfile import ZipFile
from copy import copy
from lxml import etree as E
import json
import re

A = Path(__file__).resolve().parent
P = A.parents[1] / '调整完格式的_公式符号复核_20261019.docx'
B = A / 'before_toc_cache.docx'
assert not B.exists()
B.write_bytes(P.read_bytes())
data = json.loads((A / 'after_small/toc_pages.json').read_text(encoding='utf-8-sig'))
pages = {item['bookmark'].casefold(): item['page'] for item in data['toc_pages']}
assert len(pages) == 114
W = '{http://schemas.openxmlformats.org/wordprocessingml/2006/main}'
with ZipFile(B) as z:
    raw = z.read('word/document.xml').decode('utf-8')
    root = E.fromstring(raw.encode('utf-8'))
    texts = list(root.iter(W + 't'))
    matches = list(re.finditer(r'<w:t\b[^>]*>(.*?)</w:t>', raw, flags=re.S))
    assert len(texts) == len(matches)
    stack, changes, replacements, seen = [], [], [], set()
    for node in root.iter():
        if node.tag == W + 'fldChar':
            kind = node.get(W + 'fldCharType')
            if kind == 'begin':
                stack.append({'code': '', 'results': [], 'result': False})
            elif kind == 'separate' and stack:
                stack[-1]['result'] = True
            elif kind == 'end' and stack:
                field = stack.pop()
                page = re.search(r'\bPAGEREF\s+(_Toc[0-9]+)\b', field['code'], flags=re.I)
                if not page:
                    continue
                key = page[1].casefold()
                seen.add(key)
                assert len(field['results']) == 1
                text = field['results'][0]
                old, new = text.text, pages[key]
                assert old.isdigit() and new.isdigit()
                if old != new:
                    match = matches[texts.index(text)]
                    assert match.group(1) == old
                    replacements.append((match.start(1), match.end(1), new))
                    changes.append({'bookmark': page[1], 'before': old, 'after': new})
        elif node.tag == W + 'instrText' and stack and not stack[-1]['result']:
            stack[-1]['code'] += node.text or ''
        elif node.tag == W + 't' and stack and stack[-1]['result']:
            stack[-1]['results'].append(node)
    assert seen == set(pages)
    for start, end, value in sorted(replacements, reverse=True):
        raw = raw[:start] + value + raw[end:]
    E.fromstring(raw.encode('utf-8'))
    with ZipFile(P, 'w') as out:
        for item in z.infolist():
            out.writestr(copy(item), raw.encode('utf-8') if item.filename == 'word/document.xml'
                         else z.read(item.filename))
(A / 'toc_cache_changes.json').write_text(json.dumps({
    'checked': len(seen), 'updated': len(changes), 'pages': data['pages'], 'changes': changes,
}, ensure_ascii=False, indent=2), encoding='utf-8')
print('TOC page caches:', len(changes), 'updated;', data['pages'], 'pages')
