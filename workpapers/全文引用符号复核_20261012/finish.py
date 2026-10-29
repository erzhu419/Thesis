"""Import only Word-native field results; preserve native objects and formatting."""
from collections import defaultdict
from pathlib import Path
from zipfile import ZipFile
import json
import re
import sys
from lxml import etree as E
from audit import ROOT, HERE, W, NS

sys.path.insert(0, str(ROOT / 'workpapers/七章重写稿整合_20261010'))
from integrate import body_spans
OUT = ROOT / 'output/排版完成_编号符号引用复核_20261012.docx'
STAGE = Path('/mnt/c/Users/erzhu/AppData/Local/Temp/thesis_refs_20261012/word.docx')


def result_nodes(root):
    stack, occurrences = [], defaultdict(int)
    for n in root.iter():
        if n.tag == W + 'fldChar':
            kind = n.get(W + 'fldCharType')
            if kind == 'begin': stack.append({'code': '', 'result': False, 'nodes': []})
            elif kind == 'separate' and stack: stack[-1]['result'] = True
            elif kind == 'end' and stack:
                f = stack.pop()
                code = ' '.join(f['code'].split())
                if re.match(r'^(SEQ|REF|PAGEREF)\s', code):
                    occurrences[code] += 1
                    yield (code, occurrences[code]), f['nodes']
        elif n.tag == W + 'instrText' and stack: stack[-1]['code'] += n.text or ''
        elif n.tag == W + 't' and stack and stack[-1]['result']: stack[-1]['nodes'].append(n)


with ZipFile(STAGE) as z:
    root = E.fromstring(z.read('word/document.xml'))
    caches = {key: ''.join(n.text or '' for n in nodes) for key,nodes in result_nodes(root)}
with ZipFile(OUT) as z:
    raw = z.read('word/document.xml')
    root = E.fromstring(raw)
    body = list(root.find(W + 'body'))
    original = [E.tostring(p) for p in body]
    changed_fields = 0
    total = 0
    for key,nodes in result_nodes(root):
        assert key in caches, ('missing native cache', key)
        current = ''.join(n.text or '' for n in nodes)
        assert 'Error!' not in caches[key] and '错误!' not in caches[key], key
        total += 1
        if current != caches[key]:
            assert nodes, key
            nodes[0].text = caches[key]
            for n in nodes[1:]: n.text = ''
            changed_fields += 1
    spans = body_spans(raw)
    edits = [i for i,p in enumerate(body) if E.tostring(p) != original[i]]
    for i in reversed(edits):
        a,b = spans[i]
        raw = raw[:a] + E.tostring(body[i], encoding='utf-8', with_tail=False) + raw[b:]
    parts = [(info,raw if info.filename == 'word/document.xml' else z.read(info.filename)) for info in z.infolist()]
with ZipFile(OUT, 'w') as z:
    for info,data in parts: z.writestr(info,data)
result = {'native_fields_checked': total, 'field_caches_updated': changed_fields, 'blocks_changed': len(edits)}
(HERE / 'Word域更新记录.json').write_text(json.dumps(result, ensure_ascii=False, indent=2))
print(json.dumps(result, ensure_ascii=False))
