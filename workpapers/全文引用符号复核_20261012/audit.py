"""Inspect the current manuscript's labels, field destinations, and math contexts."""
from collections import Counter, defaultdict
from pathlib import Path
from zipfile import ZipFile
import json
import re
import sys

from lxml import etree as E

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
BASE = ROOT / 'output/排版完成_章节补充整合_20261012.docx'
sys.path.insert(0, str(ROOT / 'workpapers/全文编号符号简称复核_20261011'))
import scan
scan.HERE = HERE
original_math = scan.math_content


def structured_math(node):
    if node.tag == scan.M + 'd':
        pr = node.find(scan.M + 'dPr')
        def character(name, default):
            child = pr.find(scan.M + name) if pr is not None else None
            return child.get(scan.M + 'val', default) if child is not None else default
        return character('begChr', '(') + character('sepChr', ',').join(
            scan.content(c) for c in node.findall(scan.M + 'e')) + character('endChr', ')')
    return original_math(node)


scan.math_content = structured_math
W, M, NS, text = scan.W, scan.M, scan.NS, scan.content


def style(p):
    v = p.find(W + 'pPr/' + W + 'pStyle')
    return v.get(W + 'val') if v is not None else ''


def fields(p):
    stack = []
    result = []
    for node in p.iter():
        if node.tag == W + 'fldSimple':
            result.append({'code': node.get(W + 'instr'), 'display': text(node)})
        elif node.tag == W + 'fldChar':
            kind = node.get(W + 'fldCharType')
            if kind == 'begin':
                stack.append({'code': '', 'display': '', 'result': False})
            elif kind == 'separate' and stack:
                stack[-1]['result'] = True
            elif kind == 'end' and stack:
                item = stack.pop()
                result.append(item)
                if stack and stack[-1]['result']:
                    stack[-1]['display'] += item['display']
        elif node.tag == W + 'instrText' and stack:
            stack[-1]['code'] += node.text or ''
        elif node.tag == W + 't' and stack and stack[-1]['result']:
            stack[-1]['display'] += node.text or ''
    return result


def write(name, value):
    (HERE / name).write_text(json.dumps(value, ensure_ascii=False, indent=2))


def run(path=BASE):
    scan.run(path)
    with ZipFile(path) as z:
        root = E.fromstring(z.read('word/document.xml'))
    body = root.find(W + 'body')
    inventory = json.loads((HERE / '全文块.json').read_text())
    base = json.loads((HERE / '编号与引用检查.json').read_text())
    equations, captions = base['equations'], base['captions']
    definitions = {}
    indices = {}
    for key, row in equations.items():
        indices[row['i']] = '(' + key + ')'
    for key, row in captions.items():
        indices[row['i']] = key
    for i, p in enumerate(body):
        for mark in p.iter(W + 'bookmarkStart'):
            definitions[mark.get(W + 'name')] = {'i': i, 'label': indices.get(i),
                 'style': style(p), 'text': text(p)[:240]}
    findings = []
    checks = []
    for i, p in enumerate(body):
        for f in fields(p):
            code = f['code'] or ''
            match = re.search(r'\b(REF|PAGEREF)\s+([^\s\\]+)', code)
            local_link = re.search(r'HYPERLINK\s+\\l\s+"([^"]+)"', code)
            if local_link:
                target = definitions.get(local_link[1])
                if target is None:
                    findings.append({'missing_local_field_link': local_link[1], 'i': i})
                else:
                    section = re.search(r'第([1-7](?:\.\d+){1,2})节', f['display'])
                    if section and not target['text'].startswith(section[1] + ' '):
                        findings.append({'local_section_link_mismatch': {'i': i, 'display': f['display'], 'target': target, 'anchor': local_link[1]}})
            if not match:
                continue
            target = definitions.get(match[2])
            entry = {'i': i, 'code': code, 'display': f['display'], 'target': target}
            checks.append(entry)
            if match[1] == 'REF' and target:
                actual = target['label']
                display = f['display'].strip()
                if actual and display != actual and display != actual.strip('()'):
                    findings.append({'field_label_mismatch': entry})
                elif not actual and re.fullmatch(r'\(?[2-6]-\d+\)?|[图表][1-7A-F]-\d+', display):
                    findings.append({'field_targets_unrelated_block': entry})
        for link in p.iter(W + 'hyperlink'):
            anchor = link.get(W + 'anchor')
            target = definitions.get(anchor)
            if not target:
                continue
            label = text(link).strip()
            if re.fullmatch(r'[图表][1-7A-F]-\d+|\([1-7A-F]-\d+\)', label):
                if label != target['label']:
                    findings.append({'link_label_mismatch': {'i': i, 'display': label, 'target': target, 'anchor': anchor}})
            elif style(p).startswith('TOC'):
                heading = re.sub(r'\d+$', '', label).strip()
                if heading != target['text'].strip():
                    findings.append({'toc_target_mismatch': {'i': i, 'display': heading, 'target': target, 'anchor': anchor}})
        if not style(p).startswith('TOC') and style(p) not in {'bibliography', 'figurecaption', 'tablecaption'}:
            for paragraph in ([p] if p.tag == W + 'p' else p.findall('.//' + W + 'p')):
                value = text(paragraph)
                for match in re.finditer(r'[（(]([1-7A-F]-\d+[a-z]?)[）)]', value):
                    key = match[1]
                    if i in indices and indices[i] == '(' + key + ')':
                        continue
                    if key not in equations:
                        findings.append({'unknown_equation_number': key, 'i': i, 'context': value[:300]})
                for match in re.finditer(r'第([1-7](?:\.\d+){1,2})节', value):
                    sec = match[1]
                    headings = [r for r in inventory if r['style'] in {'h2', 'h3'} and r['text'].startswith(sec + ' ')]
                    if not headings:
                        findings.append({'unknown_section': sec, 'i': i})
    sequences = {}
    for kind, items in [('eq', equations), ('caption', captions)]:
        groups = defaultdict(list)
        for key, row in items.items():
            m = re.fullmatch(r'([图表]?)([1-7A-F])-(\d+)([a-z]?)', key)
            if m:
                groups[m[1] + m[2]].append((int(m[3]), m[4], row['i']))
        for name, values in groups.items():
            values.sort(key=lambda v: v[2])
            numeric = [v[0] for v in values if not v[1]]
            sequences[kind + ':' + name] = numeric
            if numeric != list(range(1, max(numeric) + 1)):
                findings.append({'label_sequence': name, 'actual': numeric})
    write('字段与目标检查.json', {'findings': findings, 'checked_fields': checks, 'sequences': sequences})
    write('书签目标.json', definitions)
    (HERE / '公式与释义上下文.txt').write_text('\n\n'.join(
        f'## ({key}) 块{row["i"]}\n' + '\n'.join(
            f'[{r["i"]} {r["style"]}] {r["text"]}' for r in inventory[max(0,row['i']-2):row['i']+4])
        for key,row in equations.items()))
    print(json.dumps({'detailed_findings': findings, 'number_sequences': {k:len(v) for k,v in sequences.items()},
                      'fields_checked': len(checks)}, ensure_ascii=False))


if __name__ == '__main__':
    run(Path(sys.argv[1]) if len(sys.argv) > 1 else BASE)
