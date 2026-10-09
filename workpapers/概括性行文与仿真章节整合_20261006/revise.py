from pathlib import Path
from copy import deepcopy
from collections import Counter
from lxml import etree as E
import difflib, json, re, runpy, zipfile

ROOT = Path('/home/erzhu419/mine_code/Thesis')
WORK = Path(__file__).parent
OUT = ROOT / '排版完成_概括性行文与仿真章节整合_20261006.docx'
N = {'w':'http://schemas.openxmlformats.org/wordprocessingml/2006/main',
     'm':'http://schemas.openxmlformats.org/officeDocument/2006/math'}
W = '{' + N['w'] + '}'
z = zipfile.ZipFile(WORK / '修改前.docx')
root = E.fromstring(z.read('word/document.xml'))
body = root.find('w:body', N)
orig = list(body)
old = deepcopy(root)
plan = runpy.run_path(str(WORK / 'plan.py'))
changes = []

def txt(p):
    return ''.join(p.xpath('.//w:t/text()', namespaces=N))

def span(p, start, end, new):
    spans, pos = [], 0
    for t in p.xpath('.//w:t', namespaces=N):
        s = t.text or ''
        spans.append((t, pos, pos + len(s)))
        pos += len(s)
    if start == end:
        for t, a, b in spans:
            if a <= start <= b and b > a:
                s = t.text or ''
                t.text = s[:start-a] + new + s[start-a:]
                t.set('{http://www.w3.org/XML/1998/namespace}space', 'preserve')
                return
        raise AssertionError(('insertion', start))
    hits = [(t,a,b) for t,a,b in spans if b > start and a < end]
    assert hits
    for j,(t,a,b) in enumerate(hits):
        s = t.text or ''
        t.text = s[:max(0,start-a)] + (new if j == 0 else '') + s[min(len(s),end-a):]
        t.set('{http://www.w3.org/XML/1998/namespace}space', 'preserve')

def settext(p, updated):
    cur = txt(p)
    for op,a,b,c,d in reversed(difflib.SequenceMatcher(None,cur,updated,autojunk=False).get_opcodes()):
        if op != 'equal':
            span(p,a,b,updated[c:d])
    assert txt(p) == updated

for i,pairs in plan['EDITS'].items():
    p = orig[i]
    before = txt(p)
    for find, repl in pairs:
        cur = txt(p)
        if find is None:
            assert not p.xpath('.//m:oMath|.//w:drawing', namespaces=N), i
            updated = repl
        else:
            assert cur.count(find) == 1, (i, find, cur.count(find))
            updated = cur.replace(find, repl, 1)
        settext(p, updated)
    changes.append({'action':'行文调整','index':i,'before':before,'after':txt(p)})

section_map = {'2.3':'2.4','2.3.1':'2.4.1','2.3.2':'2.4.2','2.3.3':'2.4.3',
               '2.4':'2.3','2.4.1':'2.3.1','2.4.2':'2.3.2',
               '2.5':'2.4','2.5.1':'2.4.3','2.5.2':'2.4.4','2.5.3':'2.4.5','2.5.4':'2.4.6','2.6':'2.5'}
formula_map = {23:28,24:32,25:23,26:24,27:25,28:26,29:27,30:29,31:30,32:31}
for i,p in enumerate(orig[256:], 256):
    cur = txt(p)
    updated = re.sub(r'(?<=第)2\.[3-6](?:\.[1-4])?(?![\d.])',
                     lambda m: section_map.get(m[0],m[0]), cur)
    settext(p, updated)
    # Replace only the number span: matching the full '(2-24)' string can
    # accidentally move a digit across the SEQ field boundary.
    for match in reversed(list(re.finditer(r'\(2-(\d+)\)', updated))):
        number = int(match[1])
        if number in formula_map:
            span(p,match.start(1),match.end(1),str(formula_map[number]))
    if txt(p) != cur:
        changes.append({'action':'交叉引用及公式编号','index':i,'before':cur,'after':txt(p)})

headings = {503:'2.4 仿真平台与实验设置',504:'2.4.1 基础数据',510:'2.4.2 平台构建',
            528:'2.4.3 参数标定',543:'2.3 迁移学习基础',544:'2.3.1 分布差异',
            556:'2.3.2 运行预测',587:'2.4.4 控制响应',606:'2.4.5 实验设置',
            638:'2.4.6 适用范围',642:'2.5 本章小结'}
for i,s in headings.items():
    before = txt(orig[i])
    settext(orig[i],s)
    changes.append({'action':'章节合并及编号','index':i,'before':before,'after':s})

# Heading bookmarks originally extended to the following heading. Re-anchor
# chapter 2 headings so moving sections cannot reverse their bookmark ranges.
ends = {p.get(W+'id'):p for p in root.xpath('//w:bookmarkEnd',namespaces=N)}
for i in range(375,643):
    p = orig[i]
    style = p.find('w:pPr/w:pStyle',N)
    if style is None or style.get(W+'val') not in ('h2','h3'):
        continue
    target = orig[{566:503,567:528}.get(i,i)]
    starts = list(p.findall(W+'bookmarkStart'))
    for start in starts:
        end = ends[start.get(W+'id')]
        start.getparent().remove(start)
        end.getparent().remove(end)
        pp = target.find(W+'pPr')
        target.insert(1 if pp is not None else 0, start)
        target.append(end)

# Retained anchors from formerly merged subheadings follow their actual text.
for ident, last in [('104',540),('118',565)]:
    end = ends[ident]
    end.getparent().remove(end)
    orig[last].append(end)

# One coherent platform section: data -> construction -> calibration ->
# intervention checks -> experiment design -> scope.
order = (list(range(543,548)) + list(range(550,566)) +
         list(range(503,518)) + [526,527,529,530] + list(range(518,526)) +
         [541,542,548,549,528,531] + list(range(568,587)) +
         list(range(587,607)) + list(range(532,541)) + list(range(607,643)))
assert len(order) == len(set(order))
assert set(order) == set(range(503,643)) - {566,567}
for p in orig[503:643]:
    body.remove(p)
for i in order:
    body.insert(body.index(orig[643]),orig[i])
changes.append({'action':'第二章重排','source_indexes':order,'removed_heading_indexes':[566,567]})

for i,s in plan['INSERT_BEFORE'].items():
    target = orig[i]
    p = E.Element(W+'p')
    # Use existing body style, not the heading style at the insertion point.
    template = orig[373]
    p.append(deepcopy(template.find(W+'pPr')))
    run = E.SubElement(p,W+'r')
    rp = template.find('w:r/w:rPr',N)
    if rp is not None:
        run.append(deepcopy(rp))
    E.SubElement(run,W+'t').text=s
    body.insert(body.index(target),p)
    changes.append({'action':'补充概括导读','before_index':i,'after':s})

# These checks detect destructive editing of the protected manuscript content.
assert Counter(E.tostring(p) for p in root.xpath('//m:oMath',namespaces=N)) == Counter(E.tostring(p) for p in old.xpath('//m:oMath',namespaces=N))
assert [txt(p) for p in list(old.find('w:body',N))[:256]] == [txt(p) for p in list(body)[:256]]
assert root.xpath('//w:instrText/text()',namespaces=N).count('SEQ Eq2 \\* ARABIC') == old.xpath('//w:instrText/text()',namespaces=N).count('SEQ Eq2 \\* ARABIC')
starts = {p.get(W+'id'):j for j,p in enumerate(root.iter()) if p.tag == W+'bookmarkStart'}
for j,p in enumerate(root.iter()):
    if p.tag == W+'bookmarkEnd':
        assert starts[p.get(W+'id')] <= j, ('reversed bookmark',p.get(W+'id'))
with zipfile.ZipFile(OUT,'w') as out:
    for info in z.infolist():
        out.writestr(info,E.tostring(root,xml_declaration=True,encoding='UTF-8',standalone=True)
                     if info.filename == 'word/document.xml' else z.read(info.filename))
(WORK/'目录更新前.docx').write_bytes(OUT.read_bytes())
(WORK/'修改记录.json').write_text(json.dumps(changes,ensure_ascii=False,indent=2))
print(json.dumps({'output':str(OUT),'prose_paragraphs':len(plan['EDITS']),
                  'new_bridges':len(plan['INSERT_BEFORE']),'changes':len(changes)},ensure_ascii=False))
