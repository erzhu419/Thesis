"""Add public implementation links without rebuilding the current Word file."""
import copy
import json
import re
import shutil
from pathlib import Path
from zipfile import ZipFile

from lxml import etree as E

ROOT = Path('/home/erzhu419/mine_code/Thesis')
WORK = Path(__file__).resolve().parent
SOURCE = ROOT / 'output/排版完成_章内留白压缩_20261007.docx'
OUTPUT = ROOT / 'output/排版完成_自有论文与仓库补齐_20261007.docx'
BACKUP = WORK / '修改前.docx'
W = 'http://schemas.openxmlformats.org/wordprocessingml/2006/main'
R = 'http://schemas.openxmlformats.org/officeDocument/2006/relationships'
REL = 'http://schemas.openxmlformats.org/package/2006/relationships'
NS = {'w': W, 'r': R}
q = lambda name: '{' + W + '}' + name

assert not OUTPUT.exists()
if BACKUP.exists():
    assert BACKUP.read_bytes() == SOURCE.read_bytes()
else:
    shutil.copy2(SOURCE, BACKUP)
with ZipFile(BACKUP) as archive:
    infos = archive.infolist()
    parts = {item.filename: archive.read(item.filename) for item in infos}
root = E.fromstring(parts['word/document.xml'])
body = root.find('w:body', NS)
original = copy.deepcopy(body)
relationships = E.fromstring(parts['word/_rels/document.xml.rels'])
rid = max(int(n.get('Id')[3:]) for n in relationships if re.fullmatch(r'rId\d+', n.get('Id', ''))) + 1
bid = max(int(n.get(q('id'))) for n in root.findall('.//w:bookmarkStart', NS)) + 1


def text(node):
    return ''.join(node.xpath('.//w:t/text()', namespaces=NS))


def style(node):
    s = node.find('w:pPr/w:pStyle', NS)
    return s.get(q('val')) if s is not None else ''


def signature(node):
    return (node.tag, tuple(sorted(node.attrib.items())), node.text, node.tail,
            tuple(signature(child) for child in node))


def run(value, superscript=False):
    node = E.Element(q('r'))
    if superscript:
        E.SubElement(E.SubElement(node, q('rPr')), q('vertAlign'), {q('val'): 'superscript'})
    t = E.SubElement(node, q('t'))
    t.text = value
    if value != value.strip():
        t.set('{http://www.w3.org/XML/1998/namespace}space', 'preserve')
    return node


def url(value):
    global rid
    node = E.Element(q('hyperlink'), {'{' + R + '}id': f'rId{rid}', q('history'): '1'})
    node.append(run(value))
    E.SubElement(relationships, '{' + REL + '}Relationship', Id=f'rId{rid}',
                 Type=R + '/hyperlink', Target=value, TargetMode='External')
    rid += 1
    return node


def citation(number, bookmark):
    link = E.Element(q('hyperlink'), {q('anchor'): bookmark, q('history'): '1'})
    link.append(run(str(number), True))
    return [run('[', True), link, run(']', True)]


projects = [
    ('类别特征嵌入共享公交控制', 'Categorical-Feature-SAC-in-bus-simulation', 210, 'ref-categorical-code'),
    ('RE-SAC', 'RE-SAC', 211, 'ref-resac-code'),
    ('BAPR', 'BAPR', 212, 'ref-bapr-code'),
    ('公交H2O+', 'sumo-rl', 213, 'ref-bus-h2o-code'),
    ('MC-WM', 'MC-WM', 214, 'ref-mcwm-code'),
    ('CS-BAPR', 'CS-BAPR', 159, 'ref-csbapr-code'),
]
by_name = {item[0]: item for item in projects}
chapter_projects = {3: [projects[0][0], 'RE-SAC'], 4: ['BAPR', 'CS-BAPR'],
                    5: ['公交H2O+'], 6: ['MC-WM']}
notes = []
for chapter, names in chapter_projects.items():
    heading = next(p for p in body if style(p) == 'chapter' and text(p).startswith(f'第{chapter}章 '))
    index = body.index(heading) + 1
    while style(body[index]) == 'body':
        index += 1
    assert style(body[index]) == 'h2'
    p = E.Element(q('p'))
    p.append(copy.deepcopy(body[index - 1].find('w:pPr', NS)))
    p.append(run('本章相关研究的公开代码仓库为：'))
    for j, name in enumerate(names):
        _, repo, number, bookmark = by_name[name]
        if j:
            p.append(run('；'))
        p.append(run(name + '，'))
        p.append(url('https://github.com/erzhu419/' + repo))
        p.extend(citation(number, bookmark))
    p.append(run('。'))
    body.insert(index, p)
    notes.append({'chapter': chapter, 'text': text(p), 'repositories': names})

bibliography = [p for p in body if style(p) == 'bibliography']
assert len(bibliography) == 209 and text(bibliography[-1]).startswith('[209] ')
index = body.index(bibliography[-1]) + 1
for j, (name, repo, number, bookmark) in enumerate(projects[:5]):
    p = E.Element(q('p'))
    p.append(copy.deepcopy(bibliography[-1].find('w:pPr', NS)))
    E.SubElement(p, q('bookmarkStart'), {q('id'): str(bid + j), q('name'): bookmark})
    title = repo if repo != 'sumo-rl' else 'sumo-rl: bus holding implementation'
    p.append(run(f'[{number}] ZHANG Y. {title}: source code[CP/OL]. GitHub[2026-10-07]. '))
    p.append(url('https://github.com/erzhu419/' + repo))
    p.append(run('.'))
    E.SubElement(p, q('bookmarkEnd'), {q('id'): str(bid + j)})
    body.insert(index + j, p)

# Only nine new paragraphs are authorized. All pre-existing nodes remain intact.
old_iter = iter(body)
for old in original:
    for current in old_iter:
        if signature(old) == signature(current):
            break
    else:
        raise AssertionError('An existing body node was modified or reordered.')
assert len(body) == len(original) + 9
for _, _, number, bookmark in projects:
    assert len(root.xpath('//w:bookmarkStart[@w:name=$name]', namespaces=NS, name=bookmark)) == 1
    assert root.xpath('//w:hyperlink[@w:anchor=$name]', namespaces=NS, name=bookmark)
parts['word/document.xml'] = E.tostring(root, xml_declaration=True, encoding='UTF-8', standalone=True)
parts['word/_rels/document.xml.rels'] = E.tostring(relationships, xml_declaration=True, encoding='UTF-8', standalone=True)
with ZipFile(OUTPUT, 'w') as archive:
    for item in infos:
        archive.writestr(item, parts[item.filename])
(WORK / '补充位置.json').write_text(json.dumps(notes, ensure_ascii=False, indent=2), encoding='utf-8')
print(json.dumps({'output': str(OUTPUT), 'chapter_notes': notes, 'references': 214,
                  'all_original_body_nodes_preserved': True}, ensure_ascii=False, indent=2))
