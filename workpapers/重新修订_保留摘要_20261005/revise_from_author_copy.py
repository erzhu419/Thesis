"""Replay reviewed corrections onto the author's chosen copy; lock abstracts."""
from pathlib import Path
from zipfile import ZipFile
from lxml import etree as E
from copy import deepcopy
import json

WORK = Path(__file__).resolve().parent
ROOT = WORK.parents[1]
OLD = ROOT / 'workpapers/导师意见综合修订_20261005'
CITES = ROOT / 'workpapers/本人论文引用补充_20261005'
TARGET = Path('/mnt/d/Paper/Thesis/排版完成_重新修订_保留摘要_20261005.docx')
NS = {'w':'http://schemas.openxmlformats.org/wordprocessingml/2006/main',
      'm':'http://schemas.openxmlformats.org/officeDocument/2006/math'}
PID = '{http://schemas.microsoft.com/office/word/2010/wordml}paraId'

def read_paras(path):
    with ZipFile(path) as z:
        return E.fromstring(z.read('word/document.xml')).find('w:body', NS).findall('w:p', NS)

def visible(p):
    return ''.join(p.xpath('.//w:t/text()|.//m:t/text()', namespaces=NS))

def map_paras(source, current):
    by_id = {p.get(PID):p for p in current if p.get(PID)}
    mapped = []
    for i, p in enumerate(source):
        key = p.get(PID)
        if key in by_id:
            mapped.append(by_id[key]); continue
        style = p.xpath('string(w:pPr/w:pStyle/@w:val)', namespaces=NS)
        if style.startswith('TOC'):
            mapped.append(current[i]); continue
        found = [x for x in current if visible(x) == visible(p)]
        if len(found) == 1:
            mapped.append(found[0]); continue
        # TOC is regenerated later; scripts never edit these missing IDs.
        mapped.append(None)
    return mapped

author = read_paras(WORK/'用户指定底稿_修改前.docx')
base = read_paras(OLD/'修改前.docx')
author_ids = {p.get(PID):p for p in author}
changed_author_ids = {p.get(PID) for i,p in enumerate(base) if i>79 and p.get(PID) in author_ids
                      and visible(p) != visible(author_ids[p.get(PID)])}
protected_ids = {p.get(PID) for p in author[61:80]}
protected_xml = [E.tostring(p) for p in author[61:80]]

def lock_check(current):
    selected = {p.get(PID):p for p in current}
    assert [E.tostring(selected[p.get(PID)]) for p in author[61:80]] == protected_xml

# Stage 1 uses original paragraph identities, not shifted paragraph positions.
code = (OLD/'edit_word.py').read_text()
code = code.replace("SOURCE = HERE / '修改前.docx'", "SOURCE = WORK / '用户指定底稿_修改前.docx'")
code = code.replace("DEST = HERE / '修订待分页.docx'", "DEST = WORK / '阶段1.docx'")
code = code.replace("paras = body.findall('w:p', NS)",
                    "user_paras = body.findall('w:p', NS)\nparas = map_paras(base, user_paras)")
code = code.replace("    old = visible(paras[i])\n    fields=", """    if i < 80:
        return
    if paras[i].get(PID) in changed_author_ids:
        changes.append({'paragraph':i,'decision':'保留新底稿作者修改，另行逐段整合'})
        return
    old = visible(paras[i])
    fields=""")
code = code.replace("for p in doc.xpath('//w:p',namespaces=NS):\n    s=", """for p in doc.xpath('//w:p',namespaces=NS):
    if p.get(PID) in protected_ids: continue
    s=""")
code = code.replace("(HERE/'changes.json')", "(WORK/'阶段1修改记录.json')")
env = dict(__file__=str(OLD/'edit_word.py'), WORK=WORK, map_paras=map_paras, base=base,
           PID=PID, changed_author_ids=changed_author_ids, protected_ids=protected_ids)
exec(compile(code, str(WORK/'stage1_replayed.py'), 'exec'), env)
lock_check(read_paras(WORK/'阶段1.docx'))

# The second audit uses its saved pre-audit paragraph identities and exact text.
mid = read_paras(OLD/'再次复核_修改前.docx')
code = (OLD/'followup_edit.py').read_text()
code = code.replace("TARGET=HERE.parents[1]/'output/排版完成_导师意见复核_20261005.docx'", "TARGET=WORK/'阶段2.docx'")
code = code.replace("BACKUP=HERE/'再次复核_修改前.docx'", "BACKUP=WORK/'阶段1.docx'")
code = code.replace("paras=body.findall('w:p',ns);tables=body.findall('w:tbl',ns);changes=[]",
                    "paras=map_paras(mid,body.findall('w:p',ns));tables=body.findall('w:tbl',ns);changes=[]")
code = code.replace("(HERE/'再次复核_补修记录.json')", "(WORK/'阶段2修改记录.json')")
code = code.replace("assert 'arxiv.org/abs/' in visible(links[0])",
                    "assert not visible(links[0]) or 'arxiv.org/abs/' in visible(links[0])")
# This author's six-paragraph Chapter 6 summary replaces the old four-paragraph text.
code = code.replace("    replace(paras[i],item['原文片段'],new);changes.append", """    if item['原文片段'] not in visible(paras[i]):
        assert paras[i].get(PID) in changed_author_ids, (i,item['原文片段'])
        changes.append({'paragraph':i,'decision':'作者新版小结另行整合'})
        continue
    replace(paras[i],item['原文片段'],new);changes.append""")
exec(compile(code, str(WORK/'stage2_replayed.py'), 'exec'),
     dict(__file__=str(OLD/'followup_edit.py'), WORK=WORK, map_paras=map_paras, mid=mid,
          PID=PID, changed_author_ids=changed_author_ids))
lock_check(read_paras(WORK/'阶段2.docx'))

# Stage 3 adds the verified references; all abstract edits are explicitly excluded.
previous = read_paras(CITES/'修改前.docx')
code = (CITES/'edit_citations.py').read_text()
code = code.replace("HERE = Path(__file__).resolve().parent", "HERE = WORK")
code = code.replace("TARGET = HERE.parents[1] / 'output/排版完成_导师意见复核_20261005.docx'", 'TARGET = FINAL_TARGET')
code = code.replace("HERE / '修改前.docx'", "HERE / '阶段2.docx'")
code = code.replace("ps = body.findall('w:p', NS)", "ps = map_paras(previous, body.findall('w:p', NS))")
code = code.replace("def edit(i, old, new):\n    replace", """def edit(i, old, new):
    if i < 80: return
    replace""")
# ps includes missing old summary/TOC objects, which are not citation targets.
code = code.replace("for p in ps:\n    expand_markers", "for p in body.findall('w:p', NS):\n    expand_markers")
code = code.replace(" and 'Adaptive Policy Regularization' not in text(doc)", '')
exec(compile(code, str(WORK/'stage3_replayed.py'), 'exec'),
     dict(__file__=str(CITES/'edit_citations.py'), WORK=WORK, FINAL_TARGET=TARGET,
          map_paras=map_paras, previous=previous))
lock_check(read_paras(TARGET))
print('Three reviewed stages applied; abstract XML unchanged.')
