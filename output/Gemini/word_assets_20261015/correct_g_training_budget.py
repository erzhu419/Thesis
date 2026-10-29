"""Correct the G training-budget clause explicitly approved by the author."""
from pathlib import Path
from zipfile import ZipFile, ZIP_DEFLATED
from lxml import etree as E
from copy import deepcopy
import json, pypandoc

A = Path(__file__).resolve().parent
TARGET = A/'通篇细节检查_工作稿.docx'
W = '{http://schemas.openxmlformats.org/wordprocessingml/2006/main}'
M = '{http://schemas.openxmlformats.org/officeDocument/2006/math}'
prefix = ('。在5个独立训练随机种子下，MC-WM与H2O+在每个对应种子下共用完成'
          '$50,000$次IQL离线更新的初始化，随后继续训练$20,000$次；纯源SAC从随机网络参数开始，'
          '仅使用源仿真经验训练$20,000$次。在各方案均采集约$25,000$次源仿真决策事件'
          '（MC-WM与H2O+各累计$70,000$次梯度更新，纯源SAC累计$20,000$次，源样本权重置')
fragment = A/'qa_g_budget_fragment.docx'
pypandoc.convert_text(prefix, 'docx', format='markdown+tex_math_dollars-smart', outputfile=str(fragment))
with ZipFile(fragment) as z:
    fr = E.fromstring(z.read('word/document.xml')).find(W+'body').find(W+'p')
text = lambda e: ''.join(n.text or '' for n in e.iter() if n.tag in (W+'t', M+'t'))
with ZipFile(TARGET) as z:
    entries = {n: z.read(n) for n in z.namelist()}
r = E.fromstring(entries['word/document.xml'])
p, = [p for p in r.find(W+'body') if text(p).startswith('（五）独立源经验序列')]
nodes = list(p)
assert text(nodes[6]) == '。在' and text(nodes[22]) == '次梯度更新，源样本权重置'
before_tail = b''.join(E.tostring(n) for n in nodes[23:])
for n in nodes[6:23]: p.remove(n)
at = 6
for n in fr:
    if n.tag == W+'pPr': continue
    p.insert(at, deepcopy(n)); at += 1
assert b''.join(E.tostring(n) for n in list(p)[at:]) == before_tail
entries['word/document.xml'] = E.tostring(r, encoding='UTF-8', xml_declaration=True, standalone=True)
with ZipFile(TARGET, 'w', ZIP_DEFLATED) as z:
    for name, data in entries.items(): z.writestr(name, data)
md = A.parent/'重构论文.md'; source = md.read_text()
old = ('。在5个独立训练随机种子各自独立执行 $50,000$ 次IQL离线初始化更新并采集 '
       '$25,000$ 次源仿真决策事件（总计 $70,000$ 次梯度更新，源样本权重置')
assert source.count(old) == 1
md.write_text(source.replace(old, prefix))
changes = json.loads((A/'qa_applied_changes.json').read_text())
changes.append({'type': 'author_approved_G_protocol_correction', 'offline_updates': {'MC-WM': 50000, 'H2O+': 50000, 'SAC': 0}, 'continuation_updates': 20000, 'evidence': ['qa_sac_initialization_evidence.json', 'qa_sac_training_contract_evidence.json']})
(A/'qa_applied_changes.json').write_text(json.dumps(changes, ensure_ascii=False, indent=2))
print('Approved G budget corrected; all text and native objects after the budget clause preserved.')
