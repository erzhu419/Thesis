"""Retain the value and iteration definitions when expanding the fixed-point proof."""
from copy import deepcopy
import json
from zipfile import ZipFile, ZIP_DEFLATED
from lxml import etree as E
from edit_word import BASE, HERE, NS, content, prose

with ZipFile(BASE) as z:
    parts = {n: z.read(n) for n in z.namelist()}
doc = E.fromstring(parts['word/document.xml'])
body = doc.find('w:body', NS)
p = next(n for n in body if content(n).startswith('固定策略下，先对动作概率求和'))
with ZipFile(HERE / '排版完成_修改前.docx') as z:
    original = list(E.fromstring(z.read('word/document.xml')).find('w:body', NS))[3003]
maths = original.xpath('.//m:oMath', namespaces=NS)
symbols = dict(zip(['fixedvalue', 'initialvalue', 'valueiteration'], maths[1:4]))
text = '固定策略下，先对动作概率求和，得到状态转移核与期望即时奖励。策略和转移核均非负且归一化，因此该状态核也非负且每行和为1，期望奖励仍受式(E-2)约束。价值函数空间在最大范数下完备；由定理E.1及 {{γ}}<1，收缩映射不动点定理给出唯一固定点 {{fixedvalue}}。从任意初值 {{initialvalue}} 出发，令 {{valueiteration}}。连续两次迭代的差由折扣因子的幂控制，后续差值之和可用收敛几何级数界定，故迭代构成柯西序列。其极限满足固定点方程；再将每次迭代与固定点比较，得到'
prose(p, text, symbols)
parts['word/document.xml'] = E.tostring(doc, xml_declaration=True, encoding='UTF-8', standalone=True)
with ZipFile(BASE, 'w', ZIP_DEFLATED) as z:
    for name, data in parts.items():
        z.writestr(name, data)
records = json.loads((HERE / 'word_changes.json').read_text())
next(r for r in records if r['old_block'] == 3003)['new'] = text
(HERE / 'word_changes.json').write_text(json.dumps(records, ensure_ascii=False, indent=2))
print(content(p))
