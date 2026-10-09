from pathlib import Path
from zipfile import ZipFile
from lxml import etree as E
from collections import Counter
import json
out=Path('/home/erzhu419/mine_code/Thesis/排版完成_接受作者修订_全文统一_20261006.docx')
with ZipFile(out) as z:infos=z.infolist();data={i.filename:z.read(i.filename) for i in infos}
r=E.fromstring(data['word/document.xml']);ns={'w':r.nsmap['w']};b=r.find('w:body',ns)
txt=lambda n:''.join(n.xpath('.//w:t/text()',namespaces=ns))
before=Counter(E.tostring(n) for n in b)
cap=next(n for n in b if txt(n).startswith('图2-3 跨保真'));fig=cap.getprevious();assert fig.xpath('.//w:drawing',namespaces=ns)
anchor=next(n for n in b if txt(n)=='2.4.2 状态动作偏移与条件转移偏移')
for n in [fig,cap]:b.remove(n);b.insert(b.index(anchor),n)
cap=next(n for n in b if txt(n).startswith('图3-3 早期'));fig=cap.getprevious();assert fig.xpath('.//w:drawing',namespaces=ns)
anchor=next(n for n in b if txt(n).startswith('图3-3从空间分布'))
for n in [fig,cap]:b.remove(n);b.insert(b.index(anchor),n)
assert Counter(E.tostring(n) for n in b)==before
# Body prose and mathematical content retain their order; only two complete figure blocks moved.
data['word/document.xml']=E.tostring(r,xml_declaration=True,encoding='UTF-8',standalone=True)
with ZipFile(out,'w') as z:
 for i in infos:z.writestr(i,data[i.filename])
print('Moved two intact figure/caption blocks; all body elements retained byte-for-byte.')
