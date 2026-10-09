from pathlib import Path
from zipfile import ZipFile
from lxml import etree as E
from copy import deepcopy
import re
ROOT=Path('/home/erzhu419/mine_code/Thesis');WORK=ROOT/'workpapers/作者审阅接受与批注解释_20261006'
SRC=WORK/'仅接受作者修订.docx';OUT=ROOT/'排版完成_接受作者修订_全文统一_20261006.docx'
NS={'w':'http://schemas.openxmlformats.org/wordprocessingml/2006/main','a':'http://schemas.openxmlformats.org/drawingml/2006/main','r':'http://schemas.openxmlformats.org/officeDocument/2006/relationships','wp':'http://schemas.openxmlformats.org/drawingml/2006/wordprocessingDrawing'}
with ZipFile(SRC) as z:
 sr=E.fromstring(z.read('word/document.xml'));sb=sr.find('w:body',NS)
 cap=next(n for n in sb if ''.join(n.xpath('.//w:t/text()',namespaces=NS))=='图2-3 跨保真代理验证与真实运营应用的数据角色')
 fig=cap.getprevious();oldid=fig.xpath('.//a:blip/@r:embed',namespaces=NS)[0]
 rel=E.fromstring(z.read('word/_rels/document.xml.rels'));target=next(n.get('Target') for n in rel if n.get('Id')==oldid)
 media=z.read('word/'+target);fig=deepcopy(fig);cap=deepcopy(cap)
with ZipFile(OUT) as z:infos=z.infolist();data={i.filename:z.read(i.filename) for i in infos}
r=E.fromstring(data['word/document.xml']);b=r.find('w:body',NS);rels=E.fromstring(data['word/_rels/document.xml.rels'])
assert len(r.xpath('//w:drawing',namespaces=NS))==62
# Insertion anchor is an exact body heading, excluding the cached TOC entry.
anchor=next(n for n in b if ''.join(n.xpath('.//w:t/text()',namespaces=NS))=='2.4.2 状态动作偏移与条件转移偏移')
assert anchor.find('w:pPr/w:pStyle',NS).get('{'+NS['w']+'}val')=='h3'
for part in [fig,cap]:
 for n in part.xpath('.//w:bookmarkStart|.//w:bookmarkEnd',namespaces=NS):n.getparent().remove(n)
 for n in part.iter():
  for key in list(n.attrib):
   if key in ['{http://schemas.microsoft.com/office/word/2010/wordml}paraId','{http://schemas.microsoft.com/office/word/2010/wordml}textId']:del n.attrib[key]
maxrel=max(int(n.get('Id')[3:]) for n in rels if re.fullmatch(r'rId\d+',n.get('Id','')))
newid='rId'+str(maxrel+1)
for n in fig.xpath('.//a:blip',namespaces=NS):n.set('{'+NS['r']+'}embed',newid)
ids=[int(n.get('id')) for n in r.xpath('//wp:docPr',namespaces=NS)]
for n in fig.xpath('.//wp:docPr',namespaces=NS):n.set('id',str(max(ids)+1));n.set('name','图2-3 公交跨保真数据角色')
rel=E.SubElement(rels,'{http://schemas.openxmlformats.org/package/2006/relationships}Relationship');rel.set('Id',newid);rel.set('Type',NS['r']+'/image');rel.set('Target','media/author_review_fig2_3.png')
for n in [fig,cap]:b.insert(b.index(anchor),n)
data['word/document.xml']=E.tostring(r,xml_declaration=True,encoding='UTF-8',standalone=True)
data['word/_rels/document.xml.rels']=E.tostring(rels,xml_declaration=True,encoding='UTF-8',standalone=True)
with ZipFile(OUT,'w') as z:
 for i in infos:z.writestr(i,data[i.filename])
 z.writestr('word/media/author_review_fig2_3.png',media,compress_type=8)
print('All 63 figures restored; Figure 2-3 is anchored before the exact body heading 2.4.2.')
