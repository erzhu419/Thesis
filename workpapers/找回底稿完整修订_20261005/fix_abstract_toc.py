from pathlib import Path
from zipfile import ZipFile
from lxml import etree as E
import json
w=Path(__file__).resolve().parent;t=w.parents[1]/'output/排版完成_导师意见复核_20261005.docx'
n={'w':'http://schemas.openxmlformats.org/wordprocessingml/2006/main'}
with ZipFile(t) as z: infos=z.infolist();data={i.filename:z.read(i) for i in infos}
r=E.fromstring(data['word/document.xml']);ps=r.find('w:body',n).findall('w:p',n)
locked=[E.tostring(p) for p in ps[61:80]]
m={'_Toc242125967':'_Toc242108809','_Toc242125968':'_Toc242108810'}
changes=0
for p in ps[80:]:
 for h in p.xpath('.//w:hyperlink',namespaces=n):
  k=h.get('{'+n['w']+'}anchor')
  if k in m:h.set('{'+n['w']+'}anchor',m[k]);changes+=1
 for i in p.xpath('.//w:instrText',namespaces=n):
  for old,new in m.items():
   if old in (i.text or ''):i.text=i.text.replace(old,new);changes+=1
assert changes==4
assert [E.tostring(p) for p in ps[61:80]]==locked
bookmarks=set(r.xpath('//w:bookmarkStart/@w:name',namespaces=n))
assert not [a for a in r.xpath('//w:hyperlink/@w:anchor',namespaces=n) if a not in bookmarks]
data['word/document.xml']=E.tostring(r,encoding='UTF-8',xml_declaration=True,standalone=True)
with ZipFile(t,'w') as z:
 for i in infos:z.writestr(i,data[i.filename])
(w/'摘要目录链接修正.json').write_text(json.dumps({'status':'pass','corrected_anchor_and_PAGEREF':m,'abstract_xml_unchanged':True,'all_internal_hyperlink_targets_exist':True},ensure_ascii=False,indent=2))
print('Two abstract TOC links repaired; exact abstract XML preserved.')
