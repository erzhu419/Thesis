from pathlib import Path
from zipfile import ZipFile
from lxml import etree as E
from collections import Counter
import json,unicodedata,sys
from verify_content import q,local,NS,first_difference
ROOT=Path(__file__).resolve().parents[2]
def canonical(e):
 if e.tag in {q('w','rPr'),q('m','ctrlPr'),q('m','argSz')}:return []
 if e.tag==q('m','r'):
  rp=e.find('m:rPr',NS);p={local(v):v.get(q('m','val'),'1') for v in rp} if rp is not None else {}
  txt=''.join(e.xpath('./m:t/text()',namespaces=NS));tokens=[]
  for ch in txt:
   cp=dict(p)
   if not ch.isalpha() and not ch.isdigit():
    cp.pop('scr',None)
    if cp.get('sty','i') in ['i','p']:cp.pop('sty',None)
   tokens.append(['math_char',ch,sorted(cp.items())])
  return tokens
 children=[v for c in e for v in canonical(c)]
 attrs=sorted((k,v) for k,v in e.attrib.items() if E.QName(k).localname not in {'rsidR','rsidRPr'})
 return [[e.tag,attrs,e.text if e.tag in {q('m','t'),q('w','t')} else None,children]]
a=E.fromstring(ZipFile(ROOT/'output/博士论文_批注核对补充稿_20261002.docx').read('word/document.xml')).findall('.//m:oMath',NS)
b=E.fromstring(ZipFile(ROOT/'output/排版完成.docx').read('word/document.xml')).findall('.//m:oMath',NS)
bad=[]
for i,(x,y) in enumerate(zip(a,b)):
 ca=canonical(x);cb=canonical(y)
 if ca!=cb:bad.append({'index':i,'source_text':''.join(x.xpath('.//m:t/text()',namespaces=NS)),'candidate_text':''.join(y.xpath('.//m:t/text()',namespaces=NS)),'difference':first_difference(ca,cb),'source_projection':ca,'candidate_projection':cb})
Path(__file__).with_name('math_normalization_remaining.json').write_text(json.dumps(bad,ensure_ascii=False,indent=2))
print('remaining',len(bad))
for d in bad[:15]:print(d['index'],d['source_text'][:120],d['difference'])
