from pathlib import Path
from zipfile import ZipFile
from lxml import etree
from collections import Counter
import json,re,unicodedata
A=Path(__file__).resolve().parent
flags=[];counts=Counter(); scripts=Counter(); unique={}
W='http://schemas.openxmlformats.org/wordprocessingml/2006/main';M='http://schemas.openxmlformats.org/officeDocument/2006/math'
rarehan=set('観図実関転変値証仮経誤広応両畳乗単覚繋為與於則將對學數類際轉態機標隨餘體級價觀測內無擬證穩權據')
with ZipFile(A/'source_snapshot.docx') as z:
 for part in z.namelist():
  if not part.endswith('.xml') or not part.startswith('word/'):continue
  try:root=etree.fromstring(z.read(part))
  except etree.XMLSyntaxError:continue
  for n in root.iter():
   if etree.QName(n).localname not in ('t','instrText'):continue
   s=n.text or '';counts[etree.QName(n).localname]+=1
   for c in s:
    u=ord(c);category=None
    if 0x3040<=u<=0x30ff or 0x31f0<=u<=0x31ff or 0xff66<=u<=0xff9f:category='Japanese kana'
    elif 0x1100<=u<=0x11ff or 0x3130<=u<=0x318f or 0xa960<=u<=0xa97f or 0xac00<=u<=0xd7af or 0xd7b0<=u<=0xd7ff:category='Hangul'
    elif c in '\ufffd\u25a1\ufffc':category='replacement or square'
    elif 0xe000<=u<=0xf8ff:category='private use'
    elif c in rarehan:category='Han variant context'
    elif 0x400<=u<=0x52f or 0x590<=u<=0x8ff:category='unusual script'
    if category:
     p=n
     while p is not None and p.tag!=f'{{{W}}}p':p=p.getparent()
     ctx=''.join(p.itertext()) if p is not None else s
     item={'part':part,'category':category,'char':c,'code':f'U+{u:04X}','context':ctx[:1000]}
     key=(part,category,c,ctx)
     if key not in unique:unique[key]=item
     scripts[category]+=1
   if n.tag==f'{{{W}}}t' and re.search(r'\\(?:frac|alpha|beta|begin|end|mathrm|tilde)|\$[^$]+\$|&(?:nbsp|amp|lt|gt);',s):
    flags.append({'part':part,'category':'possible raw markup','text':s})
 result={'text_nodes':dict(counts),'categories':dict(scripts),'findings':list(unique.values())+flags}
(A/'character_scan.json').write_text(json.dumps(result,ensure_ascii=False,indent=2))
print(json.dumps({'text_nodes':dict(counts),'categories':dict(scripts),'findings':len(result['findings'])},ensure_ascii=False))
