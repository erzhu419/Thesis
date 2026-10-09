import json,copy,re
from pathlib import Path
from zipfile import ZipFile,ZIP_DEFLATED
from lxml import etree as E

P=Path(__file__).parent
SRC=P/'修改前.docx'
OUT=P.parent.parent/'排版完成_综述补充50篇中文文献_20261006.docx'
W='http://schemas.openxmlformats.org/wordprocessingml/2006/main'
R='http://schemas.openxmlformats.org/officeDocument/2006/relationships'
REL='http://schemas.openxmlformats.org/package/2006/relationships'
NS={'w':W,'r':R,'m':'http://schemas.openxmlformats.org/officeDocument/2006/math'}
q=lambda s:'{'+W+'}'+s
def txt(e):return ''.join(e.xpath('.//w:t/text()',namespaces=NS))
def sig(e):
 return (e.tag,tuple(sorted(e.attrib.items())),e.text,e.tail,tuple(sig(c) for c in e)) if e is not None else None
def run(s,super_=False):
 r=E.Element(q('r'))
 if super_:
  pr=E.SubElement(r,q('rPr'));E.SubElement(pr,q('vertAlign'),{q('val'):'superscript'})
 t=E.SubElement(r,q('t'));t.text=s
 if s!=s.strip():t.set('{http://www.w3.org/XML/1998/namespace}space','preserve')
 return r
refs=json.loads((P/'references.json').read_text())
groups=json.loads((P/'insertions.json').read_text())
with ZipFile(SRC) as z:parts={i.filename:z.read(i.filename) for i in z.infolist()}; infos=z.infolist()
root=E.fromstring(parts['word/document.xml']);body=root.find('w:body',NS)
original=copy.deepcopy(body)
relationships=E.fromstring(parts['word/_rels/document.xml.rels'])
rid=max(int(n.get('Id')[3:]) for n in relationships if re.match(r'rId\d+$',n.get('Id','')))+1
bid=max(int(n.get(q('id'))) for n in root.findall('.//w:bookmarkStart',NS))+1

def citation_nodes(numbers,merged=False):
 nodes=[run(',' if merged else '[',True)]
 for j,num in enumerate(numbers):
  if j:nodes.append(run(',',True))
  link=E.Element(q('hyperlink'),{q('anchor'):f'ref-cn2026-{num}',q('history'):'1'})
  link.append(run(str(num),True));nodes.append(link)
 if not merged:nodes.append(run(']',True))
 return nodes

changed=set();inserted_ids=set()
for g in groups:
 p=body[g['paragraph']];before=txt(p);anchor=g['anchor'];assert before.count(anchor)==1,(g['paragraph'],anchor)
 end=before.index(anchor)+len(anchor);offset=0
 for t in p.xpath('.//w:t',namespaces=NS):
  n=len(t.text or '')
  if offset < end <= offset+n:
   r=t.getparent();assert r.tag==q('r') and r.getparent() is p
   assert all(x.tag in [q('rPr'),q('t'),q('lastRenderedPageBreak')] for x in r)
   pos=end-offset;tail=t.text[pos:];t.text=t.text[:pos];idx=p.index(r)+1
   if tail:
    trailing=E.Element(q('r'),r.attrib)
    rp=r.find('w:rPr',NS)
    if rp is not None:trailing.append(copy.deepcopy(rp))
    nt=E.SubElement(trailing,q('t'));nt.text=tail
    if tail!=tail.strip():nt.set('{http://www.w3.org/XML/1998/namespace}space','preserve')
    p.insert(idx,trailing)
   break
  offset+=n
 else:raise AssertionError('Anchor end not found')
 # Merge additions into the already-present [40] group at this one location.
 merged=False
 if not tail and idx<len(p) and txt(p[idx])=='[':
  close=next(j for j in range(idx+1,len(p)) if txt(p[j])==']')
  assert g['paragraph']==289 and ''.join(txt(n) for n in list(p)[idx:close+1])=='[40]'
  idx=close;merged=True
 nodes=citation_nodes(g['numbers'],merged)
 for j,node in enumerate(nodes):p.insert(idx+j,node);inserted_ids.add(id(node))
 g['insert_text']=(',' if merged else '[')+','.join(map(str,g['numbers']))+('' if merged else ']')
 g['merged_with_existing']=merged
 expected=before.replace('[40]', '[40'+g['insert_text']+']') if merged else before[:end]+g['insert_text']+before[end:]
 assert txt(p)==expected
 changed.add(g['paragraph'])

bib_idx=[i for i,p in enumerate(body) if re.match(r'^\[159\] ',txt(p))]
assert len(bib_idx)==1
insert_at=bib_idx[0]+1
template=body[1980].find('w:pPr',NS)
assert template.find('w:pStyle',NS).get(q('val'))=='bibliography'
for j,d in enumerate(refs):
 p=E.Element(q('p'));pr=copy.deepcopy(template)
 # Keep each new entry together and retain the existing bibliography font/indent.
 keep=E.Element(q('keepLines'));pr.insert(1,keep);p.append(pr)
 E.SubElement(p,q('bookmarkStart'),{q('id'):str(bid+j),q('name'):d['bookmark']})
 p.append(run(f"[{d['number']}] "));tab=E.SubElement(p,q('r'));E.SubElement(tab,q('tab'))
 if d['doi']:
  prefix=d['entry'].split(' DOI:')[0]+' DOI:';p.append(run(prefix))
  relation=E.SubElement(relationships,'{'+REL+'}Relationship',Id=f'rId{rid}',Type=R+'/hyperlink',Target='https://doi.org/'+d['doi'],TargetMode='External')
  h=E.SubElement(p,q('hyperlink'),{'{'+R+'}id':f'rId{rid}'});h.append(run(d['doi']));p.append(run('.'));rid+=1
 else:p.append(run(d['entry']))
 E.SubElement(p,q('bookmarkEnd'),{q('id'):str(bid+j)})
 body.insert(insert_at+j,p)

# Check exactly what this edit is authorized to change.
assert len(body)==len(original)+50
for i,p in enumerate(original):
 now=body[i if i<insert_at else i+50]
 if i in changed:
  g=next(g for g in groups if g['paragraph']==i)
  assert txt(now).replace(g['insert_text'],'',1)==txt(p)
  assert sig(now.find('w:pPr',NS))==sig(p.find('w:pPr',NS))
 else:assert sig(now)==sig(p),f'Unrelated paragraph changed: {i}'
for tag in ['oMath','oMathPara']:
 assert [sig(n) for n in body.findall('.//m:'+tag,NS)]==[sig(n) for n in original.findall('.//m:'+tag,NS)]
assert len(root.findall('.//w:hyperlink',NS))>=50
names=[n.get(q('name')) for n in root.findall('.//w:bookmarkStart',NS)]
for d in refs:
 assert names.count(d['bookmark'])==1
 links=root.xpath('//w:hyperlink[@w:anchor=$a]',namespaces=NS,a=d['bookmark'])
 assert len(links)==1 and txt(links[0])==str(d['number'])
parts['word/document.xml']=E.tostring(root,xml_declaration=True,encoding='UTF-8',standalone=True)
parts['word/_rels/document.xml.rels']=E.tostring(relationships,xml_declaration=True,encoding='UTF-8',standalone=True)
with ZipFile(OUT,'w',ZIP_DEFLATED) as z:
 for info in infos:z.writestr(info,parts[info.filename])
(P/'insertions.json').write_text(json.dumps(groups,ensure_ascii=False,indent=2))
result={'output':str(OUT),'new_references':50,'all_references':209,'all_chinese_references':54,'insertion_locations':len(groups),'original_prose_unchanged':True,'all_unrelated_body_nodes_identical':True,'equations_identical':True,'new_citation_targets_valid':True}
(P/'verification.json').write_text(json.dumps(result,ensure_ascii=False,indent=2))
print(json.dumps(result,ensure_ascii=False,indent=2))
