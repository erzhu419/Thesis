"""Separate the two SAC papers and complete verified bibliography fields."""
from copy import deepcopy
from pathlib import Path
from zipfile import ZipFile,ZIP_DEFLATED
from lxml import etree as ET
import json,re

ROOT=Path('/home/erzhu419/mine_code/Thesis');A=Path(__file__).resolve().parent
BASE=A/'v3引用核对前.docx';OUT=ROOT/'output/Gemini/重构论文_排版稿_20261015.docx'
W='{http://schemas.openxmlformats.org/wordprocessingml/2006/main}'
M='{http://schemas.openxmlformats.org/officeDocument/2006/math}'
NS={'w':W[1:-1]}
rows=json.loads((A/'v3_reference_list_parsed.json').read_text())['references'];by_number={r['number']:r for r in rows}
newnums=json.loads((A/'v3_reference_key_order.json').read_text())
oldnums=json.loads((A/'reference_numbers.json').read_text())
if not (A/'v3_reference_numbers_before.json').exists():(A/'v3_reference_numbers_before.json').write_text(json.dumps(oldnums,ensure_ascii=False,indent=2))
oldnums=json.loads((A/'v3_reference_numbers_before.json').read_text())
def text(e):return ''.join(n.text or '' for n in e.iter() if n.tag in [W+'t',M+'t'])
def sty(e):
 n=e.find('./'+W+'pPr/'+W+'pStyle');return n.get(W+'val') if n is not None else ''
def replace(e,old,new):
 ts=list(e.iter(W+'t'));s=''.join(t.text or '' for t in ts);start=s.index(old);end=start+len(old);offset=0;first=True
 for t in ts:
  v=t.text or '';lo=max(start-offset,0);hi=min(end-offset,len(v))
  if lo<hi:t.text=v[:lo]+(new if first else '')+v[hi:];first=False
  offset+=len(v)
def reference(link,n):
 old=''.join(link.itertext())
 # itertext may duplicate OOXML container text in registered elements; use w:t only.
 old=''.join(link.xpath('.//w:t/text()',namespaces=NS))
 assert old.isdigit(),old
 replace(link,old,str(n));link.set(W+'anchor','latest_ref_'+str(n))

with ZipFile(BASE) as z:
 original=z.read('word/document.xml');root=ET.fromstring(original);body=root.find(W+'body');blocks=list(body)
 main_start=next(i for i,e in enumerate(blocks) if sty(e)=='chapter' and text(e).startswith('第1章'))
 main_end=next(i for i,e in enumerate(blocks) if sty(e)=='bibliographytitle')
 g_start=next(i for i,e in enumerate(blocks) if sty(e)=='chapter' and text(e).startswith('附录G'))
 main_links=[n for e in blocks[main_start:main_end] for n in e.iter(W+'hyperlink') if n.get(W+'anchor','').startswith('latest_ref_')]
 g_links=[n for e in blocks[g_start:] for n in e.iter(W+'hyperlink') if n.get(W+'anchor','').startswith('latest_ref_')]
 raw=(ROOT/'output/Gemini/重构论文.md').read_text();main,g=raw.split('\n# 附录F',1)
 pattern=re.compile(r'\[([A-Za-z]+\d{4}[\w-]*(?:\s*,\s*[A-Za-z]+\d{4}[\w-]*)*)\]')
 def keys(s):return [k.strip() for m in pattern.finditer(s) for k in m[1].split(',')]
 for source,links in [(main,main_links),(g,g_links)]:
  ks=keys(source);assert len(ks)==len(links)
  for key,link in zip(ks,links):
   assert link.get(W+'anchor')=='latest_ref_'+str(oldnums[key]),key
 # Adjust references outside the regenerated source (old appendices), if applicable.
 source_ids={id(n) for n in main_links+g_links}
 for link in root.iter(W+'hyperlink'):
  a=link.get(W+'anchor','')
  if a.startswith('latest_ref_') and id(link) not in source_ids:
   num=int(a.removeprefix('latest_ref_'))
   if num>=205:reference(link,num+1)
 for source,links in [(main,main_links),(g,g_links)]:
  for key,link in zip(keys(source),links):reference(link,newnums[key])
 # Shift existing bibliography entries 205..225 and add the missing independent paper.
 bib={int(re.match(r'^\[(\d+)\]',text(p))[1]):p for p in blocks if sty(p)=='bibliography'}
 for number in range(225,204,-1):
  p=bib[number];replace(p,'['+str(number)+']','['+str(number+1)+']')
  for mark in p.iter(W+'bookmarkStart'):
   if mark.get(W+'name')=='latest_ref_'+str(number):mark.set(W+'name','latest_ref_'+str(number+1))
 markid=max(int(n.get(W+'id')) for n in root.iter(W+'bookmarkStart'))+1
 p=ET.Element(W+'p');p.append(deepcopy(bib[63].find(W+'pPr')))
 b=ET.SubElement(p,W+'bookmarkStart');b.set(W+'id',str(markid));b.set(W+'name','latest_ref_205')
 r=ET.SubElement(p,W+'r');ET.SubElement(r,W+'t').text='[205]';ET.SubElement(r,W+'tab')
 algorithm_text='HAARNOJA T, ZHOU A, HARTIKAINEN K, et al. Soft Actor-Critic Algorithms and Applications[EB/OL]. arXiv, 2018[2026-10-15]. https://arxiv.org/abs/1812.05905.'
 t=ET.SubElement(r,W+'t');t.text=algorithm_text
 b=ET.SubElement(p,W+'bookmarkEnd');b.set(W+'id',str(markid));bib[204].addnext(p)
 # Five DOIs supplied by v3 and one empty field independently verified at the journal.
 doi_additions={n:by_number[n]['doi'][0] for n in [43,44,55,94,114]}
 doi_additions[147]='10.3969/j.issn.1003-3114.2024.05.001'
 for n,doi in doi_additions.items():
  assert doi not in text(bib[n])
  r=ET.SubElement(bib[n],W+'r');ET.SubElement(r,W+'t').text=' DOI:'+doi+'.'
 newbib=[e for e in body if sty(e)=='bibliography']
 assert [int(re.match(r'^\[(\d+)\]',text(e))[1]) for e in newbib]==list(range(1,227))
 names=[n.get(W+'name') for n in root.iter(W+'bookmarkStart')]
 assert len(names)==len(set(names))
 assert all(n.get(W+'anchor') in names for n in root.iter(W+'hyperlink') if n.get(W+'anchor'))
 assert [n.text for n in ET.fromstring(original).iter(M+'t')]==[n.text for n in root.iter(M+'t')]
 han=lambda s:''.join(re.findall(r'[\u3400-\u9fff]',s))
 assert han(text(ET.fromstring(original)))==han(text(root)),'Chinese changed'
 with ZipFile(OUT,'w',ZIP_DEFLATED) as out:
  for info in z.infolist():out.writestr(info,ET.tostring(root,encoding='UTF-8',xml_declaration=True,standalone=True) if info.filename=='word/document.xml' else z.read(info.filename))

# Repair the source-key index so future builds preserve the two distinct SAC papers.
cm=json.loads((A/'citation_key_map.json').read_text())
original_rec=deepcopy(cm['matched']['haarnoja2018sac']);original_rec.update(old_id='haarnoja2018sac',reason='v3首次引用顺序确认原始SAC，独立于Algorithms and Applications',title=by_number[63]['text'].split('Soft Actor-Critic:',1)[-1],source_url='https://arxiv.org/abs/1801.01290')
original_rec['title']='Soft Actor-Critic: Off-Policy Maximum Entropy Deep Reinforcement Learning with a Stochastic Actor'
applications_rec={'old_id':'haarnoja2018applications','old_number':135,'reason':'v3第205条；原Word第135条及arXiv官方记录确认，是独立的第二篇SAC论文','title':'Soft Actor-Critic Algorithms and Applications','bibliography_text':'[205] '+algorithm_text,'source_url':'https://arxiv.org/abs/1812.05905','verification_status':'verified'}
split={'haarnoja2018sac':original_rec,'haarnoja2018soft':applications_rec}
(A/'citation_additions_sac_split.json').write_text(json.dumps(split,ensure_ascii=False,indent=2));cm['matched'].update(split)
for key,n in newnums.items():
 if n in doi_additions:
  rec=cm['matched'][key];doi=doi_additions[n]
  if doi not in rec['bibliography_text']:rec['bibliography_text']+=' DOI:'+doi+'.'
  rec['DOI']=doi
(A/'citation_key_map.json').write_text(json.dumps(cm,ensure_ascii=False,indent=2))
(A/'reference_numbers.json').write_text(json.dumps(newnums,ensure_ascii=False,indent=2))
# Deliver a cleaned copy of the supplied list, leaving the original TXT untouched.
clean=[]
for row in rows:
 n=row['number'];value=row['text']
 if n==147:value=re.sub(r'DOI:\s*$','DOI:'+doi_additions[n]+'.',value)
 if n in [31,51]:value=re.sub(r'\s*DOI:\s*$','',value)
 if n==172:value=value.replace('Openai','OpenAI')
 if n==181:value=value.replace('Mcallister','McAllister')
 clean.append('['+str(n)+'] '+value)
cleanpath=ROOT/'output/Gemini/v3版引用文献列表_核对版_20261015.txt'
cleanpath.write_text('按正文首次出现顺序编号\n\n'+'\n\n'.join(clean)+'\n',encoding='utf-8-sig')
result=json.loads((A/'build_result.json').read_text());result.update(bibliography_count=226,word_finalized=False,v3_reference_list_reviewed=True,v3_reference_count=226,source_citation_keys_matched=226,sac_papers_separated=True,reference_DOI_additions_v3=6)
(A/'build_result.json').write_text(json.dumps(result,ensure_ascii=False,indent=2))
summary={'reviewed':226,'substantive_bibliography_mismatches':0,'duplicate_identifiers':0,'source_keys':226,'Word_bibliography_before':225,'Word_bibliography_after':226,'Word_repair':'Separated SAC original paper [63] from Algorithms and Applications [205]; all following references renumbered','list_repairs':{'147':'DOI:'+doi_additions[147],'172':'Openai to OpenAI','181':'Mcallister to McAllister','31,51':'Removed empty DOI labels; no DOI invented'},'Word_DOI_additions':doi_additions,'cleaned_list':str(cleanpath)}
(A/'v3_reference_review_result.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2))
print(json.dumps(summary,ensure_ascii=False,indent=2))
