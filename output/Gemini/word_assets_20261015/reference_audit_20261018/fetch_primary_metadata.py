from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from urllib.parse import quote
from xml.etree import ElementTree as E
import requests,json,re,time
T=Path(__file__).resolve().parent
refs=json.loads((T/'references.json').read_text())
headers={'User-Agent':'ThesisReferenceVerification/1.0 (bibliographic metadata only)'}
def get(url):
 r=requests.get(url,headers=headers,timeout=30);return r

def doi_fetch(ref):
 number=ref['number'];text=ref['text'];match=re.search(r'DOI:\s*(10\.[^\s]+)',text,re.I)
 if not match:return None
 doi=match[1].rstrip('.。');out={'number':number,'queried_doi':doi,'source':'https://api.crossref.org/works/'+quote(doi,safe='')}
 try:
  r=get(out['source']);out['http_status']=r.status_code
  if r.status_code==200:
   m=r.json()['message'];fields=['DOI','title','subtitle','author','container-title','short-container-title','publisher','published','published-print','published-online','issued','volume','issue','page','article-number','type','URL','link']
   out['metadata']={k:m[k]for k in fields if k in m}
  else:out['error']=r.text[:180]
 except Exception as e:out['error']=str(e)
 (T/'crossref'/f'{number:03}.json').write_text(json.dumps(out,ensure_ascii=False,indent=2))
 return out
(T/'crossref').mkdir(exist_ok=True)
jobs=[r for r in refs if 'DOI:'in r['text']]
with ThreadPoolExecutor(max_workers=4)as pool:
 for i,f in enumerate(as_completed([pool.submit(doi_fetch,r)for r in jobs]),1):
  x=f.result()
  if i%20==0 or i==len(jobs):print('CROSSREF',i,'/',len(jobs),flush=True)
# Request only compact arXiv metadata, in small id-list batches; preserve cited versions.
atom='{http://www.w3.org/2005/Atom}';ar='{http://arxiv.org/schemas/atom}'
entries=[(r,re.search(r'https://arxiv.org/abs/([^\s.]+(?:\.[^\s]+)?)',r['text']))for r in refs]
entries=[(r,m[1].rstrip('.'))for r,m in entries if m]
(T/'arxiv').mkdir(exist_ok=True)
for start in range(0,len(entries),20):
 batch=entries[start:start+20];ids=','.join(x[1]for x in batch);url='https://export.arxiv.org/api/query?id_list='+ids
 try:
  response=get(url);response.raise_for_status();root=E.fromstring(response.text);found={}
  for entry in root.findall(atom+'entry'):
   ident=(entry.findtext(atom+'id')or'').split('/abs/')[-1]
   found[ident]={'id':ident,'title':' '.join((entry.findtext(atom+'title')or'').split()),'authors':[' '.join((x.findtext(atom+'name')or'').split())for x in entry.findall(atom+'author')],'published':entry.findtext(atom+'published'),'updated':entry.findtext(atom+'updated'),'doi':entry.findtext(ar+'doi'),'journal_ref':entry.findtext(ar+'journal_ref'),'url':entry.findtext(atom+'id')}
  for ref,ident in batch:
   match=found.get(ident)or next((v for k,v in found.items()if re.sub(r'v\d+$','',k)==re.sub(r'v\d+$','',ident)),None)
   out={'number':ref['number'],'queried_arxiv':ident,'source':url,'metadata':match,'returned_exact_version':match is not None and (match['id']==ident or not re.search(r'v\d+$',ident))}
   (T/'arxiv'/f"{ref['number']:03}.json").write_text(json.dumps(out,ensure_ascii=False,indent=2))
  print('ARXIV',min(start+20,len(entries)),'/',len(entries),'returned',len(found),flush=True)
 except Exception as e:
  print('ARXIV_BATCH_ERROR',start,str(e),flush=True)
  for ref,ident in batch:
   (T/'arxiv'/f"{ref['number']:03}.json").write_text(json.dumps({'number':ref['number'],'queried_arxiv':ident,'source':url,'error':str(e)},ensure_ascii=False,indent=2))
print('DONE',flush=True)
