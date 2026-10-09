from pathlib import Path
import json,re,urllib.request,urllib.parse,concurrent.futures
work=Path(__file__).parent
annotations=json.loads((work/'annotations.json').read_text())
analysis=json.loads((work/'reference_analysis.json').read_text())[0]
refs={int(re.match(r'^\[(\d+)\]',r['text']).group(1)):r for r in analysis['references']}
a=[x for x in annotations if x['index']>=108]
def fetch(x):
 c=x['info']['content'];num=int(re.search(r'-\[(\d+)\]',c).group(1));ref=refs[num]
 doi=re.search(r'路径:DOI:(.+?)是否正确',c)
 result={'annotation_index':x['index'],'report_page':x['page'],'reference_number':num,'paragraph_index':ref['p_index'],'requirement':c,'current_text':ref['text'],'change':None}
 if doi:
  doi=doi.group(1);url='https://api.crossref.org/works/'+urllib.parse.quote(doi,safe='')
  result.update(doi=doi,source_url=url,doi_url='https://doi.org/'+doi)
  try:
   req=urllib.request.Request(url,headers={'User-Agent':'Thesis-reference-verification/1.0','Accept':'application/json'})
   with urllib.request.urlopen(req,timeout=35) as f: data=json.load(f)['message']
   result.update(metadata={k:data.get(k) for k in ['DOI','title','author','container-title','published','published-print','published-online','volume','issue','page','article-number','publisher','URL','resource']},status='doi_registered_metadata_available')
  except Exception as e:result.update(status='unverified',error=str(e))
 else:
  result.update(status='requires_author_verification')
 return result
with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
 records=list(pool.map(fetch,a))
(work/'reference_metadata_verification.json').write_text(json.dumps(records,ensure_ascii=False,indent=2))
for x in records:
 print(x['annotation_index'],x['reference_number'],x['status'],str(x.get('metadata',{}).get('title','')))
