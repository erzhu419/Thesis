import requests,json,concurrent.futures,re,fitz
from lxml import html
from pathlib import Path
P=Path(__file__).parent
urls={
37:'https://sysengi.cjoe.ac.cn/CN/10.12011/SETP2021-1149',
38:'https://zgglxb.chd.edu.cn/CN/10.19721/j.cnki.1001-7372.2023.06.020',
39:'https://ir.bjut.edu.cn/irpui/community/newsearch?info=%E7%BB%84%E5%90%88%E4%BC%98%E5%8C%96&select=keyword',
40:'https://zzszrb.chd.edu.cn/oa/DArticle.aspx?id=202306009&type=view',
41:'https://doi.org/10.12141/j.issn.1000-565X.220493',
42:'https://www.gljtkj.com/article/2025/1002-0268/gljtkj-42-10-161.htm',
43:'https://computing.hit.edu.cn/_upload/article/files/1e/10/1856ecb9495dba4be992671faa6e/abbb87e2-50de-4061-ab04-d72f1926b1c7.pdf',
44:'https://jdxb.bjtu.edu.cn/CN/10.11860/j.issn.1673-0291.20230162',
45:'https://www.ort.shu.edu.cn/CN/Y2023/V27/I3/68',
46:'https://zgglxb.chd.edu.cn/CN/10.19721/j.cnki.1001-7372.2023.10.022',
47:'https://doi.org/10.11897/SP.J.1016.2025.00156',
48:'https://aas.net.cn/cn/article/doi/10.16383/j.aas.c230019?viewType=HTML',
49:'https://fcst.ceaj.org/CN/10.3778/j.issn.1673-9418.2211113',
50:'https://xbna.pku.edu.cn/CN/10.13209/j.0479-8023.2023.086',
51:'https://zzs.ujs.edu.cn/pgjxgcxb/CN/10.3969/j.issn.1674-8530.22.0108',
52:'https://wxdt.cbpt.cnki.net/portal/journal/portal/client/paper/c4632864a2c1c2b0fb849e733d34bf7f',
53:'https://www.lamda.nju.edu.cn/lilh/file/openmarl.pdf',
54:'https://zgglxb.chd.edu.cn/CN/Y2023/V36/I2',
55:'https://doi.org/10.11897/SP.J.1016.2023.00843',
56:'https://pubs.cstam.org.cn/article/doi/10.3969/j.issn.2095-1469.2025.01.05',
}
def fetch(pair):
 i,u=pair; path=P/'sources'/f'{i:02d}.json'
 if path.exists():return json.loads(path.read_text())
 try:
  q=requests.get(u,timeout=(12,25));meta={}
  if q.content.startswith(b'%PDF'):
   doc=fitz.open(stream=q.content,filetype='pdf');t='\n'.join(p.get_text() for p in doc)
  else:
   q.encoding=q.apparent_encoding;r=html.fromstring(q.text)
   for m in r.xpath('//meta[@content]'):
    k=m.get('name') or m.get('property') or ''
    if any(x in k.lower() for x in ['citation','dc.','title','description','keywords']): meta.setdefault(k,[]).append(m.get('content'))
   for n in r.xpath('//script|//style'):n.drop_tree()
   t=re.sub(r'\s+',' ',r.text_content()).strip()
  d={'id':i,'url':u,'resolved_url':q.url,'status':q.status_code,'meta':meta,'text':t}
 except Exception as e:d={'id':i,'url':u,'error':str(e)}
 path.write_text(json.dumps(d,ensure_ascii=False,indent=2));return d
if __name__=='__main__':
 with concurrent.futures.ThreadPoolExecutor(max_workers=6) as ex:
  for d in ex.map(fetch,urls.items()):
   print(d['id'],d.get('status',d.get('error')), d.get('meta'), d.get('text','')[:160],flush=True)
