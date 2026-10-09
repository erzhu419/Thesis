import requests,json,concurrent.futures,re
from lxml import html
from pathlib import Path
P=Path(__file__).parent
urls=[
'https://jdxb.bjtu.edu.cn/CN/10.11860/j.issn.1673-0291.20230029',
'https://jdxb.bjtu.edu.cn/CN/10.11860/j.issn.1673-0291.20220173',
'https://www.zjujournals.com/eng/CN/10.3785/j.issn.1008-973X.2024.10.020',
'https://jdxb.bjtu.edu.cn/CN/10.11860/j.issn.1673-0291.20230134',
'https://tjxb.ijournals.cn/jtuns/article/abstract/202203012',
'https://glyqy.csust.edu.cn/glyqy/article/abstract/20240306',
'https://transport.chd.edu.cn/cn/article/doi/10.19818/j.cnki.1671-1637.2025.06.019',
'https://zgglxb.chd.edu.cn/CN/10.19721/j.cnki.1001-7372.2024.04.025',
'https://zgglxb.chd.edu.cn/CN/10.19721/j.cnki.1001-7372.2024.06.020',
'https://www.gljtkj.com/article/2024/1002-0268/20240222.htm',
'https://jtgc.cbpt.cnki.net/portal/journal/portal/client/paper/49d1365a6df0bca0282b15f6f1a3be9a',
'https://www.arocmag.cn/abs/2024.03.0090',
'https://www.ort.shu.edu.cn/CN/Y2023/V27/I2/49',
'https://www.jsjkx.com/CN/10.11896/jsjkx.220500261',
'https://www.aas.net.cn/cn/article/doi/10.16383/j.aas.c240392',
'https://kzyjc.alljournals.cn/html/2022/12/20221202.htm',
'https://www.aas.net.cn/cn/article/doi/10.16383/j.aas.c230546',
'https://www.aas.net.cn/cn/article/doi/10.16383/j.aas.c220823',
'https://www.aas.net.cn/cn/article/doi/10.16383/j.aas.c220631',
'https://www.jsjkx.com/CN/10.11896/jsjkx.220400235',
'https://www.china-simulation.com/CN/10.16182/j.issn1004731x.joss.25-0533',
'https://robot.sia.cn/article/doi/10.13973/j.cnki.robot.230252',
'https://crad.ict.ac.cn/article/doi/10.7544/issn1000-1239.202220388',
]
urls += ['https://www.jos.org.cn/jos/article/abstract/'+n for n in ['7006','7011','6391','6489','6706','6588','6318','6472','6485','6671','7150','7296','7354']]
if __name__=='__main__':
 out=P/'sources';out.mkdir(exist_ok=True)
 def fetch(pair):
  i,u=pair; path=out/f'{i:02d}.json'
  if path.exists():return json.loads(path.read_text())
  try:
   q=requests.get(u,timeout=25); q.encoding=q.apparent_encoding;r=html.fromstring(q.text)
   meta={}
   for m in r.xpath('//meta[@content]'):
    k=m.get('name') or m.get('property') or ''
    if any(x in k.lower() for x in ['citation','dc.','title','description','keywords']): meta.setdefault(k,[]).append(m.get('content'))
   for n in r.xpath('//script|//style'):n.drop_tree()
   t=re.sub(r'[ \t]+',' ',r.text_content());t=re.sub(r'\n\s*\n','\n',t).strip()
   d={'id':i,'url':u,'status':q.status_code,'meta':meta,'text':t}
  except Exception as e:d={'id':i,'url':u,'error':str(e)}
  path.write_text(json.dumps(d,ensure_ascii=False,indent=2));return d
 with concurrent.futures.ThreadPoolExecutor(max_workers=6) as ex:
  for d in ex.map(fetch,enumerate(urls,1)):
   print(d['id'],d.get('status',d.get('error')), {k:v for k,v in d.get('meta',{}).items() if k.lower() in ['citation_title','citation_author','citation_journal_title','citation_date','citation_publication_date','citation_volume','citation_issue','citation_firstpage','citation_lastpage','citation_doi']},flush=True)
