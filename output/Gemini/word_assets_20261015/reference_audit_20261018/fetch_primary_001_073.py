from pathlib import Path
from concurrent.futures import ThreadPoolExecutor,as_completed
from lxml import html
import requests,json,re
T=Path(__file__).resolve().parent;A=T.parent
refs=json.loads((T/'references.json').read_text())[:73]
urls={4:'https://xuebao.jlu.edu.cn/gxb/CN/10.13229/j.cnki.jdxbgxb20200676',14:'https://jdxb.bjtu.edu.cn/CN/10.11860/j.issn.1673-0291.20230029',15:'https://jdxb.bjtu.edu.cn/CN/10.11860/j.issn.1673-0291.20200010',16:'https://jdxb.bjtu.edu.cn/CN/10.11860/j.issn.1673-0291.20220173',17:'https://tjxb.ijournals.cn/jtuns/article/abstract/202203012',18:'https://glyqy.csust.edu.cn/glyqy/article/abstract/20240306',19:'https://sysengi.cjoe.ac.cn/CN/10.12011/SETP2021-1149',20:'https://journal.bjut.edu.cn/bjgydxxb/cn/article/doi/10.11936/bjutxb2019100011',25:'https://www.zjujournals.com/eng/CN/10.3785/j.issn.1008-973X.2024.10.020',26:'https://jdxb.bjtu.edu.cn/CN/10.11860/j.issn.1673-0291.20230134',27:'https://jdxb.bjtu.edu.cn/CN/10.11860/j.issn.1673-0291.20230162',28:'https://www.ort.shu.edu.cn/CN/10.15960/j.cnki.issn.1007-6093.2023.03.005',29:'https://zrb.bjb.scut.edu.cn/CN/10.12141/j.issn.1000-565X.210270',35:'https://tjxb.tongji.edu.cn/jtuns/article/abstract/201912008',37:'https://zrb.bjb.scut.edu.cn/CN/10.12141/j.issn.1000-565X.220493',51:'https://jtgc.cbpt.cnki.net/portal/journal/portal/client/paper/JTGC_da6bd34e-4473-40bd-ab25-06c8a87baf30',64:'https://www.jos.org.cn/jos/article/abstract/6391',65:'https://zzs.ujs.edu.cn/pgjx/EN/10.3969/j.issn.1674-8530.22.0108',66:'https://www.aas.net.cn/cn/article/doi/10.16383/j.aas.c220103',67:'https://www.jos.org.cn/jos/article/abstract/6740'}
for n,year,v,issue,page in [(21,2024,24,4,151),(22,2024,24,4,176),(24,2025,25,6,239),(34,2018,18,4,61),(43,2018,18,4,103),(44,2019,19,6,92),(45,2023,23,6,165),(70,2024,24,5,173)]:urls[n]=f'http://www.tseit.org.cn/CN/Y{year}/V{v}/I{issue}/{page}'
for n in(23,30,33,36,38,40):
 doi=re.search(r'DOI:([^\s]+)',refs[n-1]['text'])[1].rstrip('.');urls[n]='https://zgglxb.chd.edu.cn/CN/'+doi
out=T/'primary_001_073';out.mkdir(exist_ok=True)
def fetch(n,url):
 target=out/f'{n:03}.json'
 if target.exists():return n,'cached'
 try:
  r=requests.get(url,timeout=25);r.encoding=r.apparent_encoding if r.encoding in(None,'ISO-8859-1')else r.encoding
  soup=html.fromstring(r.text);meta=[{'name':m.get('name')or m.get('property'),'lang':m.get('xml:lang')or m.get('lang'),'content':m.get('content')}for m in soup.xpath('//meta')if(m.get('name')or m.get('property')or'').lower().startswith(('citation','dc.','dc:','prism','og:'))]
  # Only metadata and the short article header are retained, no full paper.
  plain=' '.join(x.strip()for x in soup.xpath('//body//text()[not(ancestor::script) and not(ancestor::style)]')if x.strip());data={'number':n,'url_requested':url,'url':r.url,'status':r.status_code,'title':' '.join(soup.xpath('//title/text()')),'metadata':meta,'header':plain[:2300]}
 except Exception as e:data={'number':n,'url_requested':url,'error':str(e)}
 target.write_text(json.dumps(data,ensure_ascii=False,indent=2)+'\n');return n,len(data.get('metadata',[])),data.get('status')
with ThreadPoolExecutor(max_workers=7)as ex:
 for f in as_completed([ex.submit(fetch,n,url)for n,url in urls.items()]):print(f.result(),flush=True)
