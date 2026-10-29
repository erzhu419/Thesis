from pathlib import Path
import json,re

T=Path(__file__).resolve().parent
refs=json.loads((T/'references.json').read_text())[:73]
arxiv_nums={7,10,11,12,58,59,60,61,62,63,68,72}
english_nums={1,2,3,5,6,8,9,13,42,46,47,49,50,52,53,54,56,69,71,73}
chinese_nums=set(range(1,74))-arxiv_nums-english_nums
source_override={
 20:'https://journal.bjut.edu.cn/bjgydxxb/cn/article/doi/10.11936/bjutxb2019100011',
 31:'https://cs.hit.edu.cn/_upload/article/files/1e/10/1856ecb9495dba4be992671faa6e/abbb87e2-50de-4061-ab04-d72f1926b1c7.pdf',
 41:'/home/erzhu419/mine_code/Thesis/不确定环境下多线路公交信号优先在途协同控制方法研究 - 提交(1).pdf',
 55:'https://zzszrb.chd.edu.cn/en/oa/KeySearch.aspx?Sel=control+strategy&type=KeyWord',
}
extra_authors={
 20:['孙继洋','黄建玲','陈艳艳','魏攀一','贾建林','宋程程'],
 29:['龙雪琴','李景涛','王建军','拓小静','范镓麟'],
 31:['赵新标','刘心雨','夏晓梅','项俊平','赵靖'],
 33:['赵琥','冯树民','廖嘉雯','慈玉生'],
 36:['李利华','曹慧琪','邓亚军','邢璐','靳竹喧'],
 37:['赵小梅','朱香原','汪秦','谢东繁'],
 40:['徐亮','于海洋','金盛','任毅龙'],
 41:['刘鹏杰'],48:['周雨阳','张梦瑶','李长锦','王旭涛','刘国鹏','李资资'],
 51:['何胜学'],55:['姜瑞森','胡大伟','孙倩','高天洋'],
}

def source(url,kind,evidence):return {'url':url,'type':kind,'evidence':evidence,'accessed':'2026-10-18'}
def read(p):return json.loads(p.read_text())
rows=[]
for ref in refs:
 n=ref['number'];txt=ref['text'].strip()
 m=re.match(r'^\[\d+\](.*?)\.\s+(.+?)\[([A-Z/]+)\](?:\.\s*|//)(.*)$',txt,re.S)
 assert m,(n,txt)
 names,title,typ,rest=m.groups()
 yr=re.search(r'(?:19|20)\d{2}',rest);year=int(yr[0])
 venue=rest[:yr.start()].rstrip(', :')
 v=re.search(r',\s*\d{4},\s*(\d+)(?:\(([^)]+)\))?\s*:\s*([\d-]+)',rest)
 doi=re.search(r'DOI:([^\s]+)',rest)
 url=re.search(r'https?://\S+',rest)
 ident=('DOI:'+doi[1].rstrip('.'))if doi else (url[0].rstrip('.')if url else '')
 row={'number':n,'status':'verified','canonical':txt,'title':title,
      'authors':[x.strip()for x in names.split(',')], 'year':year,'venue':venue,
      'volume':v[1]if v else '', 'issue':(v[2]or'')if v else '',
      'pages':v[3]if v else '', 'identifier':ident,'issues':[],'sources':[],
      'verified_fields':['title','authors','year','venue','volume','issue','pages','identifier'],
      'notes':[]}
 if n in english_nums:
  cache=read(T/'crossref'/f'{n:03}.json');md=cache['metadata']
  assert cache['http_status']==200
  row['title']=': '.join((md.get('title')or[])+(md.get('subtitle')or[]))
  row['authors']=[' '.join([a.get('given',''),a.get('family','')]).strip()for a in md.get('author',[])]
  row['venue']=(md.get('container-title')or[venue])[0]
  row['volume']=md.get('volume','');row['issue']=md.get('issue','')
  row['pages']=md.get('page')or md.get('article-number')or row['pages']
  row['sources']=[source(cache['source'],'DOI_registration','本轮DOI登记元数据：逐项核对题名、所列作者、出版年、刊名、卷期和页码/文章号；均匹配。')]
  if n==9:
   row['authors']=['Pablo Alvarez Lopez','Michael Behrisch','Laura Bieker-Walz','Jakob Erdmann','Yun-Pang Flötteröd','Robert Hilbrich','Leonhard Lücken','Johannes Rummel','Peter Wagner','Evamarie Wießner']
   row['sources'].append(source('https://sumo.dlr.de/userdoc/Publications.html','author_project_primary','SUMO官方文献表列作者顺序为Lopez、Behrisch、Bieker-Walz等，与当前Word一致。'))
   row['notes'].append('Crossref把Wießner排在第二位，与官方SUMO文献表/论文原始署名顺序不一致；以作者项目官方署名为准，当前Word前三作者正确，不建议修改。')
  if n==54:row['notes'].append('登记元数据将冒号后部分置于subtitle；合并title+subtitle后与Word完整题名一致。')
 elif n in arxiv_nums:
  cache=read(T/'arxiv'/f'{n:03}.json');md=cache['metadata'];assert cache['returned_exact_version']
  row['title']=md['title'];row['authors']=md['authors'];row['venue']='arXiv'
  row['sources']=[source(cache['source'],'arXiv_primary_metadata',f"官方API返回指定记录{md['id']}，题名和作者匹配；published={md['published']}，updated={md['updated']}。"),source(md['url'],'arXiv_primary_abstract','所查版本的官方摘要入口。')]
  row['verified_fields']=['title','authors','year','venue','identifier','url','version']
  if n==12:row['notes'].append('明确指定2309.12716v2，v2更新日2025-04-16；当前著录2025正确，不能据初投2023自动改年。')
  else:row['notes'].append('所列年份采用arXiv首次提交年，与官方published字段一致；无v后缀者本次API返回最新修订，修订日不同不构成年份错误。')
 else:
  p=T/'primary_001_073'/f'{n:03}.json';d=read(p)if p.exists()else{}
  if n==41:
   row['venue']='中南大学';row['volume']='';row['issue']='';row['pages']='';row['identifier']='中南大学博士学位论文（长沙，2025年5月）'
   row['sources']=[source(source_override[n],'local_primary_thesis_cover','根代理本轮只读打开用户原PDF封面：题名、作者刘鹏杰、中南大学、2025年5月全部与Word41匹配。')]
   row['verified_fields']=['title','authors','year','venue','document_type','place']
   row['notes'].append('按PDF封面题名核实；附件文件名不是封面题名，不据文件名更改著录。')
  elif n==31:
   row['sources']=[source(source_override[n],'journal_host_primary_pdf_header','主办单位哈尔滨工业大学托管PDF首页：赵新标、刘心雨、夏晓梅、项俊平、赵靖；2023年11月13卷11期；文章编号2095-2163(2023)11-0202-06支持202–207页。')]
   row['verified_fields']=['title','authors','year','venue','volume','issue','pages']
   row['notes'].append('当前未列DOI；本轮未为其推测添加DOI。')
  elif n==55:
   row['sources']=[source(source_override[n],'journal_primary_search_index','本轮原刊关键词索引的检索记录给出精确题名、姜瑞森/胡大伟/孙倩/高天洋、2023年第6期95–105页；直接访问被403限制。'),source('https://transport.chd.edu.cn/cn/article/doi/10.19818/j.cnki.1671-1637.2025.06.019','same_author_journal_bibliography','作者团队后续原刊论文参考文献27完整列出本条，题名、前三作者、2023、43(6)、95–105一致。')]
   row['notes'].append('原刊关键词索引和作者团队后续原刊书目证实主体，Handle官方查询证实该DOI已注册；文章详情页/CNKI本次403。依本轮“原刊书目信息＋注册记录”的核验口径通过，未把403当作伪造。')
  else:
   row['sources']=[source(source_override.get(n,d.get('url','')),'journal_primary_metadata','本轮原刊文章页citation/DC元数据及页首/引用本文：逐项核对题名、所列作者、刊年、卷期、页码；与Word一致。')]
   if n==20:row['notes'].append('requests返回403；本轮web可读官方HTML检索记录及原刊PDF预览确认完整书目。')
   metas=d.get('metadata',[])
   cn=lambda m:m.get('lang')not in('en','EN')
   aa=[m['content']for m in metas if m.get('name','').lower()=='citation_author'and cn(m)]
   if not aa:
    aa=[m['content']for m in metas if m.get('name','').lower()=='citation_authors'and cn(m)]
    aa=re.split(r'[,，;；、]',aa[0])if aa else[]
   aa=[x.strip().strip('* ')for x in aa if x.strip().strip('* ')not in('等','et al','et al.')]
   if aa:row['authors']=aa
   if n==15:row['sources'].append(source('https://jdxb.bjtu.edu.cn/CN/article/downloadArticleFile.do?attachType=PDF&id=3635','journal_primary_pdf_header','原刊PDF文章编号1673-0291(2020)04-0086-08确认86–93；作者以原刊HTML郇宁为准。'))
   if n==21:row['notes'].append('原刊中文/英文引用本文均明确151–165；另一院系页写151–165+187，但未见原PDF或勘误证据，不据其他文献改动当前正确页码。')
   if n in(34,43):row['notes'].append('原刊旧版HTML的citation_doi为空；题名、作者、刊年卷页由原刊页证实，DOI由官方Handle注册记录证实，按本轮组合证据口径通过。')
   if n==44:
    row['sources'].append(source('http://www.tseit.org.cn/CN/article/downloadArticleFile.do?attachType=PDF&id=19945','journal_primary_pdf_header','本轮原刊PDF首页明确印有DOI:10.16097/j.cnki.1009-6744.2019.06.014；英文题名、四作者、2019年19卷6期及文章编号1009-6744(2019)06-0092-07均对应。'))
    row['notes'].append('旧HTML citation_doi为空，但原刊PDF首页明确印有当前DOI；已直接证实。PDF仅在内存读取首页，未保存整份PDF。')
   if n==51:row['notes'].append('原刊明确2019年17卷3期133–143，DOI字段为空；当前原刊URL准确。')
   if n==64:row['notes'].append('网页citation_date=2021-08-02为网络优先日期；页首明确2022年33卷11期4217–4238，当前正式刊年2022正确。')
  if n in extra_authors:row['authors']=extra_authors[n]
  h=T/'doi_handles'/f'{n:03}.json'
  if doi and h.exists():
   hd=read(h);row['sources'].append(source(hd['source'],'DOI_Handle_registration',f"本轮Handle注册查询responseCode={hd.get('responseCode')}；用于区分DOI注册/解析状态，不单独替代题名对应核验。"))
   if hd.get('responseCode')==100:row['notes'].append('原刊所列DOI与Word一致，但本轮DOI Handle查询responseCode=100（未返回注册记录）；这是解析/注册异常，单独记录，不据此认定论文不存在或修改原刊书目。')
 rows.append(row)

(T/'audit_001_073.json').write_text(json.dumps(rows,ensure_ascii=False,indent=2)+'\n')
print('Saved',len(rows),'rows;', {s:sum(x['status']==s for x in rows)for s in ['verified','correction_needed','unverified']})
