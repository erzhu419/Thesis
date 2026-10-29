"""将本轮逐条人工核对的官方元数据整理为73条书目核验记录。"""
from pathlib import Path
from collections import Counter
import html
import json
import re

B = Path(__file__).resolve().parent
refs = json.loads((B / 'references.json').read_text())
reviewed_arxiv = {77,79,80,81,86,87,88,89,91,95,99,100,101,102,103,104,105,106,107,108,109,110,111,112,115,116,117,118,119,120,121,123,124,125,127,128,129,130,131,132,133,134,135,137,142,143,144,145}
reviewed_crossref = {84,85,90,94}
reviewed_chinese = {74,76,78,82,83,92,93,97,98,113,114,126,136,138,140,141}
years = {74:2025,76:2025,78:2024,82:2023,83:2023,92:2019,93:2023,97:2022,98:2024,113:2020,114:2019,126:2024,136:2023,138:2022,140:2023,141:2022}
records = []


def source(url, kind, cache=None):
    d = {'url': url, 'type': kind, 'retrieved_date': '2026-10-18'}
    if cache:
        d['local_metadata'] = str(cache)
    return d


for ref in refs:
    n = ref['number']
    if not 74 <= n <= 146:
        continue
    rec = {'number': n, 'input_text': ref['text'], 'status': 'unverified', 'canonical': {}, 'issues': [], 'sources': [], 'verified_fields': [], 'notes': []}
    if n in reviewed_arxiv:
        file = B / 'arxiv' / f'{n:03}.json'
        d = json.loads(file.read_text())
        m = d['metadata']
        specified = bool(re.search(r'v\d+$', d['queried_arxiv']))
        year = int((m['updated'] if specified else m['published'])[:4])
        rec.update(status='verified', canonical={'title': m['title'], 'authors': m['authors'], 'year': year, 'venue': 'arXiv', 'arxiv_id_as_cited': d['queried_arxiv'], 'metadata_version': m['id'], 'first_submitted': m['published'], 'version_updated': m['updated'], 'url_as_cited': 'https://arxiv.org/abs/' + d['queried_arxiv']}, sources=[source(d['source'], 'arXiv官方Atom API', file), source('https://arxiv.org/abs/' + d['queried_arxiv'], 'arXiv原文版本记录')], verified_fields=['title', 'listed_authors_and_order', 'year', 'arxiv_identifier', 'url'])
        rec['notes'].append('逐条对照官方返回题名及所列前3位作者的顺序和姓名缩写；未引用卷期页，不添加后来的正式出版信息。')
        rec['notes'].append('明确v版本按该版本updated年份；保留官方首次提交与版本日期。' if specified else '未指定v版本，书目年份按arXiv首次提交年份；最新修订日期不强制替换首次年份。')
        if n in {87,119,132,142}:
            rec['notes'].append('本轮另已直接读取指定版本abs页，题名/作者及提交历史与API一致。')
    elif n in reviewed_crossref:
        file = B / 'crossref' / f'{n:03}.json'
        d = json.loads(file.read_text())
        m = d['metadata']
        title = ': '.join(m.get('title', []) + m.get('subtitle', []))
        authors = [' '.join([a.get('given', ''), a.get('family', '')]).strip() for a in m['author']]
        rec.update(status='verified', canonical={'title': html.unescape(title), 'authors': authors, 'year': m['published']['date-parts'][0][0], 'venue': html.unescape(m['container-title'][0]), 'volume': m.get('volume'), 'issue': m.get('issue'), 'pages_or_article_number': m.get('page') or m.get('article-number'), 'doi': m['DOI'], 'url': m.get('URL', 'https://doi.org/' + m['DOI'])}, sources=[source(d['source'], '出版社直接登记的Crossref元数据', file)], verified_fields=['title_and_subtitle', 'listed_authors_and_order', 'year', 'venue', 'volume_if_cited', 'issue_if_cited', 'pages_or_article_number', 'doi'])
        rec['notes'].append('人工逐项对照作者顺序、题名及已引用的卷期页；忽略标题大小写和DOI大小写差异。')
        if n == 85:
            rec['notes'].append('印刷卷207出版于2026-03，DOI中含2025不构成年份错误。')
        if n == 94:
            rec['notes'].append('官方title=PressLight，subtitle为冒号后的题名，其合并与原书目一致。')
    elif n in reviewed_chinese:
        retry = B / 'primary_chinese' / f'{n:03}_retry.json'
        file = retry if retry.exists() and json.loads(retry.read_text()).get('metadata') else B / 'primary_chinese' / f'{n:03}.json'
        d = json.loads(file.read_text())
        meta = d.get('metadata', [])
        def value(name):
            return next((m['content'].strip() for m in meta if m['name'].lower() == name.lower()), None)
        if n == 98:
            canonical = {'title': '基于合作博弈和强化学习的优先信号控制方法', 'authors': ['秦浩','张维石'], 'year': 2024, 'venue': '计算机应用研究', 'volume': '41', 'issue': '11', 'pages': '3350-3356', 'doi': '10.19734/j.issn.1001-3695.2024.03.0090'}
        else:
            authors = [re.sub(r'\d+$', '', a.strip()) for a in re.split('[,，]', value('citation_authors') or '') if a.strip()]
            volume, issue = value('citation_volume'), value('citation_issue')
            first, last = value('citation_firstpage'), value('citation_lastpage')
            if n in {113,114}:
                sm = re.search(r'Vol\.\s*(\d+),\s*Issue\s*(\d+),\s*Pages:\s*(\d+)-(\d+)', value('dc.source') or '')
                volume, issue, first, last = sm.groups()
            if n == 93:
                last = '122'  # 原刊“引用本文”完整书目。
            if n == 97:
                last = '29'   # 原刊“引用本文”完整书目。
            canonical = {'title': value('citation_title') or value('dc.title'), 'authors': authors, 'year': years[n], 'venue': value('citation_journal_title'), 'volume': volume, 'issue': str(int(issue)) if issue else None, 'pages': first + '-' + last if first and last else None, 'doi': value('citation_doi') or value('dc.identifier')}
        rec.update(status='verified', canonical=canonical, sources=[source(d.get('resolved_url', d['url']), '期刊官方文章HTML及citation/DC元数据', file)], verified_fields=['title','listed_authors_and_order','year','venue','volume','issue','pages','doi'])
        rec['notes'].append('本轮亲读原刊HTML元数据与页面引用格式，逐条核对所列作者次序、题名及卷期页DOI。')
        if n in {83,126,141}:
            rec['notes'].append('网页citation_date含网络先行年份；采用页面正式卷期书目年份，不用先行日期误改正式出版年。')
    else:
        rec['notes'].append('已完成检索，尚需补足部分字段的直接原刊证据；不把网页不可达或Crossref不收录等同于虚假文献。')
    records.append(rec)

# 下列5条本轮部分字段核实，剩余字段仍按可验证范围标记，不作臆断纠正。
partial = {
 75: {'canonical': {'title':'多智能体深度强化学习及其可扩展性与可迁移性研究综述','authors':['闫超','相晓嘉','徐昕','王菖','周晗','沈林成'],'year':2022,'venue':'控制与决策','volume':'37','issue':'12','pages':'3083-3102'},'sources':[source('https://cae.nuaa.edu.cn/2020/0908/c16211a275260/page.htm','第一作者所在大学官方个人主页/发表论文表')],'verified_fields':['title','authors_and_order','year','venue','volume','issue','pages'],'issues':[{'type':'unverified_field','field':'doi','input':'10.13195/j.kzyjc.2022.0044','finding':'原刊访问超时、Crossref不收录；作者官网列出正确书目但未列DOI，未独立核实此DOI。'}]},
 96: {'canonical': {'title':'基于深度强化学习的综合干线协调控制方法','authors':['尚春琳','刘小明','田玉林','董路熙'],'year':2021,'venue':'交通运输系统工程与信息','volume':'21','issue':'3','pages':'64-70'},'sources':[source('http://www.tseit.org.cn/CN/Y2021/V21/I3/64','期刊官方文章HTML与引用本文',B/'primary_chinese/096.json')],'verified_fields':['title','authors_and_order','year','venue','volume','issue','pages'],'issues':[{'type':'unverified_field','field':'doi','input':'10.16097/j.cnki.1009-6744.2021.03.008','finding':'原刊文章citation_doi为空，引用本文及链接本文也未提供DOI；未据此断言所给DOI错误。'}]},
 122: {'canonical': {'title':'开放环境下的协作多智能体强化学习进展','authors':['袁雷','张子谦','李立和','管聪','俞扬'],'year':2025,'venue':'中国科学: 信息科学','volume':'55','issue':'2','pages':'217-268','doi':'10.1360/SSI-2023-0335'},'sources':[source('https://www.sciengine.com/SSI/doi/10.1360/SSI-2023-0335','官方出版平台定位页，本轮403/空JavaScript响应'),source('https://www.lamda.nju.edu.cn/lilh/file/openmarl.pdf','作者大学站点旧预稿索引，仅作论文身份佐证，未下载PDF')],'verified_fields':['author_order_in_author_preprint'],'issues':[{'type':'unverified_fields','fields':['final_title','year','volume','issue','pages','doi'],'finding':'检索摘要有正式书目一致记录，作者原预稿可佐证身份，但正式出版平台本轮不可读，未把二手书目当成完整primary通过证据。'}]},
 139: {'canonical': {'title':'基于双深度网络的安全深度强化学习方法','authors':['朱斐','吴文','伏玉琛','刘全'],'year':2019,'venue':'计算机学报','volume':'42','issue':'8','pages':'1812-1826','doi':'10.11897/SP.J.1016.2019.01812'},'sources':[source('https://cjc.ict.ac.cn/EN/10.11897/SP.J.1016.2019.01812','原刊定位尝试，HTTPS重置/HTTP404'),source('https://scke.jlu.edu.cn/info/1072/1321.htm','论文作者朱斐受邀报告的大学官方记录'),source('https://www.aas.net.cn/cn/article/doi/10.16383/j.aas.c220631','相关原刊参考表66，完整书目佐证，属二手引文')],'verified_fields':['title_and_first_author_research_identity'],'issues':[{'type':'unverified_fields','fields':['full_author_order','year','volume','issue','pages','doi'],'finding':'相关期刊的正式参考表完整支持输入书目，但目标原刊书目记录本轮未取得，保留未全核实。'}]},
 146: {'canonical': {'title':'离线强化学习研究综述','authors':['乌兰','刘全','黄志刚','张立华'],'year':2025,'venue':'计算机学报','volume':'48','issue':'1','pages':'156-187','doi':'10.11897/SP.J.1016.2025.00156'},'sources':[source('https://scst.suda.edu.cn/8f/7b/c11255a692091/page.htm','作者单位官方2025年论文表第70项'),source('https://cjc.ict.ac.cn/EN/10.11897/SP.J.1016.2025.00156','原刊定位尝试，HTTPS重置/HTTP404'),source('https://lib.zjsru.edu.cn/info/1012/53026.htm','大学图书馆期刊导读的卷页佐证，属二手书目')],'verified_fields':['title','authors_and_order','year','venue'],'issues':[{'type':'unverified_fields','fields':['volume','issue','pages','doi'],'finding':'作者单位官网确认论文及全部作者，期刊导读与公开原文索引支持卷页DOI，但目标原刊小HTML尚不可读，未全标通过。'}]},
}
for rec in records:
    if rec['number'] in partial:
        rec.update(partial[rec['number']])

# 本轮追加取得的刊方授权CNKI全文元数据及DOI注册记录。
for rec in records:
    n = rec['number']
    if n in {75,96}:
        doi = {75:'10.13195/j.kzyjc.2022.0044',96:'10.16097/j.cnki.1009-6744.2021.03.008'}[n]
        f = B / 'primary_chinese' / f'{n:03}_extra.json'
        rec.update(status='verified',issues=[],notes=['本轮已读取刊方授权CNKI文章页：题名、所列作者顺序、正式卷期页及DOI均与输入一致；DOI Handle注册记录另确认已注册。'])
        rec['canonical']['doi'] = doi
        rec['sources'].append(source('https://link.cnki.net/doi/'+doi,'刊方授权CNKI原文元数据，完整题名/作者/卷页/DOI',f))
        rec['verified_fields'] += ['doi','doi_registration']
    if n in {75,96,122,139,146}:
        f = B / 'doi_handles' / f'{n:03}.json'
        d = json.loads(f.read_text())
        rec['sources'].append(source(d['source'],'DOI Handle官方注册记录',f))
        if d['responseCode'] == 1:
            rec['verified_fields'].append('doi_registration') if 'doi_registration' not in rec['verified_fields'] else None
            rec['canonical']['doi_registered_target'] = next(v['data']['value'].strip() for v in d['values'] if v['type']=='URL')
        if n == 122:
            rec['sources'].append(source(d['resolver']['url'],'doi.org官方解析页：HTTP404 / DOI Not Found',f))
            rec['verified_fields'].append('doi_resolution_status')
            rec['issues'].append({'type':'doi_not_resolving','field':'doi','input':d['doi'],'finding':'2026-10-18 DOI Handle API返回HTTP404/responseCode100；doi.org解析页独立返回HTTP404和DOI Not Found。此为当前解析异常，不据此判断论文不存在，也不推定替代DOI。'})
            rec['notes'].append('作者大学站点确为2023在审旧预稿，不能替代2025正式题名及卷期页的直接出版证据。最终一次定向检索只取得其他期刊对该正式书目的引文，未据二手引文升级通过。')
        elif n in {139,146}:
            rec['sources'].append(source(rec['canonical']['doi_registered_target'],'DOI注册所指原刊原文；本轮连接断开，未下载PDF'))
            rec['issues'][0]['fields'].remove('doi')
            rec['issues'][0]['finding'] += 'DOI注册及其原刊目标已由Handle官方记录核实，未取得的字段仍保留未全核实。'
            rec['notes'].append('最后一次定向查作者正式论文表/原刊记录未补足可读原始字段；不继续重复抓取同站。')
        if n == 146:
            rec['sources'].append(source('https://lib.zjsru.edu.cn/25-2.24-5.pdf','大学图书馆原刊镜像；搜索缓存含原文首页，直接请求403/超时，不按缓存认定全部字段已核实'))
assert [r['number'] for r in records] == list(range(74,147))
(B / 'audit_074_146.json').write_text(json.dumps(records, ensure_ascii=False, indent=2) + '\n')
print(json.dumps({'entries':len(records),'status':dict(Counter(r['status'] for r in records)),'unverified':[r['number'] for r in records if r['status']=='unverified']},ensure_ascii=False))
