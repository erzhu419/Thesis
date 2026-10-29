"""Record the completed item-by-item primary-metadata review; no manuscript writes."""
from pathlib import Path
import json
import re

T = Path(__file__).resolve().parent
references = {r['number']: r['text'] for r in json.loads((T / 'references.json').read_text())}
chinese = {
    147: ('基于离线强化学习的研究综述', '陈锶奇 耿婕 汪云飞 余伟驰 赵佳宁 王仕超', 2024, '无线电通信技术', '50', '5', '831-842', '10.3969/j.issn.1003-3114.2024.05.001', 'https://wxdt.cbpt.cnki.net/portal/journal/portal/client/paper/c4632864a2c1c2b0fb849e733d34bf7f'),
    148: ('基于优先采样模型的离线强化学习', '顾扬 程玉虎 王雪松', 2024, '自动化学报', '50', '1', '143-153', '10.16383/j.aas.c230019', 'https://www.aas.net.cn/article/doi/10.16383/j.aas.c230019'),
    149: ('扩散模型期望最大化的离线强化学习方法', '刘全 颜洁 乌兰', 2025, '软件学报', '36', '10', '4695-4709', '10.13328/j.cnki.jos.007296', 'https://jos.org.cn/jos/article/abstract/7296'),
    161: ('基于表征学习的离线强化学习方法研究综述', '王雪松 王荣荣 程玉虎', 2024, '自动化学报', '50', '6', '1104-1128', '10.16383/j.aas.c230546', 'https://www.aas.net.cn/article/doi/10.16383/j.aas.c230546'),
    174: ('面向机器人系统的虚实迁移强化学习综述', '林谦 余超 伍夏威 董银昭 徐昕 张强 郭宪', 2024, '软件学报', '35', '2', '711-738', '10.13328/j.cnki.jos.007006', 'https://www.jos.org.cn/jos/article/abstract/7006'),
    175: ('深度强化学习中的知识迁移方法研究综述', '张启阳 陈希亮 曹雷 赖俊 盛蕾', 2023, '计算机科学', '50', '5', '201-216', '10.11896/jsjkx.220400235', 'https://www.jsjkx.com/CN/10.11896/jsjkx.220400235'),
    176: ('机器人学习方法综述', '曲威名 刘天林 林惟凯 罗定生', 2023, '北京大学学报（自然科学版）', '59', '6', '1069-1086', '10.13209/j.0479-8023.2023.086', 'https://xbna.pku.edu.cn/CN/10.13209/j.0479-8023.2023.086'),
    177: ('基于指标耦合的微观交通仿真参数标定方法', '高沛 周荣贵 周建 张旭冉', 2023, '交通信息与安全', '41', '5', '107-114', '10.3963/j.jssn.1674-4861.2023.05.011', 'http://www.jtxa.net/cn/article/doi/10.3963/j.jssn.1674-4861.2023.05.011'),
    178: ('驾驶模拟技术在交通仿真参数标定中的应用研究', '刘诗昆 唐易 刘永红', 2024, '系统仿真学报', '36', '6', '1359-1368', '10.16182/j.issn1004731x.joss.23-1549', 'https://www.china-simulation.com/CN/abstract/article/1004-731X/3417'),
    190: ('潜在空间中深度强化学习方法研究综述', '赵婷婷 孙威 陈亚瑞 王嫄 杨巨成', 2023, '计算机科学与探索', '17', '9', '2047-2074', '10.3778/j.issn.1673-9418.2211113', 'https://fcst.ceaj.org/CN/10.3778/j.issn.1673-9418.2211113'),
    191: ('基于预测编码的样本自适应行动策略规划', '梁星星 马扬 冯旸赫 张驭龙 张龙飞 廖世江 刘忠', 2022, '软件学报', '33', '4', '1477-1500', '10.13328/j.cnki.jos.006472', 'https://www.jos.org.cn/jos/article/abstract/6472'),
    218: ('基于模型的强化学习中可学习的样本加权机制', '黄文振 尹奇跃 张俊格 黄凯奇', 2023, '软件学报', '34', '6', '2765-2775', '10.13328/j.cnki.jos.006489', 'https://www.jos.org.cn/jos/article/abstract/6489'),
}

rows = []
for n in range(147, 219):
    row = {'number': n, 'status': 'verified', 'canonical': {}, 'issues': [], 'sources': [], 'verified_fields': [], 'notes': []}
    if n in chinese:
        title, authors, year, journal, volume, issue, pages, doi, url = chinese[n]
        row['canonical'] = dict(title=title, authors=authors.split(), year=year, container=journal, volume=volume, issue=issue, pages=pages, doi=doi, url=url)
        row['verified_fields'] = ['title', 'displayed_authors_and_order', 'year', 'container', 'volume', 'issue', 'pages', 'doi']
        if n == 190:
            row['sources'] = [
                {'url': 'https://ai.tust.edu.cn/yjsjy/yjsds/81acd4335dff434f8f3cce496c232b4a.htm', 'kind': 'author_university_profile', 'evidence': '赵婷婷作者所在高校主页第13项逐字列出五位作者、题名及2023,17(9):2047-2074。'},
                {'url': 'https://www.sciengine.com/parse/pdf/1673-9418/B48F50E3F88F4C858E55370520E37CED.pdf?attname=Review+of+Deep+Reinforcement+Learning+in+Latent+Space.pdf', 'kind': 'publisher_indexed_front_matter', 'evidence': '本轮网页检索返回原刊首部索引：题名、五位作者、2023/17(09)-2047-28及DOI；未下载PDF。'},
            ]
            row['notes'].append('原刊HTML当前403/证书域名异常；以作者高校主页与出版平台原刊首部索引交叉确认，不据访问失败判错。')
        else:
            row['sources'] = [{'url': url, 'kind': 'journal_official', 'evidence': '本轮原刊文章页/引用格式/HTML元数据逐项核实。'}]
        if n == 147:
            row['sources'].append({'url': 'https://doi.org/' + doi, 'kind': 'doi_official_resolver', 'evidence': '本轮200解析至万方该刊wxdtxjs202405001记录；期刊官方CNKI页核实题名、六位作者、50(5):831-842。'})
            row['notes'].append('Crossref不收录该DOI；官方DOI解析成功。')
        if n in (174, 191, 218):
            row['notes'].append('年份按原刊正式卷期与引用本文；citation_date为较早在线出版日期，不将其误作正式卷期年份。')
        if n == 177:
            row['notes'].append('DOI中的j.jssn与原刊citation_doi逐字一致，保留原拼写。')
            row['notes'].append('verified仅表示书目信息与原刊一致；DOI注册/解析状态由根代理集中实查单列，不能由原刊所印字符串推定注册正常。')
    elif n == 203:
        row['canonical'] = dict(title='Reinforcement Learning: An Introduction', authors=['Richard S. Sutton', 'Andrew G. Barto'], year=2018, edition='2nd', publisher='The MIT Press', place='Cambridge, MA', isbn='9780262039246', url='https://mitpress.mit.edu/9780262039246/reinforcement-learning/')
        row['sources'] = [{'url': row['canonical']['url'], 'kind': 'publisher_official', 'evidence': 'MIT Press书目页：second edition，两位作者，出版日期2018-11-13，出版社与ISBN。'}]
        row['verified_fields'] = ['title', 'authors_and_order', 'year', 'edition', 'publisher', 'isbn']
        row['notes'].append('书籍未列DOI，不作为错误；无需增加卷期或论文页码。')
    elif n == 206:
        url = 'https://papers.nips.cc/paper_files/paper/2012/hash/d1f491a404d6854880943e5c3cd9ca25-Abstract.html'
        row['canonical'] = dict(title='Robustness and risk-sensitivity in Markov decision processes', authors=['Takayuki Osogami'], year=2012, container='Advances in Neural Information Processing Systems', volume='25', pages='233-241', url=url)
        row['sources'] = [
            {'url': url, 'kind': 'conference_official', 'evidence': 'NIPS官方文章页列出题名、Takayuki Osogami及NIPS2012/25。'},
            {'url': 'https://papers.nips.cc/paper_files/paper/2012/file/d1f491a404d6854880943e5c3cd9ca25-Metadata.json', 'kind': 'conference_official_metadata', 'evidence': '官方metadata的page_first=233、page_last=241。'},
        ]
        row['verified_fields'] = ['title', 'author', 'year', 'container', 'volume', 'pages', 'url']
    elif (T / 'arxiv' / f'{n:03}.json').exists():
        data = json.loads((T / 'arxiv' / f'{n:03}.json').read_text())
        m = data['metadata']
        explicit_version = bool(re.search(r'v\d+$', data['queried_arxiv']))
        date = m['updated'] if explicit_version else m['published']
        row['canonical'] = dict(title=m['title'], authors=m['authors'], year=int(date[:4]), container='arXiv', arxiv=data['queried_arxiv'], returned_version=m['id'], first_submitted=m['published'], version_updated=m['updated'], doi=m['doi'], url='https://arxiv.org/abs/' + data['queried_arxiv'])
        row['sources'] = [{'url': row['canonical']['url'], 'kind': 'arxiv_official'}, {'url': data['source'], 'kind': 'arxiv_official_api', 'cache': 'arxiv/' + f'{n:03}.json'}]
        row['verified_fields'] = ['title', 'displayed_authors_initials_and_order', 'year', 'arxiv_id', 'url']
        row['notes'].append('已逐条人工核对所列作者缩写与顺序、完整题名及年份；使用原文arXiv著录，不改为会议/期刊版本。')
        if explicit_version:
            row['verified_fields'].append('specified_version')
            row['notes'].append('原文指定版本，年份按该版本updated日期；返回版本与指定版本相同。')
        else:
            row['notes'].append('原文未指定版本，年份按arXiv初次提交年份；后续updated年份不同不构成错误。')
        if n == 201:
            row['notes'].append('官方首作者名为JB Lanier，原文LANIER J B为其姓名缩写，正确。')
    else:
        assert n in (204, 208), n
        data = json.loads((T / 'crossref' / f'{n:03}.json').read_text())
        assert data['http_status'] == 200
        m = data['metadata']
        row['canonical'] = dict(title=m['title'][0], authors=[a['given'] + ' ' + a['family'] for a in m['author']], year=m['published']['date-parts'][0][0], container=m['container-title'][0], volume=m['volume'], issue=m['issue'], pages=m['page'], doi=m['DOI'], url=m['URL'])
        row['sources'] = [{'url': data['source'], 'kind': 'publisher_deposited_crossref_metadata', 'cache': 'crossref/' + f'{n:03}.json'}]
        if n == 208:
            row['sources'].append({'url': 'https://ojs.aaai.org/index.php/AAAI/article/view/20730', 'kind': 'conference_official', 'evidence': '官方How to Cite核实五位作者、2022、36(7):7637-7646及DOI。'})
        row['verified_fields'] = ['title', 'displayed_authors_initials_and_order', 'year', 'container', 'volume', 'issue', 'pages', 'doi']
    rows.append(row)

assert len(rows) == 72 and [r['number'] for r in rows] == list(range(147, 219))
(T / 'audit_147_218.json').write_text(json.dumps(rows, ensure_ascii=False, indent=2) + '\n')
print(json.dumps({'reviewed': len(rows), 'verified': sum(r['status'] == 'verified' for r in rows), 'correction_needed': [], 'unverified': []}, ensure_ascii=False))
