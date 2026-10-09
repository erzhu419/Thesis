"""Retain per-reference sources and current citation contexts for this review."""
import json
import re
from pathlib import Path

WORK = Path(__file__).resolve().parent
ROOT = WORK.parents[1]


def read(path):
    return json.loads(path.read_text())


refs = read(WORK / '成品文献与引用.json')
online = read(WORK / '在线元数据.json')
publishers = read(WORK / '期刊网页复核.json')
previous = read(ROOT / 'workpapers/三项闭合_20261003/primary_sources.json')
chinese = {r['number']: r for r in read(
    ROOT / 'workpapers/综述新增中文文献_20261006/references.json')}
notes = {}
for line in (ROOT / 'workpapers/三项闭合_20261003/文献逐项核对.md').read_text().splitlines():
    match = re.match(r'^\| (\d+) \|', line)
    if match:
        notes[int(match[1])] = line.rsplit('|', 2)[1].strip()

fallbacks = {
    22: ('https://jtgc.cbpt.cnki.net/portal/journal/portal/client/paper/JTGC_da6bd34e-4473-40bd-ab25-06c8a87baf30', '期刊官网登记信息与摘要'),
    23: ('https://zgglxb.chd.edu.cn/CN/10.19721/j.cnki.1001-7372.2021.07.003', '期刊官网登记信息与摘要'),
    24: ('https://zgglxb.chd.edu.cn/CN/10.19721/j.cnki.1001-7372.2023.02.017', '期刊官网登记信息与摘要'),
    168: ('https://www.ort.shu.edu.cn/CN/10.15960/j.cnki.issn.1007-6093.2023.03.005', '期刊官网登记信息与摘要'),
    170: ('https://transport.chd.edu.cn/cn/article/doi/10.19818/j.cnki.1671-1637.2025.06.019', '期刊官网登记信息与摘要'),
    172: ('https://ir.bjut.edu.cn/irpui/community/newsearch?info=%E7%BB%84%E5%90%88%E4%BC%98%E5%8C%96&select=keyword', '作者所在学校成果库：题名、作者、年卷期页与摘要；期刊网站连接失败'),
    174: ('https://zgglxb.chd.edu.cn/CN/10.19721/j.cnki.1001-7372.2024.04.025', '期刊官网登记信息与摘要'),
    176: ('https://cs.hit.edu.cn/_upload/article/files/1e/10/1856ecb9495dba4be992671faa6e/abbb87e2-50de-4061-ab04-d72f1926b1c7.pdf', '期刊所属学校提供的原文PDF'),
    185: ('https://gk.swufe.edu.cn/info/1085/1325.htm', '作者所在学校成果清单：出版信息；未取得期刊摘要与全文，不支持具体收益数字'),
    187: ('https://kzyjc.alljournals.cn/html/2022/12/20221202.htm', '期刊官网全文与登记信息'),
    191: ('https://www.ort.shu.edu.cn/CN/Y2023/V27/I2/49', '期刊官网登记信息与摘要'),
    193: ('https://www.lamda.nju.edu.cn/zhangzq/zzq_index_files/openmarl.pdf', '作者公开审稿版：研究内容；正式版期刊页未显示文章信息，不能用此稿的页码核定正式版页码'),
    199: ('https://lib.zjsru.edu.cn/25-2.24-5.pdf', '大学图书馆保存的期刊原版PDF；非本轮直连出版社取得'),
    208: ('https://www.sciengine.com/parse/pdf/1673-9418/B48F50E3F88F4C858E55370520E37CED.pdf?attname=Review+of+Deep+Reinforcement+Learning+in+Latent+Space.pdf', '出版平台原版PDF与摘要'),
}
special_notes = {
    5: '异步公交决策的背景；已改用arXiv v2，与原IJCAI条目为同一工作。',
    43: '单智能体共享策略与类别表征的前序研究；arXiv v2署名只有Yifan Zhang，不混用期刊版作者表。',
    71: 'NAU和标准NMU的原始方法；第4章明确本地算术输出采用NAU与NMU启发的二次修正，不冒称标准NMU。',
    80: '作者早期BAPR来源；现用公交条件适应配置和早期算子形式在正文与附录分别说明。',
    106: '作者的H2O+公交适配与完成率保障研究；原H2O+算法另引[10]。',
    132: '作者MC-WM来源；本文公交实验的收益仍以本文章节的实际表图为依据。',
    158: '作者CS-BAPR实现与配置来源；核对算术输出头的二次修正描述，不将该实现称为标准NMU。',
    172: '车速调控与乘车引导同时优化乘客出行成本和时距偏差；不与电动公交随机调度[171]混为同一工作。',
    185: '仅用于说明公交强化学习驻站已有研究；未扩写未取得全文的算法步骤或收益。',
    193: '用于开放多智能体任务、环境与智能体组成变化的分类；不作为所有单智能体非平稳环境的定义。',
    199: '离线强化学习研究综述；用于固定数据学习的信息条件和方法分类。',
    207: '移至第6章具体样本加权比较；环境样本损失变化加权与本文置信权重不同。',
}

ledger = []
for ref in refs:
    n, key = ref['number'], ref['key'].removeprefix('ref-')
    record = {'number': n, 'key': ref['key'], 'entry': ref['text'],
              'citation_contexts': ref['uses'], 'review_date': '2026-10-07'}
    metadata = online.get(str(n), {})
    if n in (5, 43):
        url = re.search(r'https://arxiv.org/abs/[^\s]+', ref['text'])[0].rstrip('.')
        record.update(source=url, verification='本轮arXiv官方记录及指定版本核对')
    elif metadata.get('kind') == 'arxiv':
        record.update(source=metadata['source'], verification='本轮arXiv官方题名、作者、日期与摘要核对')
    elif n in fallbacks:
        record.update(source=fallbacks[n][0], verification=fallbacks[n][1])
    elif n in chinese and publishers[str(n)].get('title_present'):
        record.update(source=publishers[str(n)]['source'], verification='本轮期刊官方登记信息、摘要或全文核对')
    elif metadata.get('kind') == 'crossref':
        record.update(source=metadata['source'], verification='本轮Crossref登记信息；结合已存出版社摘要、作者版及当前引用段落核对')
    elif n == 25:
        record.update(source='用户提供的刘鹏杰博士论文题名页', verification='沿用已核实题名页，不按文件名改题名')
    elif n == 159 or n >= 210:
        url = re.search(r'https://github.com/[^\s]+', ref['text'])[0].rstrip('.')
        record.update(source=url, verification='本轮GitHub API确认仓库存在、所属账户与仓库名称')
    else:
        raise ValueError(f'No recorded verification source: {n}')
    prior = previous.get(key, {})
    record['citation_note'] = special_notes.get(n, notes.get(prior.get('number'),
        chinese.get(n, {}).get('relevance', '对应方法的代码来源，不作为独立实验收益证据。')))
    ledger.append(record)

(WORK / '逐项文献核对.json').write_text(json.dumps(ledger, ensure_ascii=False, indent=2), encoding='utf-8')
print(f'{len(ledger)} references with sources and current citation contexts')
