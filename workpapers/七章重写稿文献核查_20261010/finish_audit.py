"""Deliver a citation-only Word copy and a per-reference evidence ledger."""
from copy import deepcopy
import importlib.util
import json
from pathlib import Path
import re
import shutil
from zipfile import ZipFile

from lxml import etree as ET

from audit import HERE, ROOT, SOURCE, NS, read, write

OUT = ROOT / 'output/排版完成_七章整合_文献核验_20261011.docx'
REPORT = ROOT / 'output/参考文献逐条核验_20261011.md'
spec = importlib.util.spec_from_file_location('merge_helper', ROOT / 'workpapers/七章重写稿整合_20261010/integrate.py')
helper = importlib.util.module_from_spec(spec)
spec.loader.exec_module(helper)
refs = read(HERE / '当前文献与引用.json')
live = read(HERE / '在线原始记录.json')
extra = read(HERE / '疑点一手来源.json')
web = read(HERE / '专项证据补记.json')
checks = read(HERE / '引用结构检查.json')
additions = {172: '10.16097/j.cnki.1009-6744.2023.06.017',
             185: '10.16097/j.cnki.1009-6744.2024.05.016'}


def edit_word():
    backup = HERE / '排版完成_七章重写稿整合_修改前.docx'
    if not backup.exists():
        shutil.copy2(SOURCE, backup)
    with ZipFile(SOURCE) as source:
        original = source.read('word/document.xml')
        spans = helper.body_spans(original)
        replacements = []
        for n, doi in additions.items():
            start, end = spans[refs[n-1]['block']]
            paragraph = original[start:end]
            assert doi.encode() not in paragraph
            text = list(re.finditer(rb'(<w:t(?:\s[^>]*)?>)(.*?)(</w:t>)', paragraph))[-1]
            end_text = text.group(2) + (' DOI:' + doi + '.').encode()
            paragraph = paragraph[:text.start(2)] + end_text + paragraph[text.end(2):]
            replacements.append((start, end, paragraph))
        updated = original
        for start, end, paragraph in reversed(replacements):
            updated = updated[:start] + paragraph + updated[end:]
        with ZipFile(OUT, 'w') as target:
            for info in source.infolist():
                target.writestr(deepcopy(info), updated if info.filename == 'word/document.xml' else source.read(info.filename))
    # This detects any accidental body/object/style change from the bibliography patch.
    with ZipFile(SOURCE) as source, ZipFile(OUT) as target:
        assert source.namelist() == target.namelist()
        assert [name for name in source.namelist() if source.read(name) != target.read(name)] == ['word/document.xml']
        a, b = source.read('word/document.xml'), target.read('word/document.xml')
        sa, sb = helper.body_spans(a), helper.body_spans(b)
        assert len(sa) == len(sb)
        changed = [i for i, ((x,y), (u,v)) in enumerate(zip(sa, sb)) if a[x:y] != b[u:v]]
        assert changed == [refs[n-1]['block'] for n in additions]
    write('文献补全与内容保留.json', dict(output=str(OUT), changed_entries=list(additions),
                                     changed_blocks=changed, other_package_parts_identical=True,
                                     all_other_document_blocks_identical=True))


def metadata_source(n):
    if n in (168, 191, 193):
        return web[str(n)]['source'], web[str(n)]['source_type']
    key = {172: '172-http', 185: '185-http', 187: '187-http'}.get(n)
    if key:
        return extra[key]['source'], '期刊官网原始条目及摘要/全文'
    rec = extra.get(str(n) + '-crossref', live[str(n)])
    if n == 7:
        return extra['7-author-order']['source'], 'Crossref出版字段；作者顺序另核对DLR及IEEE正式页面'
    if n == 25:
        return rec['source'], '用户提供的博士论文题名页；不是按文件名推断'
    if rec.get('kind') == 'repository':
        return rec['source'], 'GitHub官方API：仓库、账户、公开状态'
    if rec.get('kind') == 'arxiv':
        url = 'https://arxiv.org/abs/' + refs[n-1]['arxiv']
        return url, 'arXiv官方记录；带版本号条目另核该版本作者与日期'
    if rec.get('kind') == 'publisher-pdf':
        return rec['source'], '原版论文PDF；核首页、正文及页码'
    return rec['source'], 'Crossref原始出版登记' if rec.get('kind') == 'crossref' else '期刊原始登记及摘要/全文'


def build_report():
    ledger = []
    for ref in refs:
        n = ref['number']
        source, level = metadata_source(n)
        rec = dict(number=n, entry=ref['text'], source=source, source_level=level,
                   existence='已核实', metadata='已核对，未发现实质著录错误',
                   citation_placement='已核对，未发现实质错引',
                   citation_note=web.get(str(n), {}).get('review', ref['prior_note']),
                   current_citation_contexts=ref['uses'])
        if n in additions:
            rec['entry'] += ' DOI:' + additions[n] + '.'
            rec['metadata'] = '原作者、题名、年卷期页正确；本轮补全期刊官网DOI'
        if n == 7:
            rec['citation_note'] = 'Crossref将Wiessner提前，但IEEE正式页面和DLR与现稿前三作者一致，保留原顺序。复姓缩写属于不同登记形式，不作为虚构或错引。'
        if n in (39, 87, 141):
            rec['citation_note'] += ' 作者全名已与arXiv对应；不把复姓/名字缩写差异误判为另一篇文献。'
        ledger.append(rec)
    write('本版逐项核验与引用落点.json', ledger)
    lines = [
        '# 参考文献逐条核验', '',
        '核验完成日期：2026-10-11。原始在线记录主要取得于2026-10-10。', '',
        '基准：`排版完成_七章重写稿整合_20261010.docx`。仅在副本中补充[172]、[185]的DOI；不润色、改写或重排正文内容。', '',
        '## 结论', '',
        '- 共214条：208条论文、预印本或学位论文，6条代码仓库。所有条目均找到真实来源，没有发现虚构条目。',
        '- 逐条核对现稿所列作者、题名、年份、出版载体及适用的卷期页码、DOI或arXiv版本；未发现需要纠正的实质著录错误。补充两处已确认的DOI。',
        '- 对照本版134个含引文的段落及对应原摘要、必要方法/结论内容，没有发现需要调整位置的实质错引。',
        '- 359个内部文献链接均指向对应条目，显示编号正确；没有未被引用条目、重复主标识或未链接的数字式引文。', '',
        '## 两处补全', '',
        '| 条目 | 补充内容 | 一手依据 |', '|---|---|---|',
        '| [172] 考虑车速调控和乘车引导的公交运行优化控制策略 | DOI:10.16097/j.cnki.1009-6744.2023.06.017 | [期刊2023年第6期](http://www.tseit.org.cn/CN/Y2023/V23/I6) |',
        '| [185] 基于深度强化学习的单线路公交动态驻站控制策略研究 | DOI:10.16097/j.cnki.1009-6744.2024.05.016 | [期刊2024年第5期](http://www.tseit.org.cn/CN/Y2024/V24/I5) |', '',
        '## 容易误判的条目', '',
        '- [7] SUMO论文：Crossref的作者顺序与IEEE正式页面、DLR机构记录不一致。现稿前三作者与正式页面一致，因此没有按Crossref错误地调换作者。',
        '- [10]、[43]、[131]等：指定arXiv版本的年份与首次上传年份可能不同。按现稿所引用版本核对；没有把预印本、期刊版的作者表混在一起。',
        '- [16]、[48]：Crossref的title字段只含主标题，副标题需要结合正式记录核对；现稿完整题名正确，不删除副标题。',
        '- [25]：题名页为《不确定环境下多线路公交信号协同优先方法研究》，与现稿一致，不按文件名中的旧题名更改。',
        '- [165]：官网结构化volume字段为空，但“引用本文”明确给出40(3)，因此保留40卷。',
        '- [193]：作者审稿稿和正式版题名略有差别；已读取Science China Press提供、ResearchGate页面展示的正式排印全文首页，核定2025、55(2)、217–268及DOI。不是根据审稿稿猜测正式页码。',
        '- [159]、[210]–[214]是代码来源，不是6篇额外的学术论文；其位置用于说明实现出处，不作为独立实验收益证据。', '',
        '## 引用内容检查', '',
        '本轮以新整合的七章为准重新检查，不直接沿用旧稿的引用位置。以下是较易错配的内容：', '',
        '- [4]：第3章明确原式为负指数，并区分本文比较采用的秒制尺度与原文归一化动作，没有声称两组实验设置完全相同。',
        '- [35]、[41]、[42]：分别对应驻站/跳站合作、驻站/专用车道加速分层控制、繁忙多线路走廊的图结构与排队，没有混为同一种公交动作设置。',
        '- [43]：第3章引用的约-430k、-530k、-580k、-980k可在原文第15页找到，方法对应正确。',
        '- [68]、[69]：用于CVaR风险目标与受限重加权解释；附录明确轨迹预算与逐阶段独立更新不同，没有把该理论说成RE-SAC实际训练目标。',
        '- [71]、[158]：正文区分标准NMU与本地二次修正实现，没有把受启发的算术输出直接等同标准NMU。',
        '- [80]、[106]、[132]：分别注明作者BAPR、H2O+公交适配和MC-WM来源；本文公交实验仍由本文章节图表支撑，没有借小论文引文替代本组实验数据。',
        '- [130]、[131]：区分允许真实交互微调和主要依赖离线记录的两种信息条件，没有把机器人验证写成本论文的实车验证。',
        '- [193]：用于开放多智能体的任务、环境和组成变化，不作为所有单智能体非平稳环境的定义。',
        '- [207]：第6章区分原文基于模型更新损失变化的样本加权与本文置信权重。', '',
        '## 核验边界', '',
        '所有条目的真实性与现稿著录字段均进行了核对，但不意味着逐字读完了208篇原文。引用内容依据原摘要及必要方法、结论或原文片段核对；部分详细依据使用此前保存的原始出版记录。', '',
        '部分期刊HTTPS、IEEE机器人验证和SciEngine动态页面访问失败；[168]、[191]由web工具读取期刊官网，[172]、[185]、[187]由HTTP期刊官网读取，[193]通过正式排印版副本核定。没有把连接失败当作虚构，也没有将通用平台首页当作文献证据。', '',
        'arXiv条目作为预印本著录，只确认对应作品、版本及信息存在，不将其标为已完成同行评审。', '',
        '本报告不以“100%绝无错误”替代证据；下表逐条给出来源、当前落点和核对结论。', '',
        '## 逐条记录', '',
        '| 编号与题名 | 真实性及著录 | 当前引用位置 | 内容对应与来源 |',
        '|---|---|---|---|'
    ]
    for rec in ledger:
        n = rec['number']
        title = refs[n-1]['text'].split('. ', 1)[1].split('[', 1)[0]
        sections = list(dict.fromkeys(u['section'] for u in refs[n-1]['uses']))
        note = rec['citation_note'].replace('|', '/').replace('\n', ' ')
        link = rec['source'] if not rec['source'].startswith('/') else '<' + rec['source'] + '>'
        lines.append('| [' + str(n) + '] ' + title.replace('|', '/') + ' | 已找到真实来源；' + rec['metadata'] + ' | ' + '；'.join(sections) + ' | ' + note + ' [核对来源](' + link + ') |')
    REPORT.write_text('\n'.join(lines) + '\n')
    print(json.dumps(dict(output=str(OUT), report=str(REPORT), entries=len(ledger)), ensure_ascii=False))


if __name__ == '__main__':
    edit_word()
    build_report()
