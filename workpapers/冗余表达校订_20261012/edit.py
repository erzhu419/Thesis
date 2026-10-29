"""Apply the supplied wording review while retaining native Word objects."""
from collections import defaultdict
from copy import deepcopy
from pathlib import Path
from zipfile import ZipFile
import importlib.util
import json
import re
import shutil
import sys

from lxml import etree as E

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
BASE = ROOT / 'output/排版完成_编号符号引用复核_20261012.docx'
OUT = ROOT / 'output/排版完成_冗余表达与用词校订_20261012.docx'
REVIEW = ROOT / 'markdown/合并稿审阅_冗余与学术表达_20261012.docx'
sys.path.insert(0, str(ROOT / 'workpapers/全文引用符号复核_20261012'))
import audit
W, M, NS, text = audit.W, audit.M, audit.NS, audit.text
sys.path.insert(0, str(ROOT / 'workpapers/七章重写稿整合_20261010'))
from integrate import body_spans

XMLSPACE = '{http://www.w3.org/XML/1998/namespace}space'


class Editor:
    def __init__(self):
        with ZipFile(BASE) as z:
            self.parts = [(info, z.read(info.filename)) for info in z.infolist()]
            self.raw = z.read('word/document.xml')
        self.root = E.fromstring(self.raw)
        self.body = list(self.root.find(W + 'body'))
        self.before = [E.tostring(p) for p in self.body]
        self.deleted = set()
        self.moved = {}
        self.log = []
        self.labels = json.loads((ROOT / 'workpapers/全文引用符号复核_20261012/编号与引用检查.json').read_text())['equations']
        self.source_math = {i: [deepcopy(n) for n in p.iter(M + 'oMath')] for i,p in enumerate(self.body)}
        self.citations = {}
        for p in self.body:
            for link in p.iter(W + 'hyperlink'):
                if (link.get(W + 'anchor') or '').startswith('ref-'):
                    for number in re.findall(r'\d+', text(link)):
                        self.citations.setdefault(number, deepcopy(link))

    def record(self, i, item, old):
        self.log.append({'block': i, 'item': item, 'before': old, 'after': text(self.body[i])})

    def phrase(self, i, old, new, item):
        p = self.body[i]
        nodes = list(p.iter(W + 't'))
        value = ''.join(n.text or '' for n in nodes)
        start = value.find(old)
        if start < 0:
            raise ValueError(('phrase not found', i, old))
        before = text(p)
        end = start + len(old)
        offset = 0
        first = True
        for n in nodes:
            s = n.text or ''
            lo, hi = max(start-offset, 0), min(end-offset, len(s))
            if lo < hi:
                n.text = s[:lo] + (new if first else '') + s[hi:]
                n.set(XMLSPACE, 'preserve')
                first = False
            offset += len(s)
        self.record(i, item, before)

    def full(self, i, contents, item):
        p = self.body[i]
        assert p.tag == W + 'p' and not p.findall('.//' + W + 'drawing') and not p.findall('.//' + W + 'object'), i
        before = text(p)
        props = deepcopy(p.find(W + 'pPr'))
        rprops = next((deepcopy(n) for n in p.findall('.//' + W + 'rPr')), None)
        starts = [deepcopy(n) for n in p if n.tag == W + 'bookmarkStart']
        ends = [deepcopy(n) for n in p if n.tag == W + 'bookmarkEnd']
        for n in list(p):
            p.remove(n)
        if props is not None:
            p.append(props)
        p.extend(starts)
        def plain(value):
            if not value:
                return
            run = E.SubElement(p, W + 'r')
            if rprops is not None:
                run.append(deepcopy(rprops))
            t = E.SubElement(run, W + 't')
            t.text = value
            t.set(XMLSPACE, 'preserve')
        for value in (contents if isinstance(contents, list) else [contents]):
            if not isinstance(value, str):
                p.append(deepcopy(value))
                continue
            offset = 0
            for match in re.finditer(r'\[(\d+(?:[,，]\s*\d+)*)\]', value):
                plain(value[offset:match.start()] + '[')
                tokens = re.split(r'(\d+)', match[1])
                for token in tokens:
                    if token.isdigit():
                        link = deepcopy(self.citations[token])
                        ts = list(link.iter(W + 't'))
                        ts[0].text = token
                        for t in ts[1:]: t.text = ''
                        p.append(link)
                    else:
                        plain(token)
                plain(']')
                offset = match.end()
            plain(value[offset:])
        p.extend(ends)
        self.record(i, item, before)

    def math(self, i, n):
        return deepcopy(self.source_math[i][n])

    def delete(self, i, item):
        assert self.body[i].find('.//' + W + 'sectPr') is None, ('section boundary', i)
        self.deleted.add(i)
        self.log.append({'block': i, 'item': item, 'before': text(self.body[i]), 'after': ''})

    def cell(self, i, row, col, value, item):
        tr = self.body[i].findall(W + 'tr')[row]
        tc = tr.findall(W + 'tc')[col]
        ts = list(tc.iter(W + 't'))
        before = text(tc)
        assert ts and not tc.findall('.//' + M + 'oMath')
        ts[0].text = value
        ts[0].set(XMLSPACE, 'preserve')
        for n in ts[1:]: n.text = ''
        self.log.append({'block': i, 'cell': [row,col], 'item': item, 'before': before, 'after': text(tc)})

    def rename_heading(self, i, old, new, item):
        self.phrase(i, old, new, item)
        for j,p in enumerate(self.body):
            if audit.style(p).startswith('TOC') and old in text(p):
                self.phrase(j, old, new, item + '/目录')

    def ordered(self):
        moved_indices = {i for indices in self.moved.values() for i in indices}
        for i,p in enumerate(self.body):
            if i not in self.deleted and i not in moved_indices:
                yield i,p
            for j in self.moved.get(i, []):
                yield j,self.body[j]

    def renumber(self):
        aliases = {'3-31':'3-27', '5-24':'5-18', '5-25':'5-18', '5-26':'5-19'}
        by_index = {r['i']:key for key,r in self.labels.items() if re.fullmatch(r'[356]-\d+', key)}
        counters, mapping = defaultdict(int), {}
        for i,p in self.ordered():
            if i in by_index:
                old = by_index[i]
                ch = old.split('-')[0]
                counters[ch] += 1
                mapping[old] = ch + '-' + str(counters[ch])
        for key,target in aliases.items():
            mapping[key] = mapping[target]
        anchor_aliases = {}
        for key,target in aliases.items():
            src = self.body[self.labels[key]['i']]
            dst = self.body[self.labels[target]['i']]
            target_anchor = next(dst.iter(W + 'bookmarkStart')).get(W + 'name')
            for mark in src.iter(W + 'bookmarkStart'):
                anchor_aliases[mark.get(W + 'name')] = target_anchor
        for i,p in self.ordered():
            for n in p.iter(W + 'instrText'):
                if n.text:
                    n.text = re.sub(r'\b(REF|PAGEREF)\s+([^\s\\]+)',
                        lambda m:m[1]+' '+anchor_aliases.get(m[2],m[2]), n.text)
            for n in p.iter(W + 'hyperlink'):
                anchor = n.get(W + 'anchor')
                if anchor in anchor_aliases:
                    n.set(W + 'anchor', anchor_aliases[anchor])
            # Replace each reference once in its original coordinate system.
            for paragraph in ([p] if p.tag == W + 'p' else p.findall('.//' + W + 'p')):
                label_cell = p.findall(W + 'tr')[0].findall(W + 'tc')[-1] if i in by_index else None
                if label_cell is not None and paragraph in list(label_cell.iter(W + 'p')):
                    continue
                nodes = list(paragraph.iter(W + 't'))
                value = ''.join(n.text or '' for n in nodes)
                spans = []
                for n in nodes:
                    start = sum(len(s[1].text or '') for s in spans)
                    spans.append((start,n))
                matches = list(re.finditer(r'([（(])([356]-\d+)([）)])', value))
                for match in reversed(matches):
                    if match[2] not in mapping: continue
                    new = match[1]+mapping[match[2]]+match[3]
                    first, offset = True, match.start()
                    end = match.end()
                    for start,n in spans:
                        s = n.text or ''
                        lo,hi = max(offset-start,0),min(end-start,len(s))
                        if lo<hi:
                            n.text = s[:lo]+(new if first else '')+s[hi:]
                            n.set(XMLSPACE,'preserve')
                            first=False
            if i in by_index:
                ch,newnum = mapping[by_index[i]].split('-')
                label = p.findall(W + 'tr')[0].findall(W + 'tc')[-1]
                ts = list(label.iter(W + 't'))
                ts[0].text = '('+ch+'-'
                for n in ts[1:-1]:
                    if (n.text or '').strip(): n.text = newnum
                ts[-1].text = ')'
        (HERE / '公式编号对应.json').write_text(json.dumps(mapping,ensure_ascii=False,indent=2))
        return mapping

    def save(self):
        spans = body_spans(self.raw)
        raw_by_index = {i:self.raw[a:b] for i,(a,b) in enumerate(spans)}
        output = []
        for i,p in self.ordered():
            output.append(raw_by_index[i] if E.tostring(p)==self.before[i] else E.tostring(p,encoding='utf-8',with_tail=False))
        raw = self.raw[:spans[0][0]] + b''.join(output) + self.raw[spans[-1][1]:]
        with ZipFile(OUT,'w') as z:
            for info,value in self.parts:
                z.writestr(info,raw if info.filename=='word/document.xml' else value)
        (HERE / '逐处修改记录.json').write_text(json.dumps(self.log,ensure_ascii=False,indent=2))
        print(json.dumps({'output':str(OUT),'operations':len(self.log),'removed_blocks':len(self.deleted),'moved_blocks':self.moved},ensure_ascii=False))


def apply_review(e):
    # The adviser-edited Chinese abstract remains unchanged.
    e.phrase(72, 'An arrival-event-based control model takes decisions after required passenger boarding and alighting and connects vehicle and stop information, forward and backward headways, a holding action, and passenger time costs.', 'An event-driven controller selects the additional holding time after required passenger boarding and alighting have been completed. Its observations describe vehicles, stops, headways, and passenger-service conditions available at the decision time. Policies are trained using either a headway-based reward or a passenger-time-based reward.', 'E02')
    e.phrase(72, 'All bus-control studies in the thesis use a single holding action.', 'In all bus-control experiments, the only control variable is additional holding time.', 'E03')
    e.phrase(75, 'The bus adaptation organizes decisions for simultaneously arriving vehicles', 'The bus-control implementation handles simultaneous decision events', 'E04')
    e.phrase(75, '1.25-times demand', 'a demand level 1.25 times the baseline', 'E04')
    e.phrase(76, 'matched source-interaction amounts and execution rules', 'the same number of source-simulator interactions and the same execution rules', 'E04')
    e.delete(77, 'E05')
    e.delete(61, '排版提示残留')

    # Retain the literature lineage; the detailed RL taxonomy is in Chapter 2.
    e.full(272, '强化学习驻站研究借鉴价值学习、演员评论家和策略优化等方法[181,182]，其策略表示与探索设计影响运行经验的利用及新动作的尝试[183,184]。深度Q网络（Deep Q-Network，DQN）、深度确定性策略梯度（Deep Deterministic Policy Gradient，DDPG）、信赖域策略优化（Trust Region Policy Optimization，TRPO）和近端策略优化（Proximal Policy Optimization，PPO），分别为离散动作评价、连续控制及限制策略更新幅度提供求解思路[31,32,33,34]。是否使用环境预测，以及怎样处理不确定条件下的不利结果，进一步形成基于模型、无模型与鲁棒学习的不同路线[28,29,30]。这些方法在公交中的应用，仍需与车辆事件、服务目标和允许的驻站范围相对应。', 'C13')
    e.delete(273, 'C13')
    e.phrase(448, '平台介绍不把某一次标定检验扩大为所有任务的统一保证，也不将仿真结果替代真实道路闭环验证。', '标定结果的适用范围限于相应场景与检验指标，真实道路闭环效果仍需现场验证。', 'B07')

    e.phrase(458, '公交运行的物理控制权限', '公交运行的控制范围', 'D05')
    e.full(459, '环境本身的随机性与有限知识造成的估计不足可以同时存在。价值网络之间的分歧反映所用估计器的一致程度，不能直接解释为环境随机方差，也不意味着已经区分出每一种物理扰动的来源。平稳随机运行的具体条件在第3.2.1节给出。', 'C01')
    e.full(460, '完整方法与SAC的比较检验总体控制效果，类别编码和正则项对照则分别分析经验表示与价值修正的作用。策略分别依据车头时距目标和乘客时间目标训练，服务评价结合计划时距偏差、前后不均衡度、人均广义时间成本及行程完成情况。', 'C01')
    e.phrase(536, '以下分别从最不利动作评价和局部乘客成本出发，说明本章学习目标的依据。', '', 'C03')
    e.phrase(567, '这几项分别作用于动作选择、当前价值拟合与自举目标，不能仅笼统地称为“增加鲁棒性”。', '这些修正分别作用于动作选择、当前价值拟合与后继价值目标。', 'B04')
    e.phrase(569, '模型部分由此明确了控制输入、动作、环境奖励和网络价值的对应关系；训练部分再给出各目标修正、经验回放与策略更新的具体计算。', '', 'B06')
    e.full(577, 'Embedding-SAC与RE-SAC的完整网络结构分别按原研究保留[43,67]，具体宽度见第3.2.5节；两项完整方法的结果不作为仅改变嵌入的单因素比较。类别表示的独立作用在第3.4.5节另行检验。', 'C02')
    e.full(586, ['设集成由',e.math(586,0),'个价值网络',e.math(586,1),'组成，当前网络的均值与标准差按式(3-27)计算。'], 'C02')
    e.delete(587, 'C02/重复均值定义')
    e.full(590, ['式(3-27)定义的当前价值网络样本标准差用于策略目标和价值回归损失；', e.math(590,1), '由目标网络计算，用于后继价值修正。两者分别是标准差和方差，网络参数及相应系数均不互换。'], 'C02')
    e.full(785, '本组的策略训练和固定策略评价均在快速公交事件仿真器中进行，不在SUMO微观仿真器中执行。道路、站点、需求和时刻表规定多线路场景，事件仿真器依据相应运行参数产生状态与动作反馈。比较保持同组车辆供给、需求、交通随机条件和驻站范围一致；基础资料及运行参数用于建立场景，不作为预先采集的目标域策略训练转移。', 'A02/已核实运行接口')
    e.phrase(799, '较好的平均奖励和物理时距指标', '较好的平均奖励、计划时距偏差和前后车头时距不均衡度', 'D04')
    for i in range(784,804):
        p=e.body[i]
        if '正常需求' in text(p):
            e.phrase(i, '正常需求', '基准需求（倍率为1）' if i==800 else '基准需求', 'A07')

    e.phrase(858, '无邻车时，所核对源环境可以不生成有效控制反馈，SUMO接口另有兜底奖励，必须结合实际采样过滤规则解释，不能把两者无条件拼为同一训练转移。', '本章快速事件仿真接口只在至少存在一侧邻车且满足控制站点条件时生成观测与奖励；无前后邻车的事件不调用驻站策略，也不作为独立控制转移进入经验回放。', 'A01/已核实观测过滤')
    e.phrase(862, '时距目标和乘客时间目标分别训练，不能在同一组学习记录中因结果较好而事后交换目标名称。', '两类奖励分别用于独立训练，训练目标与相应评价结果一一对应。', 'B08')
    e.delete(827, 'C05/理论与公交实现的对应集中于4.2.2')
    e.phrase(865, '近期反馈用于调节训练中的保守程度；公交评价时固定策略参数，不进行部署期间的梯度更新，也不把该反馈称为通过跨任务元训练得到的交通模式后验。', '近期反馈用于调节训练中的保守程度；公交评价时固定策略参数。', 'C05')
    e.full(882, ['运行条件隐藏且影响预测时，理论价值需要以相应历史或信念为条件，例如',e.math(882,0),'。公交实现以已定义观测拟合价值，并利用近期统计反馈约束训练，具体计算如下。'], 'C05')
    e.phrase(893, '该权重反映近期统计对历史信息的影响，不是某种交通模式的后验概率，也不能直接输出驻站时长。', '该权重反映近期统计对历史信息的影响，用于调节策略的保守评价。', 'C05')
    e.phrase(879, '价值判断、学习强度和策略结构', '价值判断、保守评价强度和策略结构', 'D07')
    e.phrase(907, '模型由此明确，CS-BAPR并不是在RE-SAC上无差别叠加全部正则项，而是针对近期反馈、稳定化及策略响应采用特定组合。', '近期反馈调节策略的保守评价强度，奖励缩放与熵温度限制用于稳定训练，软符号约束则作用于完整策略的响应。', 'B06')
    e.full(908, '符号导数约束的解释依赖输入变量的类型及归一化方式。对线路和站点等类别编号计算的导数，只反映网络在所用数值编码下的变化，不具有类别连续变化的物理含义。连续运行量的理论分析固定离散类别，并采用相同归一化尺度；样本平均导数损失较小，也不足以推出整个外推区域的统一误差界。本章因此通过完整方法与相同输出结构下的约束消融评价该设计的作用，其结果不构成对真实交通因果关系的识别。', 'B01/示例替换')
    e.full(922, '近期反馈采用第4.2.3节给出的统计量与长度权重。前100次更新按该节的预热设置关闭反馈，其后按以下诊断、平滑和重置次序更新。', 'C04')
    e.full(923, ['最近256条在策略样本用于计算奖励偏差与价值分歧，统计量的组成见第4.2.3节。价值分歧只保留标准差的上升，记相邻两次诊断值为',e.math(923,0),'和',e.math(923,1),'，其单向相对增量为'], 'C04')
    e.full(926, ['按第4.2.3节综合奖励偏差、式(4-24)的价值分歧增量及权重范数差，取最大值后以0.1进行指数平滑，得到',e.math(926,0),'。平滑值用于更新式(4-15)的证据权重。'], 'C04')
    e.full(927, '历史长度索引取0至19，初始权重相同。每次诊断先将证据权重与原长度权重相乘并归一化，再按0.02的预设重置比例向长度0分配权重，其余权重向下一长度移位，截断后再次归一化。历史长度按诊断次数计量，与物理运行时长的对应取决于诊断频率。', 'C04')
    e.full(928, '更新后的长度权重代入式(4-16)，计算反馈权重及价值标准差的保守系数，再与集成均值和熵项共同更新策略。前100次更新关闭反馈；启用后，反馈权重越大，对估计分歧较大动作的惩罚越强。', 'C04')
    e.rename_heading(930, '4.3.2 策略表示', '4.3.2 输出结构及其响应性质', 'C06')

    meanings = [
        '线路与运行方向对应的类别编码，用于区分多线路服务对象。',
        '当前车辆对应的班次服务类别编码。',
        '车辆在线路内的服务次序编码。',
        '按仿真时刻确定的小时类别编码。',
        '车辆运行方向的类别编码。',
        '决策时获得的本线路前向与后向车头时距，单位为秒。',
        '本次服务对应站点的候车人数，单位为人。',
        '线路的固定计划间隔，单位为秒；相邻班次的计划时差另用于奖励。',
        '已经完成的必要上下客时间，单位为秒。',
        '当前运行时刻，单位为秒。',
        '前车存在时为对应计划前向时距减去观测前向时距，单位为秒；无前车时取0。',
        '当前共线服务的前向、后向车辆间隔，单位为秒。',
        '路段有效速度的两倍，原始有效速度以米/秒计；该量为运行观测。']
    for table_i in (788,837):
        e.cell(table_i,0,2,'运行含义、单位与编码','C06')
        for row,value in enumerate(meanings,1):
            e.cell(table_i,row,2,value,'C06')
    e.phrase(789, '线路、车辆、站点、时段和方向按其身份意义使用，不以编号间距离解释交通状态差异。', '类别编号用于区分服务身份，不表示物理距离；运行观测在决策时取得，不读取未来服务结果。', 'C06/共同表注') if '线路、车辆、站点、时段和方向按其身份意义使用，不以编号间距离解释交通状态差异。' in text(e.body[789]) else None
    e.phrase(840, '类别编码的数值延拓只用于网络计算，其导数不能直接解释为线路编号或站点类别连续改变时的物理效应。', '类别编号用于区分服务身份，不表示物理距离；运行观测在决策时取得，不读取未来服务结果。类别编码的数值延拓只用于网络计算，导数的解释见第4.2.3节。', 'C06/共同表注')

    # Batch composition is defined once before the detailed feature table.
    e.full(1142, '状态同时描述当前车辆、其他待决策车辆及本批已经选择的动作。最多设置16个车辆记录位置，每个位置包含12项局部特征、有效车辆标记及本批已选动作，共14维；另外记录当前物理时刻、待决策车辆数、当前决策序号、系统候车人数和车内人数五个全局量，形成16×14+5=229维输入。乘客人数和需求量按容量归一化，空位置由有效标记排除；超过16辆待决策车辆时终止该次运行。该表示与多线路离线任务的16维输入分别使用。控制信息的来源见表5-2，局部特征及固定尺度见表5-3。', 'C07')
    e.full(1161, '局部特征由事件观测去除已单独放入全局的时间和系统人数后形成，人数与需求按容量归一化，不再单独保留容量数值。有效标记区分实际车辆与补齐位置；同批已选动作仅表示已经确定的控制要求，其尚未发生的执行结果不进入输入。', 'C07')
    e.delete(1162, 'C07/信息含义已由表5-2和估计公式说明')
    e.full(1150, '成本以实际经过时间累计，单位为加权后的乘客秒。同批中间转移成本为0；批内顺序编码保留动作之间的条件依赖，不推进仿真时钟，也不重复累计乘客时间成本。', 'B05/C08')
    e.phrase(1153, '乘客发生换乘时，计入实际发生的各段候车和车内时间，但分母仍按该任务规定的已出发乘客集合计算，不因同一乘客有多个乘车段而重复增加人数。未完成行程者只累计评价终点前已经发生的时间，不补估时域外的剩余行程。', '乘客时间的统计范围与评价分母按式(5-10)定义。', 'C08')
    e.delete(1154, 'C08/批内零时间说明合并')
    e.phrase(1158, '为以秒计的人均广义时间成本，未完成行程者截至评价终点已经发生的时间仍计入，不估计终点后的剩余行程。', '为以秒计的人均广义时间成本。换乘旅客累计实际发生的各段候车与车内时间，分母仍按首次进入系统的旅客计算，不重复增加人数；未完成行程者截至评价终点已经发生的时间仍计入，不估计终点后的剩余行程。', 'C08')
    e.full(1188, ['仅累计环境奖励的状态价值',e.math(1188,0),'、由Q网络与当前策略计算的软状态价值',e.math(1188,3),'，以及由数据内动作价值期望分位回归获得的IQL状态价值',e.math(1188,1),'具有不同定义。软状态价值不需要另设状态价值网络；IQL状态价值偏向目标记录支持的较好动作价值，不是对当前策略全部动作取普通平均。'], 'B07')
    e.phrase(1193, '补造目标记录', '生成新的目标域观测', 'D08')
    advantage=e.math(1224,0)
    last=advantage.findall('.//'+M+'t')[-1]
    if (last.text or '').endswith('.'): last.text=last.text[:-1]
    e.full(1223, ['离线初始化采用第5.2.4节的IQL更新，状态价值按式(5-17)学习，策略按式(5-18)的优势权重拟合已有动作。其中，数据内优势可记为',advantage,'。'], 'C09')
    for i in (1224,1225,1226): e.delete(i,'C09/重复IQL定义')
    e.full(1230, '初始化后，先固定策略在源仿真中采集5000个预热事件，再等量抽取目标离线数据和源经验更新动作价值网络。策略网络改用SAC式熵正则化目标，后继价值沿用式(5-19)的混合定义，将离线状态价值与当前策略对应的目标软价值相结合[10,28,89]。', 'C09')
    e.delete(1231, 'C09/重复混合后继价值定义')
    e.phrase(1232, '式中的0.1与0.9', '式(5-19)中的0.1与0.9', 'C09')
    e.phrase(1198, '再组合当前车辆、全部车辆、已选动作车辆与未选动作车辆的信息，并形成256维表示。', '将当前车辆特征、全部有效车辆特征的均值与最大值、已选动作车辆特征的均值及尚未决策车辆特征的均值，与全局量合并，形成256维表示。', 'C07/保留完整聚合定义')

    e.full(1462, 'H2O/H2O+允许源目标动力学存在差异，但经验混合与样本加权并不直接修正已记录的后继事件。目标数据未覆盖当前策略所需的状态动作时，重复回放也无法提供新的目标反馈；有限轨迹上的分类器估计还会限制比率修正的可靠性。因此，当源仿真标定精度下降时，本章研究如何利用固定目标记录修正动作后果预测，降低价值学习对未经修正的源模型预测的依赖。', 'B02/示例替换')
    e.full(1512, '策略、动作价值和状态价值三类网络分别使用独立的批次编码参数，同一编码器在有效车辆之间共享局部映射，形成包含当前车辆、全部车辆及决策进度的表示。三类网络分别通过相应的256维表示及后续网络完成动作或价值输出；两个动作价值网络保留各自的目标网络。置信评价器的输出仅用于训练样本加权，不作为新的物理状态。模型预测的后继状态用于构造训练回归目标，执行策略仍只读取决策时实际可得的观测。', 'A05')
    e.phrase(1515, '非终止真实转移', '非终止环境交互转移', 'A06')
    e.phrase(1505, '运营末段仍由目标离线记录和源仿真转移支持训练；模型生成仅以运营时域前75%的已观测状态为起点，限制尚未充分处理的服务终点预测误差。', '模型样本的生成范围及其参与价值学习的方式见第6.2.3节。', 'C10')
    e.full(1521, '公交模型选择使用第6.2.2节定义的六类综合预测损失。选择依据是给定数据和训练条件下的预测误差，不预先规定残差模型更优，也不依据最终测试乘客成本选择模型。预测误差排序与控制效果的关系由强制模型类别对照及服务评价检验。', 'C10')
    e.full(1519, ['目标数据的三类划分及用途见第6.2.2节。以模型选择验证集上候选',e.math(1519,0),'的综合误差为',e.math(1519,1),'，选择规则为'], 'C10')
    # Put the two distinct confidence backups next to one another, not delete either.
    e.full(1525, [e.math(1525,0),'依据规定的拟合误差统计更新，用于协调不同训练阶段的误差数值。局部评分越接近1，表示在该误差定义下的预测更可靠。置信评价器由该评分学习对数可靠性，区别于以乘客奖励训练的动作价值网络。原递推目标及样本权重为'], 'C11')
    e.moved[1525] = [1593,1594,1592]
    e.full(1592, '原递推将当前对数评分与后继对数评价相加。为缓解权重随递推范围增加而衰减，主要公交配置将当前项乘以1减去递推系数，采用归一化形式', 'C11')
    e.phrase(1528, '目标和源真实转移', '目标和源环境交互转移', 'A06')
    e.phrase(1532, '组合真实经验与模型经验', '组合环境交互样本与模型生成样本', 'A06')
    e.phrase(1532, '部分补充试验改变模型系数或直接修正源目标，不应写成所有MC-WM配置同时启用的模块。', '模型系数调整与源转移价值目标修正分别在补充配置中考察，其设置和结果在相应实验小节报告。', 'B03')
    e.full(1586, '公交训练按数据划分、源模型拟合与冻结、目标候选拟合与选择、置信评价更新、一步样本生成及策略价值更新的顺序执行。各阶段采用第6.2.2至6.2.3节的模型和损失定义，以下说明具体更新安排。', 'C12')
    e.rename_heading(1585,'6.3.3 策略训练','6.3.3 公交事件模型与策略训练','C12')
    e.delete(1601, 'C11/归一化原理已并列说明')
    e.phrase(1600, '模型选择决定训练样本的预测来源，置信权重决定每条模型样本对价值拟合的影响，模型损失系数则调节整个模型项的强度。', '完成一步样本生成后，依次更新动作价值、策略及相应目标网络。模型类别、逐样本置信与模型项系数分别决定预测来源、样本贡献及整体强度。', 'C12')

    # Replace placeholders with the configurations and repetition counts already reported.
    e.cell(1710,3,1,'复用50000次IQL初始化，新增20000次策略与价值更新；6种源训练配置各2次初始化，另有2次纯离线继续训练','A03')
    e.cell(1710,3,2,'正常标定、联合50%与75%偏差下，各采集约25000个源事件；纯离线组不采集源事件','A03')
    e.cell(1710,3,3,'按第6.4.4节分别比较无模型、原置信、归一化、固定0.5权重及源目标修正；模型组采用分阶段拟合，模型项系数0.25','A03')
    e.cell(1710,5,3,'源交互量与执行规则一致；网络更新及模型拟合计算量不匹配。完整控制器结果见第6.4.4节','A03')
    e.cell(1712,2,2,'18个主比较策略产生180回合；第6.4.3节三种损失对照的27个策略另评价270回合','A03')
    e.cell(1712,3,1,'3种源条件×6种源训练配置×2次初始化，加2次纯离线继续训练，共38个策略','A03')
    e.cell(1712,3,2,'38个策略×2档需求×5日＝380回合；Daganzo与无控制规则另评价20回合。配置见第6.4.4节，共享结果不按源条件重复计数','A03')
    e.full(1713, '表6-9分别统计独立训练与固定策略评价回合：同一策略在不同需求或评价日的运行不增加训练重复数，共享基线只计一次。各对照的数据用途及补充结果见附录F.4；完整控制器比较的225回合由3种控制器、5次初始化、3档需求和5个评价日组成。', 'C15')
    e.delete(1714,'C15/合并重复计数说明')
    e.full(1718, '六组采用相同的驻站幅度变换与完成保障规则，并在幅度变换前，将策略给出的附加驻站建议时长乘以0.25。网络结构相同；纯离线组的策略与价值学习率为0.0003，混合组为0.00005，因此该组训练方式比较同时包含学习率差异。后续更新次数相同的对照另行报告。', 'A04')

    # Table terminology describes inputs rather than repeating warnings in every row.
    e.full(654, 'RE-SAC比较包括原网络结构下的已保存训练记录，以及两次统一网络结构的扩展训练（初始化编号32、40）。原结构中SAC的隐藏层宽度为32，RE-SAC及惩罚消融为64；统一结构采用宽度64的策略网络。表3-1分别报告两类结构下的训练回报与评价回报，其中原记录的策略存档编号为8。训练回报取最后100回合平均值；评价回报先平均每个最终确定性策略的10个运行回合，再汇总训练重复。', 'D10')
    for col,value in enumerate(['方法','原网络结构训练回报','统一网络结构训练回报','原网络结构评价回报','统一网络结构评价回报']):
        e.cell(657,0,col,value,'D10')
    e.phrase(656,'RE-SAC既有策略与扩展实验的公交回报','RE-SAC原网络结构与统一网络结构的公交回报','D10')
    e.phrase(658,'既有运行结果','原网络结构评价结果','D10')
    e.phrase(658,'扩展运行列','统一网络结构评价列','D10')
    e.phrase(661,'既有编号8的结果','原网络结构下已保存策略的结果','D10')

    for i,p in enumerate(e.body):
        if i in e.deleted or i<243 or audit.style(p).startswith('TOC') or audit.style(p)=='bibliography': continue
        replacements=[('模型型强化学习','基于模型的强化学习'),('违例代价日程','违例惩罚系数的分阶段调整规则')]
        if 1110<=i<=1241:
            replacements.extend([('正式公交控制器','源权重为1的H2O+控制器'),('正式控制器','源权重为1的H2O+控制器')])
        if i>=1460:
            replacements.extend([('第5章正式H2O+','第5章源权重为1的H2O+控制器'),('归一化置信度配置','归一化置信度MC-WM')])
        if 783<=i<=803: replacements.append(('正常需求','基准需求'))
        replacements.extend([('物理时距指标','车头时距指标'),('物理时距和服务指标','车头时距和乘客服务指标'),('物理时距偏差','车头时距偏差')])
        if '正驻站' in text(p):
            replacements.extend([('正驻站分布','非零驻站时长分布'),('正驻站时长的条件分布','非零驻站时长的条件分布'),('正驻站','驻站时长大于0的连续分支')])
        for old,new in replacements:
            while old in ''.join(n.text or '' for n in p.iter(W+'t')):
                e.phrase(i,old,new,'D01–D08/术语统一')
    e.phrase(1511,'根据真实转移和模型转移','根据环境交互转移和模型生成转移','A06')
    # Establish the meaning once without mislabelling simulator data as road data.
    e.phrase(1509,'MC-WM在已有混合策略学习中增加环境预测及其可靠性评价，但不改变策略、动作价值与状态价值的基本职责。', 'MC-WM在已有混合策略学习中增加环境预测及其可靠性评价。环境交互样本包括固定SUMO离线记录和快速源仿真新采集的转移；模型生成样本是世界模型预测的一步转移。策略、动作价值与状态价值的基本职责保持不变。','A06')

    e.phrase(1851,'为明确实际应用所需的后续工作，本章从真实运营记录利用、观测与执行误差、组件作用分离及现场受控验证等方面提出研究方向。','最后，围绕大语言模型辅助的运行条件识别、奖励设计、世界模型修正与分层控制，讨论向真实公交运营扩展的研究方向。','E06')
    e.phrase(1857,'该配置不另加动作值拟合损失，其完整动作仍由策略优化确定。','完整动作由策略优化确定。','E07')
    e.full(1856, '正则项对照中，保留价值分歧惩罚和策略保守评价，相对于同时去除两项，在正常运行、客流激增和快速切换下分别降低14.20%、4.02%和11.29%的人均广义时间成本，配对区间均支持成本下降。单项及交互效应的多数差值区间仍包含零。增加价值网络数量未持续改善服务结果，延长训练还可能出现后期退化，说明集成规模与训练时长不能替代可靠的价值评价和训练稳定性设计。', 'E07')
    e.phrase(1873,'最终配置同时在价值目标中使用停止梯度后的批平均导数误差，但不另加动作值拟合或行为克隆损失。','该方法从近期反馈和状态响应两个方面改善驻站学习。','E07')
    e.full(1879, '交通管制通告、活动安排和调度记录包含结构化车辆观测之外的事件信息。未来可结合决策时已经获得的文本资料与车辆运行记录，利用大语言模型提取事件类型、影响线路、涉及站点及持续时段，并将经核验的结果编码为驻站策略的附加状态。例如，将活动散场与站点客流变化相联系，将道路施工与相关路段运行时间相联系。验证应考察这些信息能否缩短运行变化的识别延迟，同时分析信息过期、描述不完整及文本判断与实际观测不一致时的影响。', 'E07/示例替换')
    e.full(2509, ['设',e.math(2509,0),'为第4.2.3节定义的策略网络，其输出尚未经过最后的动作压缩；',e.math(2509,2),'为初值为0的可学习标量。完整确定性动作及秒数换算为'], 'C14')
    e.full(2512,'符号代价模型的输入、函数库、辨识数据及拟合过程见第4.2.3节。以下只保留构造动作参考和分析导数误差所需的定义。','C14')
    e.phrase(2513,'包含13个点，分别对应每隔5秒取值的0至60秒驻站','为第4.2.3节使用的驻站候选网格','C14')
    # Captions retain number bookmarks; add a preserved separator outside them.
    for i,p in enumerate(e.body):
        if audit.style(p) not in {'figurecaption','tablecaption'}: continue
        value=''.join(n.text or '' for n in p.iter(W+'t'))
        m=re.match(r'^([图表][1-7A-F]-\d+)(?!\d)([^\s\d])',value)
        if m: e.phrase(i,m[0],m[1]+' '+m[2],'题注分隔空格')


def main():
    HERE.mkdir(exist_ok=True)
    if not (HERE/'修改前.docx').exists(): shutil.copy2(BASE,HERE/'修改前.docx')
    e=Editor()
    apply_review(e)
    e.renumber()
    e.save()


if __name__=='__main__': main()
