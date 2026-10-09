"""Apply the reread corrections to the current Word package, not older sources."""
from copy import deepcopy
import json
from pathlib import Path
import re
from zipfile import ZipFile, ZIP_DEFLATED

from lxml import etree as E
from extract_word import NS, content

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
BASE = ROOT / 'output/排版完成.docx'
BACKUP = HERE / '排版完成_校审前.docx'
NS['r'] = 'http://schemas.openxmlformats.org/officeDocument/2006/relationships'
REL = 'http://schemas.openxmlformats.org/package/2006/relationships'


def q(tag):
    prefix, local = tag.split(':')
    return '{' + NS[prefix] + '}' + local


def run(text, superscript=False):
    n = E.Element(q('w:r'))
    if superscript:
        E.SubElement(E.SubElement(n, q('w:rPr')), q('w:vertAlign')).set(q('w:val'), 'superscript')
    t = E.SubElement(n, q('w:t'))
    t.text = text
    t.set('{http://www.w3.org/XML/1998/namespace}space', 'preserve')
    return n


def mr(text):
    n = E.Element(q('m:r'))
    fonts=E.SubElement(E.SubElement(n,q('w:rPr')),q('w:rFonts'))
    fonts.set(q('w:ascii'),'Cambria Math');fonts.set(q('w:hAnsi'),'Cambria Math')
    E.SubElement(n, q('m:t')).text = text
    return n


def sub(base, label):
    n = E.Element(q('m:sSub'))
    E.SubElement(n, q('m:e')).append(mr(base))
    r=mr(label)
    if len(label)>1:
        pr=E.Element(q('m:rPr'));E.SubElement(pr,q('m:sty')).set(q('m:val'),'p')
        r.insert(0,pr)
    E.SubElement(n, q('m:sub')).append(r)
    return n


class Editor:
    def __init__(self):
        with ZipFile(BASE) as z:
            self.parts = {n: z.read(n) for n in z.namelist()}
        with ZipFile(BACKUP) as z:
            assert z.read('word/document.xml') == self.parts['word/document.xml'], 'Word changed since backup; reread it before applying.'
        self.doc = E.fromstring(self.parts['word/document.xml'])
        self.body = self.doc.find('w:body', NS)
        self.blocks = list(self.body)
        assert len(self.blocks) == 3161
        self.rels = E.fromstring(self.parts['word/_rels/document.xml.rels'])
        self.bookmark_id = max(map(int, self.doc.xpath('.//w:bookmarkStart/@w:id', namespaces=NS))) + 1
        self.changes = []

    def replace(self, i, old, new):
        node = self.blocks[i]
        ts = node.xpath('.//w:t', namespaces=NS)
        text = ''.join(t.text or '' for t in ts)
        start = text.find(old)
        assert start >= 0, (i, old, content(node))
        end = start + len(old)
        offset = 0
        for t in ts:
            value = t.text or ''
            lo, hi = offset, offset + len(value)
            if lo < end and hi > start:
                left, right = max(start - lo, 0), min(end - lo, len(value))
                t.text = value[:left] + (new if lo <= start < hi else '') + value[right:]
            offset = hi
        self.changes.append({'block': i, 'old': old, 'new': new})

    def plain(self, i, text):
        p = self.blocks[i]
        assert not p.xpath('.//w:instrText', namespaces=NS), i
        links = {content(n): deepcopy(n) for n in p.findall('w:hyperlink',NS)}
        assert all(label in text for label in links), (i,list(links))
        before = content(p)
        for n in list(p):
            if n.tag not in (q('w:pPr'), q('w:bookmarkStart'), q('w:bookmarkEnd')):
                p.remove(n)
        pattern = r'(\{\{[^{}]+\}\}' + ''.join('|'+re.escape(label) for label in links) + ')'
        for part in re.split(pattern, text):
            if part in links:
                p.append(deepcopy(links[part]))
                continue
            if part.startswith('{{'):
                math = E.SubElement(p, q('m:oMath'))
                symbol = part[2:-2]
                if '_' in symbol:
                    base, label = symbol.split('_', 1)
                    math.append(sub(base, label))
                else:
                    math.append(mr(symbol))
            elif part:
                p.append(run(part))
        self.changes.append({'block': i, 'old': before, 'new': text})

    def math_tokens(self, indices, old, new):
        count = 0
        for i in indices:
            for t in self.blocks[i].xpath('.//m:t', namespaces=NS):
                if t.text == old:
                    if '_' in new:
                        base, label = new.split('_', 1)
                        r=t.getparent()
                        r.getparent().replace(r, sub(base,label))
                    else:
                        t.text = new
                    count += 1
                elif old == 't' and re.search(r'(?<![a-zA-Z])t(?![a-zA-Z])',t.text or ''):
                    t.text=re.sub(r'(?<![a-zA-Z])t(?![a-zA-Z])',new,t.text)
                    count += 1
        self.changes.append({'blocks': list(indices), 'math_token': old, 'new': new, 'count': count})

    def para(self, text, template=438):
        p = E.Element(q('w:p'))
        p.append(deepcopy(self.blocks[template].find('w:pPr', NS)))
        p.append(run(text))
        return p

    def citation(self, number, key):
        link = E.Element(q('w:hyperlink'))
        link.set(q('w:anchor'), 'ref-' + key)
        link.append(run(str(number), True))
        return [run('[', True), link, run(']', True)]

    def bibliography(self, i, text, url):
        p = self.blocks[i]
        for n in list(p):
            if n.tag not in (q('w:pPr'), q('w:bookmarkStart'), q('w:bookmarkEnd')):
                p.remove(n)
        p.append(run(text + ' '))
        rid = 'rId' + str(max(int(n.get('Id')[3:]) for n in self.rels) + 1)
        rel = E.SubElement(self.rels, '{' + REL + '}Relationship')
        rel.set('Id', rid)
        rel.set('Type', NS['r'] + '/hyperlink')
        rel.set('Target', url)
        rel.set('TargetMode', 'External')
        link = E.SubElement(p, q('w:hyperlink'))
        link.set(q('r:id'), rid)
        link.append(run(url))
        p.append(run('.'))

    def add_reference(self, number, key, text, url):
        p = self.para('', 2330)
        for n in list(p)[1:]:
            p.remove(n)
        start = E.SubElement(p, q('w:bookmarkStart'))
        start.set(q('w:id'), str(self.bookmark_id))
        start.set(q('w:name'), 'ref-' + key)
        p.append(run('[' + str(number) + ']\t' + text + ' '))
        rid = 'rId' + str(max(int(n.get('Id')[3:]) for n in self.rels) + 1)
        rel = E.SubElement(self.rels, '{' + REL + '}Relationship')
        rel.set('Id', rid); rel.set('Type', NS['r'] + '/hyperlink')
        rel.set('Target', url); rel.set('TargetMode', 'External')
        link=E.SubElement(p,q('w:hyperlink'));link.set(q('r:id'),rid)
        link.append(run(url));p.append(run('.'))
        end = E.SubElement(p, q('w:bookmarkEnd'))
        end.set(q('w:id'), str(self.bookmark_id))
        self.bookmark_id += 1
        self.blocks[2331].addprevious(p)
        return p


def citation_groups(p):
    groups, pending = [], []
    for n in p:
        superscript = n.xpath('.//w:vertAlign[@w:val="superscript"]', namespaces=NS)
        if superscript:
            pending.append(n)
        else:
            if pending:
                groups.append(pending)
                pending = []
    if pending:
        groups.append(pending)
    return [(g, ''.join(content(n) for n in g)) for g in groups
            if re.fullmatch(r'\[\d+(?:[,，、–—-]\d+)*\]', ''.join(content(n) for n in g))]


def renumber_references(ed):
    refs, old_keys = {}, {}
    for p in ed.body:
        if p.xpath('./w:pPr/w:pStyle[@w:val="bibliography"]', namespaces=NS):
            number = int(re.match(r'\[(\d+)\]', content(p))[1])
            key = p.xpath('.//w:bookmarkStart/@w:name', namespaces=NS)[0][4:]
            refs[key], old_keys[number] = p, key
    all_groups = []
    first = []
    for p in ed.body.xpath('.//w:p', namespaces=NS):
        if p.xpath('./w:pPr/w:pStyle[@w:val="bibliography"]', namespaces=NS):
            continue
        for nodes, text in citation_groups(p):
            nums = []
            for item in re.split('[,，、]', text[1:-1]):
                values = re.split('[–—-]', item)
                nums.extend(range(int(values[0]), int(values[-1])+1))
            keys = [old_keys[n] for n in nums]
            for key in keys:
                if key not in first:
                    first.append(key)
            all_groups.append((p, nodes, keys))
    assert set(first) == set(refs), set(refs)-set(first)
    numbers = {key: i+1 for i, key in enumerate(first)}
    for p, nodes, keys in all_groups:
        new = [run('[', True)]
        for j, key in enumerate(sorted(set(keys), key=numbers.get)):
            if j:
                new.append(run(',', True))
            link = E.Element(q('w:hyperlink'))
            link.set(q('w:anchor'), 'ref-' + key)
            link.append(run(str(numbers[key]), True))
            new.append(link)
        new.append(run(']', True))
        source_pr = nodes[0].find('w:rPr',NS)
        if source_pr is not None:
            for n in new:
                r = n if n.tag==q('w:r') else n.find('w:r',NS)
                r.remove(r.find('w:rPr',NS))
                r.insert(0,deepcopy(source_pr))
        pos = p.index(nodes[0])
        for n in nodes:
            p.remove(n)
        for n in reversed(new):
            p.insert(pos, n)
    insert = ed.body.index(next(iter(refs.values())))
    for p in refs.values():
        ed.body.remove(p)
    for key in first:
        p = refs[key]
        t = p.xpath('.//w:t', namespaces=NS)[0]
        t.text = re.sub(r'^\[\d+\]', '['+str(numbers[key])+']', t.text)
        ed.body.insert(insert, p)
        insert += 1
    (HERE/'reference_number_map.json').write_text(json.dumps(
        {n: {'key': key, 'new': numbers[key]} for n, key in old_keys.items()}, ensure_ascii=False, indent=2))


def main():
    ed = Editor()
    ed.replace(383, '本文将控制动作限定为车辆完成必要上下客后的附加停站，保持既定线路、班次及供给计划，以比较相同控制权限下的策略效果。',
        '本文以车辆完成必要上下客后的附加驻站为共同基础，并在奖励与动作的交叉比较中加入有界速度调节。各组保持既定线路、班次及供给计划；学习方法之间在相同动作权限下比较，传统驻站规则则作为无需训练的运行参照。')
    ed.replace(384, '后者主要调节车辆附加停站，并研究有限目标离线数据的使用。', '后者以车辆附加驻站及其与有界速度调节的协同为对象，并研究有限目标离线数据的使用。')
    ed.replace(451, '根据当前运行信息选择有界附加停站时长，调节车头时距并改善乘客服务。', '根据当前运行信息选择有界附加驻站时长，并在联合动作实验中协同调节离站后的运行速度，以改善车头时距或乘客服务。')
    ed.replace(653, '读取状态并写入附加停站时长。', '读取状态并写入附加驻站时长，联合动作配置还写入离站后下一路段的速度倍率。')
    ed.replace(653, '学习器决定可控的附加停站', '学习器决定可控的驻站与速度调节')
    ed.replace(654, '具体场景不混作同一实验。', '联合动作配置在离站后施加有界速度调节，保持两平台的动作语义一致；具体场景不混作同一实验。')
    ed.replace(1057, '相对300秒的计划', '相对360秒的计划')
    ed.replace(1203, '第D.5.3节', '附录D.5.3节')
    ed.replace(1867, '第E.3.3节', '附录E.3.3节')
    ed.replace(1867, '附录E第E.4节', '附录E.4节')
    ed.plain(1745, '上述固定步长、固定折扣的无限时域形式用于受控连续任务的机制研究。公交任务沿用第2章的事件决策，在规定运营时域内无折扣累计乘客成本；事件持续时间通过人数时间积分进入即时成本，而非再次构造时间折扣。其SUMO离线数据固定，新增训练交互来自简易仿真，与第5章H2O+的数据权限一致。两类任务都利用目标证据修正源预测，受控任务可以追加目标证据，公交任务则在既有目标数据覆盖内利用新的源交互。')
    ed.plain(1371, '结构化输出的优势具有任务依赖性。图4-4至图4-6显示，算术输出与普通平滑输出的相对排序随需求强度和变化日程改变；表4-10进一步区分完整训练稳定化的收益与输出结构的附加作用。公交20倍需求下的个别深尾部失败说明，典型运行的改善与最差情形的保障仍是不同问题。')
    ed.replace(2115, '在联合75%源偏差下，双动作MC-WM相对共同梯度约束的H2O+降低3.61%的广义时间和0.80%的平均时距惩罚；相对Daganzo式规则，两种相应训练目标的降幅分别为4.13%和3.18%。',
        '在联合75%源偏差下，以时距训练的双动作MC-WM相对同目标、同梯度约束的H2O+降低3.61%的广义时间和0.80%的平均时距惩罚，见表6-17；相对Daganzo式规则，乘客目标训练的广义时间降幅为4.13%，时距目标训练的平均惩罚降幅为3.18%，分别见表6-16和表6-17。')
    ed.replace(439, '本文提出的MC-WM', '本文提出的基于模型修正的世界模型方法（Model-Corrected World Model，MC-WM）')
    ed.replace(1411, '累积分布函数（CDF）', '累积分布函数（Cumulative Distribution Function，CDF）')
    ed.replace(1980, 'GPU预热后', '图形处理器（Graphics Processing Unit，GPU）预热后')
    ed.replace(879, 'Double DQN分开', '双重深度Q网络（Double Deep Q-Network，Double DQN）分开')
    ed.replace(879, 'Dueling网络分解', '竞争式深度Q网络（Dueling Deep Q-Network，Dueling DQN）分解')
    ed.replace(404, 'Bootstrapped DQN利用', '自助采样深度Q网络（Bootstrapped Deep Q-Network，Bootstrapped DQN）利用')
    # First definitions in the introduction remain; subsequent prose uses the abbreviations.
    for i, p in enumerate(ed.blocks):
        if not 486 <= i < 2178:
            continue
        for full, short in [
            ('马尔可夫决策过程（Markov Decision Process，MDP）', 'MDP'),
            ('部分可观测马尔可夫决策过程（Partially Observable Markov Decision Process，POMDP）', 'POMDP'),
            ('软演员评论家（Soft Actor-Critic，SAC）', 'SAC')]:
            if full in ''.join(p.xpath('.//w:t/text()', namespaces=NS)):
                ed.replace(i, full, short)

    # Discrete decisions use k; physical time t in traffic schedules is unchanged.
    discrete = [885, *range(1146,1155), *range(1161,1167), *range(1317,1321),
                1743,1744, *range(2434,2631),3050]
    ed.math_tokens(discrete, 't', 'k')
    for i in range(2434,2631):
        text = ''.join(ed.blocks[i].xpath('.//w:t/text()', namespaces=NS))
        for old, new in [('第t步','第k步'),('t时刻','第k步')]:
            if old in text:
                ed.replace(i, old, new)
    # History and headway are distinct: H_k versus h.
    for i in [354,608,1161,1166]:
        for s in ed.blocks[i].xpath('.//m:sSub', namespaces=NS):
            if content(s.find('m:e',NS)) == 'h' and content(s.find('m:sub',NS)) in ('t','k'):
                s.find('.//m:t',NS).text = 'H'
                for t in s.find('m:sub',NS).xpath('.//m:t',namespaces=NS):
                    if t.text == 't': t.text = 'k'
    for i in [2524,2537,2538]:
        for s in list(ed.blocks[i].xpath('.//m:sSub', namespaces=NS)):
            history_base = s.getparent().tag == q('m:e') and s.getparent().getparent().tag == q('m:sSub')
            if content(s) == 'N_{hor}' and (i==2524 or history_base):
                s.getparent().replace(s, mr('H'))
    # A physical horizon is not a Bellman operator.
    for i in [531,532, *range(569,580),1258,1259]:
        for t in list(ed.blocks[i].xpath('.//m:t',namespaces=NS)):
            if t.text == 'T':
                r = t.getparent()
                r.getparent().replace(r, sub('T','hor'))
    for i in [569,570,573,574,575,1430]:
        for t in ed.blocks[i].xpath('.//m:t',namespaces=NS):
            if t.text == 'u': t.text = 'τ'
            elif t.text == 'du': t.text = 'dτ'
    ed.replace(569, '设时刻 ', '设积分时刻 ')
    ed.replace(573, '在时刻 ', '在积分时刻 ')

    # R is a conditional mean; r_k is the realized reward.
    math = ed.blocks[1147].find('.//m:oMath',NS)
    r_node = next(n for n in math if n.tag == q('m:sSub') and content(n.find('m:e',NS)) == 'r')
    pos = math.index(r_node)
    state = next(n for n in math if n.tag == q('m:sSub') and content(n) == 's_{k}')
    action = next(n for n in math if n.tag == q('m:sSub') and content(n) == 'a_{k}')
    z = sub('z','k')
    inserted = [mr('E['), deepcopy(r_node), mr('∣'), deepcopy(state), mr(','), deepcopy(action), mr(','), z, mr(']')]
    math.remove(r_node)
    for n in reversed(inserted): math.insert(pos,n)
    ed.math_tokens([1853,3047], 'r', 'R')
    ed.math_tokens([1816], 't', 'n_int')
    for i in [1814,1816]:
        for s in list(ed.blocks[i].xpath('.//m:sSub',namespaces=NS)):
            if content(s.find('m:e',NS)) == 'ι':
                s.find('.//m:t',NS).text = 'I'
                label = s.find('m:sub',NS)
                s.tag = q('m:sSubSup')
                sup = E.SubElement(s,q('m:sup')); sup.append(mr('viol'))
                pr = s.find('m:sSubPr',NS)
                if pr is not None: pr.tag = q('m:sSubSupPr')
    ed.plain(1815, '在总在线步数{{N_on}}内，违例代价系数前半程保持为零，后半程线性上升到10；以{{n_int}}表示当前累计在线交互次数，违例指示量取1表示发生违例、取0表示未发生：')
    for i in [1822]:
        for s in ed.blocks[i].xpath('.//m:sSub',namespaces=NS):
            if content(s.find('m:e',NS)) == 'V':
                s.find('.//m:t',NS).text = 'N'
                s.find('m:sub',NS).insert(0,mr('viol,'))
    ed.math_tokens([1837], 'u', 'e_Hub')
    ed.plain(1836, '其中{{e_Hub}}表示评论家预测与贝尔曼目标之差，Huber损失采用阈值1，即')
    for i in [2722,2913,2914,2915]:
        ed.math_tokens([i], 'k', 'c_fb' if i==2722 else 'c_symfb')
        text = ''.join(ed.blocks[i].xpath('.//w:t/text()',namespaces=NS))
        for old,new in [('若 k','若 c_fb'),('当 k','当 c_symfb')]:
            if old in text: ed.replace(i,old,new)

    # The fitted symbolic function f_c is not the unknown shared component f_sh.
    for i, chosen in [(2949,0),(2951,1),(2953,2)]:
        fs = [s for s in ed.blocks[i].xpath('.//m:sSub',namespaces=NS)
              if content(s) == 'f_{c}']
        fs[chosen].find('m:sub/m:r/m:t',NS).text = 'sh'
    for i in [2951,2953]:
        for s in ed.blocks[i].xpath('.//m:sSub',namespaces=NS):
            if content(s) == 'ε_{c}': s.find('m:sub/m:r/m:t',NS).text = 'sh'
    # Preserve editable math in the definition paragraph.
    ed.plain(2950, '其中{{f_sh}}是真实映射中在各环境共享的部分，{{f_s,e}}为环境相关部分；前文的{{f_c}}仍表示由数据拟合的符号模型。以{{ε_sh}}表示符号模型相对共享真实部分的导数误差上界，假设')
    ed.replace(2954, '这一分解将共享部分的估计误差与环境特有扰动分开。', '这一分解将符号模型对共享部分的估计误差与环境特有扰动分开。')
    symbol_rows=[
        ('r_k,R_ι', '一次实际奖励与环境条件下的期望单步奖励；二者通过条件期望联系'),
        ('τ,n_int,N_on', '乘客时间积分中的物理时刻、累计在线交互次数及总在线交互预算'),
        ('c_fb,c_symfb', '信念价值反馈与符号反馈反例中的距离因子，不表示决策序号'),
        ('f_sh,f_c,ε_sh', '共享真实映射、拟合符号模型及二者的导数误差上界')]
    table=ed.blocks[354]
    source=table.findall('w:tr',NS)[2]
    for symbols,definition in symbol_rows:
        row=deepcopy(source)
        for cell,text in zip(row.findall('w:tc',NS),[symbols,definition]):
            p=cell.find('w:p',NS)
            for n in list(p):
                if n.tag != q('w:pPr'): p.remove(n)
            if cell is row.findall('w:tc',NS)[0]:
                for j,symbol in enumerate(symbols.split(',')):
                    if j: p.append(run('，'))
                    math=E.SubElement(p,q('m:oMath'))
                    if '_' in symbol:
                        base,label=symbol.split('_',1);math.append(sub(base,label))
                    else: math.append(mr(symbol))
            else:
                p.append(run(text))
            for r in p.xpath('.//w:r | .//m:r',namespaces=NS):
                pr=r.find('w:rPr',NS)
                if pr is None: pr=E.SubElement(r,q('w:rPr'))
                if r.tag==q('w:r'):
                    fonts=E.SubElement(pr,q('w:rFonts'))
                    fonts.set(q('w:ascii'),'Times New Roman');fonts.set(q('w:hAnsi'),'Times New Roman')
                    fonts.set(q('w:eastAsia'),'宋体')
                E.SubElement(pr,q('w:sz')).set(q('w:val'),'21')
        table.insert(len(table)-1,row)
    ed.replace(2416, '先对耦合取下确界，', '由前式的绝对值界，目标分布下的价值期望不小于名义期望减去距离成本与价值敏感度的乘积。先对耦合取下确界得到Wasserstein距离，再使用给定的距离半径界，')
    ed.replace(1469, 'Daganzo启发式采用前向比例反馈', '前向Daganzo式启发规则采用比例反馈')
    ed.replace(1473, 'Daganzo启发式体现车头时距控制原则，所比较对象是上述比例规则本身。',
        '该前向规则体现Daganzo的车头时距反馈思想，所比较对象是式(5-12)的具体比例规则。第6.4.3.7节及联合动作实验采用前后车间隔差构造双向Daganzo式规则，二者使用的信息不同，不作为同一个固定控制器跨章节合并。')

    # ArXiv records use one consistent electronic-document type.
    for p in ed.body:
        if p.xpath('./w:pPr/w:pStyle[@w:val="bibliography"]',namespaces=NS):
            for t in p.xpath('.//w:t',namespaces=NS):
                if t.text: t.text=t.text.replace('&amp;', '&').replace('[A/OL]', '[EB/OL]')
    ed.bibliography(2227, '[49]\tWEI H, XU N, ZHANG H, et al. CoLight: Learning Network-level Cooperation for Traffic Signal Control[EB/OL]. arXiv, 2019[2026-10-03].', 'https://arxiv.org/abs/1905.05717')
    ed.bibliography(2263, '[85]\tPRUDENCIO R F, MAXIMO M R O A, COLOMBINI E L. A Survey on Offline Reinforcement Learning: Taxonomy, Review, and Open Problems[EB/OL]. arXiv, 2022[2026-10-03].', 'https://arxiv.org/abs/2203.01387')
    ed.add_reference(153,'feng2023finetuning', 'FENG Y, HANSEN N, XIONG Z, et al. Finetuning Offline World Models in the Real World[EB/OL]. arXiv, 2023[2026-10-03].','https://arxiv.org/abs/2310.16029')
    ed.add_reference(154,'li2026uncertaintyworld', 'LI C, KRAUSE A, HUTTER M. Uncertainty-Aware Robotic World Model Makes Offline Model-Based Reinforcement Learning Work on Real Robots[EB/OL]. arXiv, 2026[2026-10-03].','https://arxiv.org/abs/2504.16680v3')
    p=ed.para('近期真实机器人研究进一步区分了世界模型的数据使用条件。Feng等用真实机器人离线经验预训练世界模型，再通过真实环境在线交互微调，并以规划期不确定性约束抑制外推误差。')
    p.extend(ed.citation(153,'feng2023finetuning'))
    p.append(run('Li等在2026年修订的工作中，从既有数据学习带认识不确定性估计的自回归世界模型，并利用不确定性惩罚支持离线策略训练及真实机器人部署。'))
    p.extend(ed.citation(154,'li2026uncertaintyworld'))
    p.append(run('两条路线分别强调追加目标交互与离线模型使用的风险控制。本论文公交实验采用固定目标离线数据与源仿真交互，研究已有目标证据如何修正源训练；这一数据权限与真实在线微调不同，也没有以SUMO评价替代真实车辆部署证据。'))
    ed.blocks[439].addprevious(p)

    # Only normalize the six newly integrated table headers to the adopted template.
    titles = {'表3-10','表4-10','表4-11','表5-13','表6-16','表6-17'}
    for i,p in enumerate(ed.blocks):
        if any(content(p).startswith(title+' ') for title in titles):
            table=next(n for n in ed.blocks[i+1:] if n.tag==q('w:tbl'))
            for b in table.find('w:tr',NS).xpath('.//w:rPr/w:b',namespaces=NS):
                b.getparent().remove(b)
    renumber_references(ed)
    ed.parts['word/document.xml']=E.tostring(ed.doc,xml_declaration=True,encoding='UTF-8',standalone=True)
    ed.parts['word/_rels/document.xml.rels']=E.tostring(ed.rels,xml_declaration=True,encoding='UTF-8',standalone=True)
    stage=HERE/'排版完成_校审中.docx'
    with ZipFile(stage,'w',ZIP_DEFLATED) as z:
        for name,value in ed.parts.items(): z.writestr(name,value)
    (HERE/'changes.json').write_text(json.dumps(ed.changes,ensure_ascii=False,indent=2))
    print(stage)
    print('Changes:', len(ed.changes))


if __name__=='__main__':
    main()
