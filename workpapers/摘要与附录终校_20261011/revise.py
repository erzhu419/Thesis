"""Apply the final abstract replacements and appendix-only corrections."""
from copy import deepcopy
import importlib.util
import json
from pathlib import Path
import re
import shutil
from zipfile import ZipFile

from lxml import etree as E
import pypandoc

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
SOURCE = ROOT / 'output/排版完成_七章整合_文献核验_20261011.docx'
OUT = ROOT / 'output/排版完成_摘要与附录终校_20261011.docx'
NS = {'w': 'http://schemas.openxmlformats.org/wordprocessingml/2006/main',
      'm': 'http://schemas.openxmlformats.org/officeDocument/2006/math'}
W, M = ('{'+NS[k]+'}' for k in ('w', 'm'))
spec = importlib.util.spec_from_file_location('merge', ROOT / 'workpapers/七章重写稿整合_20261010/integrate.py')
helper = importlib.util.module_from_spec(spec)
spec.loader.exec_module(helper)


def visible(node):
    return ''.join(node.xpath('.//w:t/text()|.//m:t/text()', namespaces=NS))


def convert(value):
    data = pypandoc.convert_text(value, to='docx', format='markdown', outputfile=str(HERE / 'fragment.docx'))
    with ZipFile(HERE / 'fragment.docx') as z:
        body = E.fromstring(z.read('word/document.xml')).find('w:body', NS)
    paragraphs = [p for p in body if p.tag == W+'p']
    assert len(paragraphs) == 1, value
    return deepcopy(paragraphs[0])


def paragraph(value, template):
    if value.startswith('证明。') and template.xpath('./w:r[w:rPr/w:b]/w:t[starts-with(.,"证明")]', namespaces=NS):
        value = '**证明。**'+value[len('证明。'):]
    generated = convert(value)
    result = deepcopy(template)
    for item in list(result):
        if item.tag not in (W+'pPr', W+'bookmarkStart', W+'bookmarkEnd'):
            result.remove(item)
    text_runs = template.xpath('./w:r[w:t]', namespaces=NS)
    ordinary_runs = [r for r in text_runs if r.find('w:rPr/w:b',NS) is None]
    representative = max(ordinary_runs or text_runs, key=lambda r:len(visible(r)), default=None)
    properties = representative.find('w:rPr',NS) if representative is not None else None
    if properties is None:
        properties = template.find('w:pPr/w:rPr', NS)
    for child in generated:
        if child.tag == W+'pPr':
            continue
        copied = deepcopy(child)
        if copied.tag == W+'r' and properties is not None:
            old = copied.find('w:rPr', NS)
            emphasis = [deepcopy(p) for p in old if p.tag in (W+'b',W+'i',W+'vertAlign')] if old is not None else []
            if old is not None:
                copied.remove(old)
            copied_properties = deepcopy(properties)
            for prop in emphasis:
                existing = copied_properties.find(prop.tag)
                if existing is not None:
                    copied_properties.remove(existing)
                copied_properties.append(prop)
            copied.insert(0, copied_properties)
        for item in copied.xpath('.//m:r|.//m:ctrlPr', namespaces=NS):
            rp = item.find('w:rPr', NS)
            if rp is None:
                rp = E.SubElement(item, W+'rPr')
            fonts = rp.find('w:rFonts', NS)
            if fonts is None:
                fonts = E.SubElement(rp, W+'rFonts')
            fonts.set(W+'ascii', 'Cambria Math')
            fonts.set(W+'hAnsi', 'Cambria Math')
        result.append(copied)
    return result


def replace_words(node, old, new):
    """Replace prose across text runs without touching equations or fields."""
    texts = node.xpath('.//w:t', namespaces=NS)
    joined = ''.join(t.text or '' for t in texts)
    count = joined.count(old)
    for _ in range(count):
        joined = ''.join(t.text or '' for t in texts)
        start = joined.index(old)
        end = start+len(old)
        position = 0
        inserted = False
        for t in texts:
            value = t.text or ''
            a, b = position, position+len(value)
            position = b
            if b <= start or a >= end:
                continue
            left = value[:max(start-a, 0)]
            right = value[min(end-a, len(value)):]
            t.text = left+(new if not inserted else '')+right
            t.set('{http://www.w3.org/XML/1998/namespace}space', 'preserve')
            inserted = True
    return count


def append_prose(node, value):
    generated = paragraph(value, node)
    for child in generated:
        if child.tag not in (W+'pPr', W+'bookmarkStart', W+'bookmarkEnd'):
            node.append(child)


def link_supplementary_equations(node):
    for text in list(node.xpath('.//w:r/w:t', namespaces=NS)):
        value = text.text or ''
        matches = list(re.finditer(r'\(([DE]-\d+[a-z])\)', value))
        if not matches:
            continue
        run = text.getparent()
        parent = run.getparent()
        position = parent.index(run)
        properties = run.find('w:rPr', NS)
        pieces = []
        start = 0
        for match in matches:
            if match.start() > start:
                pieces.append(helper.make_run(value[start:match.start()], properties))
            name = 'supp_'+match.group(1).replace('-', '_')
            pieces.extend(helper.field_runs('REF '+name+' \\h', match.group(0), properties))
            start = match.end()
        if start < len(value):
            pieces.append(helper.make_run(value[start:], properties))
        parent.remove(run)
        for offset, piece in enumerate(pieces):
            parent.insert(position+offset, piece)


def main():
    backup = HERE / '修改前_文献核验版.docx'
    if not backup.exists():
        shutil.copy2(SOURCE, backup)
    with ZipFile(SOURCE) as source:
        original = source.read('word/document.xml')
        body = E.fromstring(original).find('w:body', NS)
        spans = helper.body_spans(original)
        nodes = [deepcopy(x) for x in body]
        changed, additions, removed = {}, {}, set()
        reasons = {}
        new_bookmark_id = max(int(v) for v in body.xpath('.//w:bookmarkStart/@w:id',namespaces=NS))+1

        def edit(index, value, reason):
            nodes[index] = paragraph(value, body[index])
            changed[index] = nodes[index]
            reasons[index] = reason

        def words(index, old, new, reason):
            assert replace_words(nodes[index], old, new) > 0, (index, old)
            changed[index] = nodes[index]
            reasons[index] = reason

        def add(index, value, reason):
            template = deepcopy(body[2399])
            for item in list(template):
                if item.tag in (W+'bookmarkStart', W+'bookmarkEnd'):
                    template.remove(item)
            additions.setdefault(index, []).append(paragraph(value, template))
            reasons.setdefault(index, reason)

        def equation(index, value, number, template_index):
            nonlocal new_bookmark_id
            equation_table = deepcopy(body[template_index])
            for item in equation_table.xpath('.//w:bookmarkStart|.//w:bookmarkEnd',namespaces=NS):
                item.getparent().remove(item)
            cells = equation_table.findall('w:tr/w:tc', NS)
            if number.startswith('E-'):
                widths = (960,6386,960)
                for column, width in zip(equation_table.findall('w:tblGrid/w:gridCol', NS),widths):
                    column.set(W+'w',str(width))
                for cell, width in zip(cells,widths):
                    cell.find('w:tcPr/w:tcW',NS).set(W+'w',str(width))
            middle = cells[1].find('w:p', NS)
            for item in list(middle):
                if item.tag != W+'pPr':
                    middle.remove(item)
            generated = convert('$$'+value+'$$')
            math = generated.find('m:oMathPara', NS)
            assert math is not None
            for item in math.xpath('.//m:r|.//m:ctrlPr',namespaces=NS):
                rp = item.find('w:rPr', NS)
                if rp is None:
                    rp = E.SubElement(item,W+'rPr')
                fonts=rp.find('w:rFonts',NS)
                if fonts is None:
                    fonts=E.SubElement(rp,W+'rFonts')
                fonts.set(W+'ascii','Cambria Math')
                fonts.set(W+'hAnsi','Cambria Math')
            middle.append(deepcopy(math))
            last = cells[2].find('w:p', NS)
            rp = deepcopy(last.find('w:r/w:rPr', NS))
            for item in list(last):
                if item.tag != W+'pPr':
                    last.remove(item)
            mark = E.SubElement(last, W+'bookmarkStart')
            mark.set(W+'id',str(new_bookmark_id))
            mark.set(W+'name','supp_'+number.replace('-','_'))
            prefix, suffix = re.fullmatch(r'([DE]-\d+)([a-z])',number).groups()
            letter, numeral = prefix.split('-')
            last.append(helper.make_run('('+letter+'-',rp))
            # Repeat the existing integer, so all original SEQ results stay unchanged.
            for r in helper.field_runs('SEQ Eq'+letter+' \\c \\* ARABIC',numeral,rp):
                last.append(r)
            last.append(helper.make_run(suffix+')',rp))
            E.SubElement(last,W+'bookmarkEnd').set(W+'id',str(new_bookmark_id))
            new_bookmark_id += 1
            additions.setdefault(index,[]).append(equation_table)

        abstracts = (ROOT / 'markdown/摘要第2至4项_中英文替换稿_20261011.md').read_text().splitlines()
        cn = [line for line in abstracts if re.match(r'^（[234]）',line)]
        en = [line for line in abstracts if re.match(r'^\([234]\)',line)]
        assert len(cn)==len(en)==3
        for index, value in zip([66,67,68,74,75,76],cn+en):
            result = deepcopy(body[index])
            text_runs = result.xpath('./w:r[w:t]',namespaces=NS)
            ordinary = [r for r in text_runs if r.find('w:rPr/w:highlight',NS) is None]
            representative = max(ordinary or text_runs,key=lambda r:len(visible(r)))
            properties = representative.find('w:rPr',NS)
            for item in list(result):
                if item.tag not in (W+'pPr', W+'bookmarkStart', W+'bookmarkEnd'):
                    result.remove(item)
            for item in result.xpath('./w:pPr/w:numPr', namespaces=NS):
                item.getparent().remove(item)
            result.append(helper.make_run(value, properties))
            nodes[index] = result
            changed[index] = result
            reasons[index] = '摘要（2）至（4）逐字使用指定替换稿'
            assert ''.join(nodes[index].xpath('.//w:t/text()',namespaces=NS)) == value

        # Only these four main-text paragraphs need to follow the appendix corrections.
        words(996,'利用SINDy参考关系对动作及其状态导数施加软约束',
              '利用SINDy构造的动作参考对完整归一化动作的状态导数施加软约束','R07—R09：软符号实现与附录对应')
        append_prose(nodes[996],'确定性动作映射、参考状态、符号参考的拟合方式及两类符号损失的完整定义见附录D.1，式(D-4a)至式(D-4e)。')
        append_prose(nodes[1022],'实际软符号配置在策略损失中使用当前状态的导数误差，并将后继状态上的批平均导数误差停止梯度后加入价值目标；附录D.6只分析相关量固定时的价值算子，不证明策略与价值网络联合训练的收敛。')
        changed[1022]=nodes[1022]
        reasons[1022]='R07：区分策略正则、价值目标与固定算子结论'
        words(1535,'只有其完整映射满足','只有其完整映射在非空合法残差闭集上保持该集合并满足','R11：合法残差闭集上的条件固定点')
        append_prose(nodes[1543],'有限时域公交任务的对应条件分析见附录E.3的式(E-23a)至式(E-23c)，其中事件数上界与运营秒数分别定义。')
        changed[1543]=nodes[1543]
        reasons[1543]='R10：补充有限事件时域误差分析的正文引用'

        # Keep the correct mathematical objects and all existing equation numbers.
        for i in range(2002,2097):
            for old,new in [('最大值备份','最大值贝尔曼更新'),('熵正则化理想备份','熵正则化理想更新'),('固定备份','固定贝尔曼更新'),('鲁棒贝尔曼备份','鲁棒贝尔曼更新'),('阶段独立使用同一尾部水平的备份','阶段独立使用同一尾部水平的更新')]:
                if replace_words(nodes[i],old,new):
                    changed[i]=nodes[i]; reasons[i]='R16：统一贝尔曼更新措辞'
        edit(2056,r'右端趋于零，完备性保证迭代收敛。由收缩映射的连续性，极限满足固定点方程；两个固定点的距离不超过其自身的$\gamma$倍，故固定点唯一。将迭代与该固定点反复比较，即得式(A-15)。证毕。','R16：压缩标准收缩证明')

        for i in range(2098,2274):
            for old,new in [('规范策略','参考模式下的策略'),('规范动作','参考模式下的动作'),('规范信号','参考模式下的执行信号'),('保持动作盒','保持逐维动作上下界'),('因果控制器','基于历史信息的控制器'),('实际因果识别','基于已完成转移的模式识别'),('因果执行','按上述信息时序执行'),('同一因果行为规则','同一基于已有信息的行为规则'),('因果分支','待评价控制器分支'),('因果轨迹','待评价控制器轨迹'),('因果后继状态','待评价控制器的后继状态'),('因果状态分布','待评价控制器的状态分布'),('因果转移','待评价控制器转移'),('因果信号','待评价控制器信号')]:
                if replace_words(nodes[i],old,new):
                    changed[i]=nodes[i]; reasons[i]='R14：明确参考模式、历史信息与动作边界'
        words(2100,'因果信息与可辨识性','已有信息与模式可辨识性','R14：明确非前视的信息条件')
        add(2159,'本附录中的“因果”仅指动作使用当前及历史信息，而不使用未来观测；它不表示已经识别环境的真实因果结构。下文据此称为基于历史信息的控制器。','R14：区分信息时序与因果结构')
        edit(2117,'证明。截断的值域为$[-1,1]$，且在该区间内保持输入不变，两个方向均由定义直接得到。证毕。','R16：缩短截断基本性质')
        edit(2152,'证明。零步结论成立；若当前状态相同，相同名义信号和共同外生量给出相同执行量、奖励与后继状态。逐步归纳，末状态与累计奖励均相同。证毕。','R16：压缩路径归纳的基础步骤')
        edit(2162,'证明。相同记录前缀依次产生相同置信，再由相同选择映射得到相同动作。证毕。','R16：压缩确定性映射基础性质')
        add(2206,r'为将含记忆的控制规则写成当前状态的函数，自本节起，$x$表示增广状态：它包含物理运行状态、已完成转移历史$H_k$及置信$\rho_k$。历史与置信按$H_{k+1}=H_k\mathbin{\frown}\xi_k$、$\rho_{k+1}=U(\rho_k,\xi_k)$更新，其中$\frown$表示在历史末尾追加一条记录；物理转移、奖励和这些更新共同定义增广状态转移。若真实执行模式不能由时间和物理状态确定，也将其作为分析用隐变量纳入增广状态，但控制器只读取其中可观测的部分。第B.3节的轨迹等价针对物理状态，本节的增广则用于表示控制器记忆。','R14：在路径分析前定义增广状态及其更新')
        words(2207,'参照续接步数','执行参照策略的后续步数','R14：明确后续价值对象')
        words(2219,'参照续接价值是在','执行参照策略后的后续价值是在','R14：明确反事实后续价值')

        for i in range(2275,2385):
            for old,new in [('下一次备份比较','下一次贝尔曼更新比较'),('共同续接','共同后继项'),('有效下置信界系数','保守评价系数')]:
                if replace_words(nodes[i],old,new):
                    changed[i]=nodes[i]; reasons[i]='R15—R16：明确代数评价对象'
        edit(2314,'证明。非负分子除以正归一化常数仍非负，且','R16：压缩概率归一化证明')
        edit(2316,'证毕。证据分数本身无需预先归一化。','R16：压缩概率归一化说明')
        edit(2323,'证明。有限非负和保持非负；两种边缘的总质量分别为','R16：合并边缘分布基础证明')
        edit(2325,'以及','R16：合并边缘分布基础证明')
        words(2363,'识别速度','持续证据下的后验阈值','R15：不是无条件检测时延定理')
        edit(2380,r'在初始概率比$\Omega_{\mathrm{prior}}$与单步证据强度$L_{\mathrm{evid}}>1$固定、$\delta_{\mathrm{post}}$趋于0时，上述充分步数的阶为$O(\log(1/\delta_{\mathrm{post}}))$。这是给定持续同向证据条件下的后验阈值分析，不是带噪声、重置和模式切换的检测器时延定理；实际时间还取决于观测间隔。','R15：准确给出极限方向与条件')

        edit(2387,'本附录分别分析完整策略的动作误差、结构化函数及其复合的正则性，以及固定惩罚价值算子的性质。第D.1节先给出最终公交软符号配置的实现对应；第D.2至D.5节及第D.8至D.10节给出条件误差分析；第D.6至D.7节单独讨论价值目标与固定算子的关系。早期NAU结构的性质不自动成为最终无NAU策略的性质。','R07—R09：区分最终实现与早期结构')
        add(2400,r'**最终公交软符号配置及其分析坐标。** 第4.4.5节的主要配置采用15维归一化状态$s$和一个附加驻站动作。以下坐标从1编号；$s_6,s_7,s_9$分别为前向时距、后向时距和计划时距的归一化值，其时距尺度相同。参考状态投影$\Pi$只将$s_6,s_7$替换为$s_9$，其余坐标保持当前值；这是一种网络输入投影，不是重新模拟出的物理平衡状态。设$m_\theta(s)$为两层、每层64个单元的tanh网络输出的未压缩均值，$b_\theta\in\mathbb R$为初始值为0的可学习标量。完整确定性动作及秒数映射为','R08—R09：补齐最终动作映射与坐标')
        equation(2400,r'\begin{aligned}q_\theta(s)&=b_\theta+m_\theta(s)-m_\theta(\Pi(s)),\\ \mu_{\mathrm{det},\theta}(s)&=\tanh q_\theta(s),\qquad u_\theta(s)=30[1+\mu_{\mathrm{det},\theta}(s)].\end{aligned}','D-4a',2398)
        add(2400,r'随机训练动作在$q_\theta(s)$上加入策略标准差对应的高斯噪声后再取tanh；固定策略评价使用式(D-4a)。投影幂等，因此参考状态处完整确定性动作等于$\tanh b_\theta$，但不必等于符号参考。故下文的基准误差$\varepsilon_{\mathrm{val}}$不能仅因神经修正为零就取为0。','R08：保留完整动作的基准误差')
        add(2400,r'符号参考按12个线路—方向组合分别拟合。输入选取归一化状态坐标集合$\mathcal I=\{6,7,8,9,10,12,13,14,15\}$与归一化动作$a$，候选库包含二次及以下的多项式项，稀疏回归阈值为0.01。辨识数据由在0、5、…、60秒中随机选择驻站时长的控制器生成，前三个运行日用于拟合，另一个运行日用于验证。同车相邻观测且站点、物理时间均递增的记录用于拟合状态增量与两类代价：原时距奖励取负，以及观测间隔内全系统的两倍候车时间加车内时间。两类代价按各线路拟合集的均值和标准差标准化，标准差下限为1。最终软参考只使用相应目标的一步代价预测，不使用早期两站动力学展开。','R08：符号拟合的数据、输入、输出与版本')
        add(2400,r'令$\widehat c_l(s,a)$为线路—方向组合$l$的标准化一步代价预测，候选动作集合为$\mathcal A_g=\{-1,-5/6,\ldots,1\}$，共13点，对应0至60秒的5秒间隔。定义温度$\tau_g=0.25$，参考映射为','R08：一步代价参考的完整定义')
        equation(2400,r'f_{\mathrm{ref}}(s)=\operatorname{clip}_{[-0.999,0.999]}\!\left[\frac{\sum_{a\in\mathcal A_g}a\exp[-\widehat c_l(s,a)/\tau_g]}{\sum_{a\in\mathcal A_g}\exp[-\widehat c_l(s,a)/\tau_g]}\right].','D-4b',2398)
        add(2400,r'符号系数在每个初始化对应的辨识完成后保持不变，不在策略训练或评价中重新拟合。策略和参考的雅可比均在同一归一化输入上计算，作用对象为$[-1,1]$中的完整确定性动作，而不是以秒计的驻站时长。记$J_\mu(s)=D_s\mu_{\mathrm{det},\theta}(s)$、$J_f(s)=D_sf_{\mathrm{ref}}(s)$，归一化导数误差为','R08—R09：约束尺度与冻结对象')
        equation(2400,r'\begin{aligned}\ell_{\mathrm{sym}}(s)&=\frac{\|J_\mu(s)-J_f(s)\|_F^2}{1+\|J_f(s)\|_F^2},\\ \mathcal L_\pi&=\mathcal L_{\pi,\mathrm{base}}+\frac{0.01}{B_{\mathrm{batch}}}\sum_{i=1}^{B_{\mathrm{batch}}}\ell_{\mathrm{sym}}(s_i).\end{aligned}','D-4c',2398)
        add(2400,r'其中$\|\cdot\|_F$为Frobenius范数，$B_{\mathrm{batch}}=256$为小批量大小，$\mathcal B$表示当前小批量，$\mathcal L_{\pi,\mathrm{base}}$为第4.3.1节的策略基础目标。最终配置未另加策略动作与参考动作的平方误差损失，行为克隆系数也为0。价值目标使用同一批后继状态上的平均导数误差，但停止该量的梯度：','R07—R08：实际未启用动作值损失')
        equation(2400,r'\Gamma_{\mathcal B}^{+}=\frac1{B_{\mathrm{batch}}}\sum_{i=1}^{B_{\mathrm{batch}}}\ell_{\mathrm{sym}}(s_i^{\prime}),\qquad \lambda_{\mathrm{sym}}=0.01.','D-4d',2398)
        equation(2400,r'\begin{aligned}y_i=\frac{r_i}{\sigma_r}+(1-d_i^{\mathrm{term}})\gamma\big[&\min_{1\le j\le3}Q_j^-(s_i^{\prime},\widetilde a_i^{\prime})-\alpha_{\mathrm{ent}}\log\pi_\theta(\widetilde a_i^{\prime}\mid s_i^{\prime})\\ &-\lambda_{\mathrm{sym}}\operatorname{sg}(\Gamma_{\mathcal B}^{+})\big].\end{aligned}','D-4e',2398)
        add(2400,r'式(D-4e)中$\sigma_r>0$为当前奖励标准差缩放量，$d_i^{\mathrm{term}}$为终止标记，$\widetilde a_i^{\prime}$由当前策略在后继状态采样，$\operatorname{sg}$表示求梯度时将括号内的值视为常数；这里$\gamma=0.99$。符号项为整批共享的标量，不是对每条后继记录独立施加的动作值惩罚，亦不通过价值目标反向更新策略。策略损失另在当前状态上计算式(D-4c)。','R07：价值目标的计算状态与梯度方向')
        add(2400,'实现中的雅可比对15维数值编码计算，包含类别编号的数值延拓偏导；这些偏导不表示真实线路或站点切换的物理导数。将下文定理用于公交连续运行量时，应固定线路、车辆、站点、时段及方向等离散条件，在相同归一化坐标中考察连续截面，并避开参考截断的不可微边界。训练样本上的平均导数损失较小，不等于整条外推线段上的统一误差上界。早期ReLU配置还需另行处理激活边界，不能用最终配置的tanh网络替代其正则性条件。','R09：离散类别、连续截面和不可微边界')
        words(2468,'先以NAU的线性映射为例','本节保留早期输出对照的数学性质，先以NAU的线性映射为例','R08：NAU结论的版本归属')
        edit(2506,'前述分析关注完整策略对状态的响应。下面单独研究固定惩罚价值算子。实际公交配置同时使用式(D-4c)的策略正则和式(D-4e)的价值目标，两种训练操作不相互替代。本节以固定参数、精确期望和有限集合为条件分析理想价值更新，不将其直接等同于随机梯度训练。','R07：价值算子与策略正则分开')
        edit(2519,'式(D-43)中的符号惩罚可在一次冻结批数据、策略和符号参考的比较中取为式(D-4d)的批平均量，对全部模式使用同一值；公交配置的相应系数为0.01，价值目标中的权重范数项关闭。实际后继动作由当前策略采样，而式(D-42)采用最大值聚合；固定策略采样聚合的性质可按附录A.1的固定策略说明处理。公交实现并未显式构造各模式转移核并按模式混合价值目标，故命题D.11是对固定惩罚这一环节的条件解释，而不是完整软符号策略训练的收敛证明。','R07：明确可对应的惩罚环节和不直接对应的聚合')
        words(2571,'NAU线性层和NMU启发的标量二次函数的正则性还需通过复合映射分析，才能用于完整网络',
              '早期NAU线性层和NMU启发的二次函数需通过完整复合映射分析；最终无NAU配置则使用第D.1节给出的完整动作及参考映射，二者的正则性不能互相代用','R08：最终与早期结构分开')

        edit(2575,'本附录围绕模型误差和置信尺度两条主线展开。第E.1至E.5节分析固定策略价值、数据覆盖及合法残差更新条件，第E.3节另给出无折扣有限事件时域的条件误差界；第E.6节只讨论指定静态误差信号的排序关系。第E.9节直接对应公交改进中的归一化置信度递推更新，解释预测误差稳定时权重仍可能持续减小的原因。','R10—R13：突出模型误差与实际置信尺度')
        edit(2595,'证明。非负概率加权满足三角不等式，且每项绝对值不超过最大值。证毕。','R16：缩短概率加权基本性质')
        edit(2602,r'固定策略后的状态核仍非负且归一化，期望奖励仍满足式(E-2)。由有限维空间的完备性、定理E.1及收缩映射不动点定理，存在唯一价值固定点$V_P^\pi$，精确迭代$V_{n+1}=T_P^\pi V_n$满足','R16：复用标准收缩结论')
        add(2634,r'**无折扣有限事件时域。** 对第5、6章的公交任务，将运营时刻、待决策事件及终止信息纳入状态，并将终止后补齐为零奖励吸收状态。本补充允许可测状态动作空间，假定策略、奖励和转移核均可测，总变差定义为可测集合上的概率差上确界。两概率测度之差的全变差质量等于这一上确界的两倍，故式(E-15)的有界函数估计仍适用于积分形式。比较模型共用状态、动作及终止定义；在对两模型均封闭的分析范围内，另假设每条路径至多有$H$个决策事件，各模型每事件奖励绝对值不超过$R_{\max}$，期望即时奖励的统一误差不超过$\epsilon_r$，后继事件、运行状态、事件间隔和终止信息的联合转移核总变差误差不超过$\epsilon_P$。对同一固定策略，令剩余$h$个事件的价值误差为','R10：加入有限事件时域的条件补充')
        equation(2634,r'e_h=\|V_h^\pi-\widehat V_h^\pi\|_\infty,\qquad e_0=0.','E-23a',2632)
        add(2634,r'在两个有限时域递推式之间加入并减去使用同一后继价值的中间项。即时奖励贡献至多$\epsilon_r$，后继价值差贡献至多$e_h$；剩余$h$个事件的价值绝对值不超过$hR_{\max}$，由式(E-15)得到联合转移误差项。因此','R10：自包含的递推证明')
        equation(2634,r'e_{h+1}\le\epsilon_r+e_h+2hR_{\max}\epsilon_P.','E-23b',2632)
        add(2634,r'从$e_0=0$开始对$h=0,\ldots,H-1$逐项求和，使用$2\sum_{h=0}^{H-1}h=H(H-1)$，得到','R10：有限事件价值误差界')
        equation(2634,r'\|V_H^\pi-\widehat V_H^\pi\|_\infty\le H\epsilon_r+H(H-1)R_{\max}\epsilon_P.','E-23c',2632)
        add(2634,r'该推导不使用严格折扣，且把事件时间和结束条件包含在转移中。$H$是决策事件数上界，不是24,000秒运营时长；同时到站车辆的序贯选择即使不推进物理时间，也占据决策事件。现有预测均方误差不是上述联合核的总变差，已有实验也未估计统一的$H$、奖励界和转移误差界。因此式(E-23c)用于说明有限时域的误差传播条件，不作为已经验证的公交数值性能保证，也不将$\gamma=1$代入式(E-23)。','R10：区分条件推导与公交已验证保证')
        edit(2651,'在式(E-6)的概率约束下，将合法转移残差集合写为闭集，并直接分析保持该集合的更新映射：','R11：显式合法残差闭集')
        # Replace the mathematical center only; the original equation number and bookmark stay.
        table = nodes[2652]
        math = table.find('.//m:oMath',NS)
        generated = convert(r'$$\begin{aligned}X_\delta&=\mathbb R^{|S|\,|A|\,|S|},\\ \mathcal C_\delta&=\{\delta\in X_\delta:P_{\mathrm{sim}}+\delta\ge0,\ \sum_{s^{\prime}}\delta(s,a,s^{\prime})=0\},\\ F&:\mathcal C_\delta\longrightarrow\mathcal C_\delta.\end{aligned}$$').find('.//m:oMath',NS)
        math.getparent().replace(math,deepcopy(generated))
        changed[2652]=table; reasons[2652]='R11：式(E-27)限定在合法闭集上的自映射'
        edit(2653,r'其中$X_\delta$是残差的环境空间，$\mathcal C_\delta$为合法残差集合，$F(\delta)$为当前残差诱导的策略、数据条件和下一次拟合共同产生的更新。有限状态动作下，概率非负与行和约束定义闭集；零残差属于其中，故集合非空，并继承最大范数下的完备性。范数仍为','R11：合法残差空间的完备性')
        words(2658,'在上述完备空间上','在上述合法残差闭集上','R11：条件固定点的准确对象')
        edit(2660,r'证明。非空闭集$\mathcal C_\delta$完备，$F$保持该集合并满足式(E-29)，由收缩映射不动点定理得到唯一固定点；反复比较迭代与固定点即得式(E-30)。证毕。','R11、R16：压缩条件固定点证明')
        edit(2661,'这是以完整残差更新收缩为前提的条件结论，不是对实际MC-WM拟合过程已经满足该条件的证明。唯一固定点也不必等于真实转移差，识别真实残差仍需数据可辨识性及拟合一致性。','R11：条件结论的用途')
        edit(2666,r'本节仅比较特定误差信号的静态排序。$w_{\mathrm{rank}}>0$为指数型正值信号，$\tau_{\mathrm{rank}}>0$为尺度，$c_{\mathrm{rank}}$为加性常数，$\epsilon_{\mathrm{rank}}$为比较域内误差波动界。它们不是第6章采用$1/(1+e/\tau)$局部评分并递推后继置信得到的实际权重，也不是H2O+由联合分类器与状态动作分类器构造的条件转移比。','R12：静态理想分数不等于两算法实际权重')
        edit(2673,r'式(E-32)、式(E-33)是以下结论的指定前提。其一种概率解释为：对同一个$d$维观测$y$，两类条件密度分别是均值为$\mu_{\mathrm{real}}$、$\mu_{\mathrm{resid}}$且协方差均为$\sigma^2I$的高斯密度，类别先验相等，$\varphi_{\mathrm{cls}}$表示“real”类的后验概率；取$e_{\mathrm{real}}=\|y-\mu_{\mathrm{real}}\|_2^2$、$e_{\mathrm{resid}}=\|y-\mu_{\mathrm{resid}}\|_2^2$及$\tau_{\mathrm{rank}}=2\sigma^2$，对数密度比即为式(E-33)。若误差定义为逐维均方值，尺度应改为$2\sigma^2/d$；协方差不同时还需保留行列式及相应二次型项。式(E-32)来自另行指定的指数信号，不由上述分类解释自动推出。','R12：明确平方误差、协方差、类别方向与维度')
        words(2699,'未来置信备份','后继置信度递推更新','R12—R13：统一实际递推对象')
        words(2716,'第6.4.4节的配对实验','第6.4.4至6.4.5节的配对实验','R13：尺度理论同时连接主结果')
        add(2716,r'例如无终止、局部评分恒为0.5且$\gamma_\Delta=0.9$时，原精确权重为$0.5^{10}\approx0.00098$，归一化权重为0.5。该例解释的是更新强度的尺度差，固定0.5权重的对照仍须保留，不能将其改善全部归因于更准确的状态排序。','R13：突出置信尺度的直接解释')

        for i in range(2719,2756):
            for old,new in [('已完成50,000次更新的离线演员','完成50,000次离线更新的策略网络'),('分别冻结该演员','分别固定该策略网络参数而不再训练'),('冻结离线初始化','仅离线预训练，不继续更新'),('共同冻结源回放','各组共用且保持不变的源经验数据'),('冻结初始化到模型分支','仅离线预训练到模型分支'),('名义25,000次源事件预算','计划采集约25,000个源事件'),('固定端点','训练结束时保存的策略'),('归一化置信备份','归一化置信度递推更新'),('名义值','计划值'),('共同环境块','共同场景组'),('同一环境块','同一场景组'),('同一目标环境块','同一目标场景组'),('归一化备份','归一化置信度递推更新'),('共38条训练','共38次训练'),('上下文调节','按源仿真偏差调节模型损失系数'),('新增五条MC-WM续训','新增五次MC-WM继续训练')]:
                if replace_words(nodes[i],old,new):
                    changed[i]=nodes[i]; reasons[i]='R18—R21：实验对象、版本和统计单位说明'
        edit(2740,'仿真标定误差涉及状态转移规律。本节使用6个完整目标场景组的离线数据，并固定16,384次计划源交互，分别改变道路平均速度、OD强度、新增信号路口数量及必要上下客服务时间，再设置联合偏差。速度偏差25%表示速度乘以0.75，OD和服务时间偏差25%表示相应量乘以1.25。经核对，新增路口沿用第6.4.4节的泊松计数规则：均值为新增路口数量的比例参数乘以线路既有信号路口数，再除以仿真路段数，沿用既有绿灯比例和信号周期产生延误。联合偏差50%同时使用0.5倍速度、1.5倍OD和必要服务时间，以及0.5的新增路口数量比例参数。必要服务时间不同于策略选择的附加驻站动作。目标SUMO保持原设置。本节保留归一化置信度改进前的单训练结果，训练预算、源采样和目标测试日均与第F.2节不同。','R20：按早期代码核定场景划分与物理偏差')
        words(2744,'停站误差主要改变服务时长','必要服务时间误差主要改变上下客时长','R20：区分服务参数和控制动作')
        words(2754,'同预算适配按完整源运行日','策略网络和价值网络更新次数相同的适配比较按完整源运行日','R21：说明匹配的更新预算')
        add(2756,'第6.4.5节的车头时距奖励补充试验采用66001、66002两个初始化，在正常标定与联合75%源偏差下新增4次MC-WM训练，并复用对应4个H2O+策略。MC-WM的模型损失系数为0.125，混合经验损失系数为0.75。各策略在两档需求、每档五个既有评价日中，分别直接执行与叠加行程完成保障，新增MC-WM评价80回合，另复用H2O+评价80回合。该组与乘客时间目标下20次独立初始化的主要比较分别统计。','R21：补齐车头时距奖励批次登记')

        # C.8's renamed heading also changes its existing TOC label, not its anchor.
        words(220,'C.8 识别速度','C.8 持续证据下的后验阈值','R15：目录同步附录小节名称') if 'C.8 识别速度' in visible(body[220]) else None
        toc_index = next(i for i in range(83,251) if visible(body[i]).startswith('C.8 '))
        if toc_index not in changed:
            words(toc_index,'识别速度','持续证据下的后验阈值','R15：目录同步附录小节名称')

        for node in list(changed.values())+[n for group in additions.values() for n in group]:
            link_supplementary_equations(node)

        patches = []
        for i,(a,b) in enumerate(spans):
            if i not in changed and i not in additions:
                continue
            before = b''.join(helper.serialize(n) for n in additions.get(i,[]))
            content = helper.serialize(changed[i]) if i in changed else original[a:b]
            patches.append((a,b,before+content))
        updated = original
        for a,b,content in reversed(patches):
            updated=updated[:a]+content+updated[b:]
        header=source.read('word/header16.xml')
        assert '探索性比较' in header.decode()
        header=header.replace('探索性比较'.encode(),'补充实验'.encode())
        with ZipFile(OUT,'w') as target:
            for info in source.infolist():
                data=updated if info.filename=='word/document.xml' else header if info.filename=='word/header16.xml' else source.read(info.filename)
                target.writestr(deepcopy(info),data)
    validation=dict(source=str(SOURCE),output=str(OUT),abstract_replacements=[66,67,68,74,75,76],
                    body_changed_indices=sorted(i for i in changed if 251<=i<1787),
                    replaced_blocks=sorted(changed),added_blocks=sum(map(len,additions.values())),
                    unchanged_original_equation_numbers=True,
                    modified_existing_equations=['E-27'],new_equations=['D-4'+c for c in 'abcde']+['E-23'+c for c in 'abc'],
                    changes=[dict(index=i,reason=reasons[i],before=visible(body[i]),after=visible(changed[i])) for i in sorted(changed)],
                    insertions=[dict(before_index=i,text=visible(n)) for i,group in additions.items() for n in group])
    (HERE/'修改记录.json').write_text(json.dumps(validation,ensure_ascii=False,indent=2))
    print(json.dumps({k:v for k,v in validation.items() if k not in ('changes','insertions')},ensure_ascii=False,indent=2))


if __name__=='__main__':
    main()
