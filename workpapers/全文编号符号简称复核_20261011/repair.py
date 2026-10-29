"""Apply scoped notation, abbreviation, and comparison-attribution corrections."""
from copy import deepcopy
from pathlib import Path
from zipfile import ZipFile
import json
import shutil
import sys

from lxml import etree as E
import pypandoc

from scan import SOURCE, HERE, ROOT, W, M, NS, content

sys.path.insert(0, str(ROOT / 'workpapers/七章重写稿整合_20261010'))
from integrate import body_spans

OUT = ROOT / 'output/排版完成_全文编号符号简称复核_20261011.docx'
math_cache = {}


def math(latex):
    if latex not in math_cache:
        temporary = HERE / '符号转换中间件.docx'
        pypandoc.convert_text('$' + latex + '$', 'docx', format='markdown', outputfile=str(temporary))
        with ZipFile(temporary) as z:
            generated = E.fromstring(z.read('word/document.xml'))
        math_cache[latex] = generated.find('.//' + M + 'oMath')
        assert math_cache[latex] is not None, latex
    return deepcopy(math_cache[latex])


def replace_text(p, old, new):
    nodes = list(p.iter(W + 't'))
    joined = ''.join(n.text or '' for n in nodes)
    assert old in joined, (old, joined[:160])
    start = joined.index(old)
    end = start + len(old)
    offset = 0
    inserted = False
    for node in nodes:
        value = node.text or ''
        right = offset + len(value)
        if right > start and offset < end:
            left_cut, right_cut = max(0, start-offset), min(len(value), end-offset)
            node.text = value[:left_cut] + (new if not inserted else '') + value[right_cut:]
            node.set('{http://www.w3.org/XML/1998/namespace}space', 'preserve')
            inserted = True
        offset = right


def replace_math(p, index, latex):
    old = p.findall('.//' + M + 'oMath')[index]
    fresh = math(latex)
    properties = old.find('.//' + W + 'rPr')
    if properties is not None:
        for run in fresh.iter(M + 'r'):
            if run.find(W + 'rPr') is None:
                run.insert(1 if run.find(M + 'rPr') is not None else 0, deepcopy(properties))
    for child in list(old):
        old.remove(child)
    old.extend(list(fresh))


def append(p, *parts):
    properties = p.find('.//' + W + 'rPr')
    for part in parts:
        if isinstance(part, E._Element):
            p.append(part)
        else:
            run = E.SubElement(p, W + 'r')
            if properties is not None:
                run.append(deepcopy(properties))
            t = E.SubElement(run, W + 't')
            t.set('{http://www.w3.org/XML/1998/namespace}space', 'preserve')
            t.text = part


def prepend(p, *parts):
    temporary = E.Element(W + 'p')
    properties = p.find('.//' + W + 'rPr')
    if properties is not None:
        sample = E.SubElement(temporary, W + 'r')
        sample.append(deepcopy(properties))
    append(temporary, *parts)
    children = list(temporary)
    if properties is not None:
        children = children[1:]
    index = 1 if p.find(W + 'pPr') is not None else 0
    for child in children:
        p.insert(index, child)
        index += 1


def rename_atom(p, old, new, sub=None, bare_only=False, sub_latex=False):
    """Rename a complete atomic base; preserve existing indices and accents."""
    count = 0
    for t in list(p.iter(M + 't')):
        if t.text != old:
            continue
        run = t.getparent()
        if run.tag != M + 'r' or len(run.findall(M + 't')) != 1:
            raise ValueError(('non-atomic run', old))
        argument = run.getparent()
        owner = argument.getparent()
        indexed = argument.tag == M + 'e' and owner.tag in (M + 'sSub', M + 'sSubSup')
        if bare_only and indexed:
            continue
        t.text = new
        if sub:
            label = sub if sub_latex else r'\mathrm{' + sub + '}'
            if indexed:
                target = owner.find(M + 'sub')
                prefix = math(label + ',')
                for n, child in enumerate(list(prefix)):
                    target.insert(n, child)
            else:
                replacement = E.Element(M + 'sSub')
                base = E.SubElement(replacement, M + 'e')
                argument.replace(run, replacement)
                base.append(run)
                target = E.SubElement(replacement, M + 'sub')
                target.extend(list(math(label)))
        count += 1
    return count


def split_math_run(p, old, pieces):
    for t in list(p.iter(M + 't')):
        if t.text == old:
            run = t.getparent()
            for piece in reversed(pieces):
                fresh = deepcopy(run)
                fresh.find(M + 't').text = piece
                run.addnext(fresh)
            run.getparent().remove(run)


def replace_inline_math(p, old, latex):
    for t in p.iter(W + 't'):
        if old not in (t.text or ''):
            continue
        before, after = t.text.split(old, 1)
        run = t.getparent()
        suffix = deepcopy(run)
        suffix.find(W + 't').text = after
        t.text = before
        fresh = math(latex)
        run.addnext(fresh)
        fresh.addnext(suffix)
        return
    raise ValueError(('inline math not found', old))


def replace_subscript(p, base, old_sub, new_sub):
    count = 0
    for script in list(p.iter(M + 'sSub')) + list(p.iter(M + 'sSubSup')):
        e, sub = script.find(M + 'e'), script.find(M + 'sub')
        if e is not None and sub is not None and content(e) == base and content(sub) == old_sub:
            for child in list(sub):
                sub.remove(child)
            sub.extend(list(math(new_sub)))
            count += 1
    return count


def symbol_row(table, latex, meaning):
    row = deepcopy(table.findall(W + 'tr')[-1])
    for cell in row.findall(W + 'tc'):
        for child in list(cell):
            if child.tag != W + 'tcPr':
                cell.remove(child)
    for cell, value in zip(row.findall(W + 'tc'), (math(latex), meaning)):
        p = E.SubElement(cell, W + 'p')
        sample = table.findall(W + 'tr')[-1].find('.//' + W + 'pPr')
        if sample is not None:
            p.append(deepcopy(sample))
        append(p, value)
    table.append(row)


def main():
    backup = HERE / '修改前_参考文献排版修复版.docx'
    if not backup.exists():
        shutil.copy2(SOURCE, backup)
    with ZipFile(SOURCE) as z:
        original = z.read('word/document.xml')
        doc = E.fromstring(original)
        body = doc.find(W + 'body')
        before = [E.tostring(p) for p in body]

        replacements = {
            80: [('第3、4章目标网络参数以上标负号标记，第6章目标置信评价器采用其首次定义的横线记号', '目标网络参数及目标置信评价器以上标负号标记')],
            65: [('RE-SAC相较于Soft Actor-Critic（SAC）方法分别降低', '在相同RE-SAC训练设置下，可学习嵌入相较于独热编码分别降低'),
                 ('同时，RE-SAC相对SAC的', '同时，RE-SAC相对软演员评论家方法（Soft Actor-Critic，SAC）的')],
            66: [('贝叶斯自适应策略正则化（Bayesian Adaptive Policy Regularization，BAPR）', '贝叶斯遗忘分段鲁棒方法（Bayesian Amnesic Piecewise-Robust SAC，BAPR）'),
                 ('因果符号贝叶斯自适应策略正则化（Causal-Symbolic Bayesian Adaptive Policy Regularization，CS-BAPR）', '因果符号增强的BAPR（Causal-Symbolic BAPR，CS-BAPR）')],
            68: [('乘客OD强度', '乘客起讫点（Origin-Destination，OD）需求强度')],
            73: [('relative to SAC under the headway objective', 'relative to Soft Actor-Critic (SAC) under the headway objective')],
            74: [('this thesis develops Causal-Symbolic Bayesian Adaptive Policy Regularization (CS-BAPR) based on Bayesian Adaptive Policy Regularization (BAPR).',
                  'this thesis builds on Bayesian Amnesic Piecewise-Robust SAC (BAPR) to develop Causal-Symbolic BAPR (CS-BAPR).')],
            258: [('SUMO为构建', '微观交通仿真工具（Simulation of Urban MObility，SUMO）为构建')],
            259: [('客流起讫分布', '客流起讫点（Origin-Destination，OD）分布')],
            260: [('H2O及其扩展H2O+', '混合离线与在线强化学习（Hybrid Offline-and-Online Reinforcement Learning，H2O）及其改进框架H2O+')],
            301: [('作者前期的BAPR研究（Bayesian Amnesic Piecewise-Robust）', '作者前期的贝叶斯遗忘分段鲁棒方法（Bayesian Amnesic Piecewise-Robust SAC，BAPR）研究'),
                  ('本文在此基础上研究贝叶斯自适应策略正则化（Bayesian Adaptive Policy Regularization，BAPR），并进一步提出因果符号贝叶斯自适应策略正则化（Causal-Symbolic Bayesian Adaptive Policy Regularization，CS-BAPR）',
                   '本文在此基础上进一步提出因果符号增强的BAPR（Causal-Symbolic BAPR，CS-BAPR）')],
            326: [('MuJoCo动力学仿真平台', '多关节接触动力学仿真平台（Multi-Joint dynamics with Contact，MuJoCo）')],
            322: [('基于模型的离线强化学习方法MOReL', '基于模型的离线强化学习方法（Model-Based Offline Reinforcement Learning，MOReL）')],
            743: [('两层ReLU网络', '两层修正线性单元（Rectified Linear Unit，ReLU）网络')],
            906: [('训练期峰值已分配显存为', '训练期峰值已分配显存以兆二进制字节（Mebibyte，MiB）计，依次为'),
                  ('CPU平均单次调用', '中央处理器（Central Processing Unit，CPU）平均单次调用')],
            986: [('算术输出和KAN关闭', '算术输出和柯尔莫哥洛夫-阿诺德网络（Kolmogorov-Arnold Network，KAN）关闭'),
                  ('普通MLP保留', '普通多层感知机（Multilayer Perceptron，MLP）保留')],
            998: [('ReLU、tanh、算术输出', 'ReLU、双曲正切函数tanh、算术输出')],
            1223: [('RORL限制', '鲁棒离线强化学习（Robust Offline Reinforcement Learning，RORL）限制')],
            1306: [('曲线下面积（AUC）', '曲线下面积（Area Under the Curve，AUC）')],
            1419: [('SUMO 1.25.0与TraCI API协议号22，同组版本保持一致',
                    'SUMO 1.25.0及交通控制接口（Traffic Control Interface，TraCI），应用程序编程接口（Application Programming Interface，API）的协议号为22；同组版本保持一致')],
            1447: [('SiLU激活', 'Sigmoid线性单元（Sigmoid Linear Unit，SiLU）激活')],
            1496: [('RCPO和PID拉格朗日方法', '奖励约束策略优化（Reward Constrained Policy Optimization，RCPO）和比例-积分-微分（Proportional-Integral-Derivative，PID）拉格朗日方法')],
            1502: [('源模型状态RMSE', '源模型状态均方根误差（Root Mean Squared Error，RMSE）')],
            1538: [('总变差定义', '总变差（Total Variation，TV）定义')],
            1628: [('GPU预热后', '图形处理器（Graphics Processing Unit，GPU）预热后')],
        }
        for i, pairs in replacements.items():
            for old, new in pairs:
                replace_text(body[i], old, new)

        # A sampled transition has the same indexed target in both lines.
        assert rename_atom(body[470], 'y', 'y', 'k', bare_only=True, sub_latex=True) == 1
        assert rename_atom(body[471], 'y', 'y', 'k', sub_latex=True) == 1
        replace_math(body[986], 1, r'\tilde a')
        replace_math(body[1075], 4, r'd_i^{\mathrm{svc},0}')
        replace_math(body[1076], 0, r'\mathbb E[d_i^{\mathrm{svc}}\mid h_i]\approx d_i^{\mathrm{svc},0}+\eta_i^{\mathrm{board}}m_{\mathrm{dem}}\lambda_i h_i.')

        # Reuse the already-defined conservative coefficient, without changing signs.
        replace_math(body[774], 0, r'q_c(s,a)=\bar Q(s,a)+\beta_{\mathrm{lcb}}\sigma_Q(s,a)')
        replace_math(body[774], 1, r'\beta_{\mathrm{lcb}}\leq0')
        replace_math(body[775], 0, r'q_c(s,a_2)-q_c(s,a_1)=\bar Q(s,a_2)-\bar Q(s,a_1)+\beta_{\mathrm{lcb}}[\sigma_Q(s,a_2)-\sigma_Q(s,a_1)].')

        replace_math(body[1091], 2, r'N_{w,j_{\mathrm{alg}},j_{\mathrm{win}}}^{p}')
        replace_math(body[1091], 3, r'N_{w,j_{\mathrm{alg}},j_{\mathrm{win}}}^{0}')
        replace_math(body[1092], 0, r'\Delta N_{w,j_{\mathrm{alg}},j_{\mathrm{win}}}=N_{w,j_{\mathrm{alg}},j_{\mathrm{win}}}^{p}-N_{w,j_{\mathrm{alg}},j_{\mathrm{win}}}^{0}.')
        replace_math(body[1094], 0, r'\varepsilon_{j_{\mathrm{alg}},j_{\mathrm{win}}}=\max\{0.1N_{w,j_{\mathrm{alg}},j_{\mathrm{win}}}^{0},20\}.')
        replace_math(body[1095], 0, r'\Delta N_{w,j_{\mathrm{alg}},j_{\mathrm{win}}}\leq\varepsilon_{j_{\mathrm{alg}},j_{\mathrm{win}}}')

        split_math_run(body[496], '\u2001\u2001w', ['\u2001\u2001', 'w'])
        for i in (496, 1407):
            assert rename_atom(body[i], 'w', 'w', 'IS') > 0
        replace_inline_math(body[497], 'w(x)', r'w_{\mathrm{IS}}(x)')
        append(body[1407], '其中，', math('N'), '为参与计算的权重个数，', math(r'w_{\mathrm{IS},i}'), '为第', math('i'), '条记录的条件转移比率权重。')
        split_math_run(body[1503], 'viol,u', ['viol,', 'u'])
        assert rename_atom(body[1503], 'u', 'n', 'val') == 4

        for i in range(2120, 2129):
            split_math_run(body[i], 'Gw', ['G', 'w'])
            split_math_run(body[i], '\u2001w', ['\u2001', 'w'])
            rename_atom(body[i], 'G', 'G', 'act', bare_only=True)
            rename_atom(body[i], 'w', 'a', 'inv')
        rename_atom(body[2132], 'I', 'I', 'n_a', sub_latex=True)
        replace_text(body[2131], '，有', '，以')
        append(body[2131], math(r'I_{n_a}'), '表示动作维数对应的单位矩阵，有')

        # Physical and augmented states must not be confused with x_k=[s_k;a_k].
        for i in list(range(2108, 2113)) + list(range(2146, 2168)):
            rename_atom(body[i], 'x', 'x', 'phys')
            replace_subscript(body[i], 'F', 'dyn', r'\mathrm{phys}')
        for i in range(2207, 2272):
            rename_atom(body[i], 'x', 'x', 'aug')
            rename_atom(body[i], 'X', 'X', 'aug')
            for script in list(body[i].iter(M + 'sSub')):
                sub = script.find(M + 'sub')
                e = script.find(M + 'e')
                if e is not None and content(e) == 'F' and sub is not None:
                    for t in sub.iter(M + 't'):
                        if t.text == 'dyn': t.text = 'aug'
                        elif t.text == 'dyn,e': t.text = 'aug,e'
        append(body[2108], '这里', math(r'x_{\mathrm{phys}}'), '表示物理状态，', math(r'F_{\mathrm{phys}}'), '表示物理状态转移映射。')
        append(body[2207], '增广转移映射记为', math(r'F_{\mathrm{aug}}'), '；参考动作与即时奖励在增广状态上的写法，表示先取物理分量再调用原映射。')

        for i in range(2338, 2344):
            replace_subscript(body[i], 'c', 'pen', r'\mathrm{bel}')
        append(body[2339], math(r'c_{\mathrm{bel}}'), '为非负的置信调节量，与合成价值惩罚', math(r'c_{\mathrm{pen}}(s,a)'), '区分。')

        for i in list(range(2453, 2461)) + list(range(2466, 2473)):
            replace_subscript(body[i], 'e', 'map', r'\mathrm{path}')
            for t in body[i].iter(M + 't'):
                if t.text == 'map,+': t.text = 'path,+'
        for i in (2477, 2478, 2479, 2480, 2488, 2489, 2490):
            split_math_run(body[i], '2ca', ['2', 'c', 'a'])
            split_math_run(body[i], '2cx', ['2', 'c', 'x'])
            split_math_run(body[i], 'wc', ['w', 'c'])
            rename_atom(body[i], 'c', 'c', 'quad')
        for i in (2484, 2486):
            rename_atom(body[i], 'w', 'c', 'NAU')
        rename_atom(body[2490], 'w', 'omega', 'quad')
        for t in body[2490].iter(M + 't'):
            if t.text == 'omega': t.text = 'ω'
        append(body[2490], '这里', math(r'c_{\mathrm{quad}}'), '为二次项系数，', math(r'\omega_{\mathrm{quad}}'), '为乘性缩放系数。')
        replace_text(body[2485], '固定其余坐标，对第', '其中各坐标的NAU权重记为')
        first_math = body[2485].find('.//' + M + 'oMath')
        fresh = math(r'c_{\mathrm{NAU},i}')
        first_math.addprevious(fresh)
        fresh.addnext(E.Element(W + 'r'))
        fresh.getnext().append(E.Element(W + 't'))
        fresh.getnext()[0].text = '。固定其余坐标，对第'

        for script in list(body[2412].iter(M + 'sSubSup')):
            if content(script.find(M + 'e')) == 'Q' and content(script.find(M + 'sub')) == 'j':
                script.getparent().replace(script, math(r'Q_{\phi_j^-}')[0])
        replace_text(body[2554], '则', '其中，')
        append(body[2554], math(r'\varepsilon_{\mathrm{sym}}'), '为符号模型与参考映射的导数误差上界，', math(r'\varepsilon_{\mathrm{pol}}'), '为策略与同一符号模型的导数误差上界。则')
        replace_text(body[2564], '若策略也满足', '这里，')
        append(body[2564], math(r'\varepsilon_{\mathrm{IRM}}'), '表示该共同符号模型在所列各环境中的统一导数误差上界。若策略也满足')
        prepend(body[2580], '其中，', math(r'\varepsilon_{\mathrm{emp}}'), '和', math(r'\Delta_{\mathrm{gen}}'), '分别表示给定的经验导数误差界及其推广误差界。')

        for i in range(2649, 2656):
            rename_atom(body[i], 'H', 'K', 'max')
            rename_atom(body[i], 'h', 'n', 'rem')
        rename_atom(body[2695], 'I', 'I', 'd')
        append(body[2695], '这里', math(r'I_d'), '为', math('d'), '阶单位矩阵。')

        # The two Unicode epsilon variants were used for the same error bounds.
        for i, p in enumerate(body):
            if 1786 <= i <= 2001 or (p.find(W+'pPr/'+W+'pStyle') is not None and p.find(W+'pPr/'+W+'pStyle').get(W+'val','').startswith('TOC')):
                continue
            for t in p.iter(M + 't'):
                if t.text: t.text = t.text.replace('ϵ', 'ε')
        for tr in body[81].findall(W + 'tr'):
            cells = tr.findall(W + 'tc')
            if content(cells[0]) == 'τ，n_{int}，N_{on}':
                append(cells[0].find(W+'p'), '，', math(r'n_{\mathrm{val}}'))
                append(cells[1].find(W+'p'), '；验证时点序号')
        for latex, meaning in [
            (r'\lambda_{\mathrm{ale}},\lambda_{\mathrm{epi}},\kappa_j', '目标网络权重范数项系数、目标集成方差项系数及第j个目标网络的未加权权重范数'),
            (r'B_{\mathrm{batch}},K_{\max},n_{\mathrm{rem}}', '训练小批量大小、有限时域分析的决策事件数上界及剩余事件数'),
            (r'w_{\mathrm{IS},i},c_{\mathrm{NAU},i},c_{\mathrm{bel}}', '条件转移比率权重、NAU线性权重及置信反馈调节量'),
            (r'G_{\mathrm{act}},a_{\mathrm{inv}},I_{n_a}', '一般对角执行器增益矩阵、逆变换命令及动作维数对应的单位矩阵'),
            (r'x_{\mathrm{phys}},x_{\mathrm{aug}},F_{\mathrm{phys}},F_{\mathrm{aug}}', '执行器分析中的物理状态、增广状态及各自的状态转移映射'),
            (r'e_{\mathrm{map}},e_{\mathrm{path}}', '策略与参考的输入误差映射，以及沿指定路径形成的误差函数'),
            (r'\varepsilon_{\mathrm{sym}},\varepsilon_{\mathrm{pol}},\varepsilon_{\mathrm{IRM}}', '符号模型误差、策略拟合误差及跨环境统一导数误差的上界'),
            (r'\varepsilon_{\mathrm{emp}},\Delta_{\mathrm{gen}}', '给定的经验导数误差界及其推广误差界'),
        ]:
            symbol_row(body[81], latex, meaning)

        spans = body_spans(original)
        changed = [i for i,p in enumerate(body) if E.tostring(p) != before[i]]
        data = original
        journal = []
        for i in reversed(changed):
            start, end = spans[i]
            journal.append({'block': i, 'before': content(E.fromstring(before[i])), 'after': content(body[i])})
            data = data[:start] + E.tostring(body[i], encoding='utf-8', with_tail=False) + data[end:]
        parts = [(deepcopy(info), data if info.filename == 'word/document.xml' else z.read(info.filename)) for info in z.infolist()]
    with ZipFile(OUT, 'w') as z:
        for info, data in parts: z.writestr(info, data)
    (HERE/'逐处修改记录.json').write_text(json.dumps(list(reversed(journal)), ensure_ascii=False, indent=2))
    print(json.dumps({'output': str(OUT), 'changed_blocks': len(changed), 'math_snippets': len(math_cache)}, ensure_ascii=False))


if __name__ == '__main__':
    main()
