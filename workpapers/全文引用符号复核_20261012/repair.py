"""Repair confirmed reference and notation faults without rebuilding the DOCX."""
from copy import deepcopy
from zipfile import ZipFile
import json
import re
import shutil
import sys
import importlib.util

from lxml import etree as E
from audit import BASE, ROOT, HERE, W, M, NS, text, style

sys.path.insert(0, str(ROOT / 'workpapers/七章重写稿整合_20261010'))
from integrate import body_spans
spec = importlib.util.spec_from_file_location('previous_notation_repair', ROOT / 'workpapers/全文编号符号简称复核_20261011/repair.py')
old = importlib.util.module_from_spec(spec)
spec.loader.exec_module(old)
old.HERE = HERE
OUT = ROOT / 'output/排版完成_编号符号引用复核_20261012.docx'


def alias(p, source, latex):
    matches = [n for n in p.iter() if n.tag.startswith(M) and
               E.QName(n).localname in {'sSub', 'sSup', 'sSubSup', 'acc', 'bar'} and text(n) == source]
    for n in matches:
        fresh = old.math(latex)
        props = n.find('.//' + W + 'rPr')
        if props is not None:
            for r in fresh.iter(M + 'r'):
                existing = r.find(W + 'rPr')
                if existing is not None:
                    r.remove(existing)
                r.append(deepcopy(props))
        parent = n.getparent()
        at = parent.index(n)
        parent.remove(n)
        for child in fresh:
            parent.insert(at, deepcopy(child))
            at += 1
    return len(matches)


def named_atom(p, source, latex, bare_only=False):
    for m in p.findall('.//' + M + 'oMath'):
        for r in list(m.iter(M + 'r')):
            ts = r.findall(M + 't')
            if len(ts) != 1 or (ts[0].text or '') != source:
                continue
            if bare_only and r.getparent().tag == M + 'e' and r.getparent().getparent().tag in {M + 'sSub', M + 'sSubSup'}:
                continue
            fresh = old.math(latex)
            parent, at = r.getparent(), r.getparent().index(r)
            parent.remove(r)
            for child in fresh:
                parent.insert(at, deepcopy(child))
                at += 1


def narrow_ref(p, label):
    end = next(r for r in p if r.find(W + 'fldChar') is not None and
               r.find(W + 'fldChar').get(W + 'fldCharType') == 'end')
    run = next(r for r in p if r.find(W + 't') is not None and r.find(W + 't').text == label)
    p.remove(end)
    p.insert(p.index(run) + 1, end)


def wrap_argument(p, preceding):
    m = p.find('.//' + M + 'oMath')
    for i, child in enumerate(m):
        if text(child) == preceding and i + 1 < len(m) and m[i + 1].tag == M + 'f':
            f = m[i + 1]
            d = E.Element(M + 'd')
            argument = E.SubElement(d, M + 'e')
            m.replace(f, d)
            argument.append(f)
            return
    raise ValueError(('fraction argument not found', preceding))


def citations(body):
    reference_anchors = {}
    for p in body:
        if style(p) == 'bibliography':
            number = int(re.match(r'\[(\d+)\]', text(p))[1])
            reference_anchors[number] = next(n for n in p.xpath('.//w:bookmarkStart/@w:name', namespaces=NS) if n.startswith('ref-'))
    count = 0
    for p in body:
        if style(p) == 'bibliography' or style(p).startswith('TOC'):
            continue
        for t in list(p.iter(W + 't')):
            if t.getparent().getparent().tag == W + 'hyperlink':
                continue
            value = t.text or ''
            matches = list(re.finditer(r'\[([1-9]\d*(?:[,，]\s*[1-9]\d*)*)\]', value))
            if not matches:
                continue
            run = t.getparent()
            if run.tag != W + 'r':
                continue
            # Supplied prose has literal numeric citations; retain its exact punctuation.
            parent, at = run.getparent(), run.getparent().index(run)
            def insert(value, number=None):
                nonlocal at, count
                if not value:
                    return
                r = deepcopy(run)
                for child in list(r):
                    if child.tag != W + 'rPr':
                        r.remove(child)
                node = E.SubElement(r, W + 't')
                node.text = value
                node.set('{http://www.w3.org/XML/1998/namespace}space', 'preserve')
                if number is not None:
                    link = E.Element(W + 'hyperlink', {W + 'anchor': reference_anchors[number], W + 'history': '1'})
                    link.append(r)
                    item = link
                    count += 1
                else:
                    item = r
                parent.insert(at, item)
                at += 1
            offset = 0
            for match in matches:
                numbers = [int(n) for n in re.findall(r'\d+', match[1])]
                assert all(n in reference_anchors for n in numbers), (value, numbers)
                insert(value[offset:match.start()] + '[')
                last = 0
                for n in re.finditer(r'\d+', match[1]):
                    insert(match[1][last:n.start()])
                    insert(n[0], int(n[0]))
                    last = n.end()
                insert(match[1][last:] + ']')
                offset = match.end()
            insert(value[offset:])
            parent.remove(run)
    return count


def main():
    backup = HERE / '修改前.docx'
    if not backup.exists():
        shutil.copy2(BASE, backup)
    with ZipFile(BASE) as source:
        raw = source.read('word/document.xml')
        root = E.fromstring(raw)
        body = list(root.find(W + 'body'))
        before = [E.tostring(p) for p in body]
        labels = json.loads((HERE / '编号与引用检查.json').read_text())['equations']
        eq = lambda key: body[labels[key]['i']]

        # Updating these four fields must never replace the explanatory sentence.
        for i, label in [(957, '图4-1'), (1017, '表4-7'), (1020, '图4-7'), (1447, '表5-18')]:
            narrow_ref(body[i], label)
        # Appendix bookmarks must cover the figure/table number, not its entire title.
        for i, anchor in [(2862, 'figF_1'), (2880, 'tblF_2'), (2879, 'figF_3')]:
            p = body[i]
            start = next(n for n in p.iter(W + 'bookmarkStart') if n.get(W + 'name') == anchor)
            end = next(n for n in p.iter(W + 'bookmarkEnd') if n.get(W + 'id') == start.get(W + 'id'))
            number_run = next(r for r in p if re.fullmatch(r'F-[123]\s*', text(r)))
            p.remove(end)
            p.insert(p.index(number_run) + 1, end)
        for t in body[1009].iter(W + 't'):
            if (t.text or '').startswith(' 20倍'):
                t.set('{http://www.w3.org/XML/1998/namespace}space', 'preserve')
        mark = body[1318].find(W + 'bookmarkStart')
        assert mark.get(W + 'name') == '_Toc242183088'
        end = next(n for p in body for n in p.iter(W + 'bookmarkEnd') if n.get(W + 'id') == mark.get(W + 'id'))
        body[1318].remove(mark)
        end.getparent().remove(end)
        body[1319].insert(1, mark)
        body[1319].append(end)

        # Common theory must use the same parameter, terminal, and clock notation.
        for i in (367, 368, 375):
            for t in body[i].iter(M + 't'):
                if any(n.tag == M + 'sub' and text(n) == 'ent' for n in t.iterancestors()):
                    continue
                if t.text:
                    t.text = re.sub(r'(?<![A-Za-z])t(?![A-Za-z])', 'k', t.text)
        for i in (376, 377, 408, 411):
            for old_name, latex in [('‾{ϕ}_{1}', r'\phi_1^-'), ('‾{ϕ}_{2}', r'\phi_2^-'),
                                    ('‾{ϕ}_{j}', r'\phi_j^-'), ('‾{ϕ}', r'\phi^-')]:
                alias(body[i], old_name, latex)
            # Superscripts on parameters must remain inside Q's parameter subscript.
            for t in body[i].iter(M + 't'):
                if t.text == 'φ': t.text = 'ϕ'
        for i in (376, 377, 410):
            named_atom(body[i], 'd', r'd^{\mathrm{term}}', bare_only=True)
        for i in (362, 363):
            alias(body[i], 'H^{f}', r'h^{f,*}')
            alias(body[i], 'H^{b}', r'h^{b,*}')
        for i in (388, 389): alias(body[i], 'p_{∘}', r'p^{\circ}')
        for i in (383, 384): alias(body[i], 'P_{s,a}', r'\mathcal U_{s,a}')
        old.replace_math(body[399], 0, r'\iota\in\{\mathrm{sim},\mathrm{tgt}\}')
        for i in range(399, 414):
            alias(body[i], 'P_{src}', r'P_{\mathrm{sim}}')
            for t in body[i].iter(M + 't'):
                if t.text == 'src': t.text = 'sim'
        for i in (425, 426):
            alias(body[i], 'f_{src}', r'f_{\mathrm{sim}}')
            for sub in body[i].iter(M + 'sub'):
                if text(sub) == 'src':
                    for child in list(sub): sub.remove(child)
                    sub.extend(list(old.math(r'\mathrm{sim}')))
            for t in body[i].iter(M + 't'):
                if t.text == 'src': t.text = 'sim'
        for i in (408, 409): named_atom(body[i], 'e', r'e_{\mathrm{IQL}}')
        named_atom(body[411], 'w', r'w_{\mathrm{IQL}}', bare_only=True)
        for i in (419, 420, 421): named_atom(body[i], 'H', r'H_{\mathrm{roll}}')
        for i in (422, 423): alias(body[i], 'ε_{R}', r'\varepsilon_r')

        # Headway observations use direction superscripts throughout all chapters.
        for i in (792, 793, 795, 854, 855, 857):
            for side in ('f', 'b'):
                alias(body[i], 'h_{' + side + '}^{*}', 'h^{' + side + ',*}')
                alias(body[i], 'h_{' + side + '}', 'h^{' + side + '}')
        for i in (793, 855): named_atom(body[i], 'ω', r'\omega_h')
        for i in (795, 857):
            old.append(body[i+1], '其中，', old.math(r'e_f,e_b'), '为两侧计划时距绝对偏差，',
                       old.math(r'\omega_h'), '为前向偏差权重，', old.math(r'R_h'), '为前后计划间隔之比；',
                       old.math(r'I_f,I_b'), '为邻车存在标记，存在取1，否则取0；',
                       old.math(r'p_f,p_b'), '为两侧大偏差惩罚项。')
        # Distinguish unscaled observations from the state-action model input x_k.
        for i in (835, 838): alias(body[i], 'x_{k}', r'x_{\mathrm{obs},k}')
        old.replace_text(body[838], '本章状态归一化采用', '其中，')
        old.append(body[838], old.math(r'x_{\mathrm{obs},k}'), '为归一化前的15维观测，',
                   old.math(r'c_{\mathrm{obs}}'), '为按相同顺序排列的固定尺度向量。本章状态归一化采用')
        wrap_argument(body[901], 'clip_{[−0.999,0.999]}')
        for i in (904, 905): named_atom(body[i], 'B', r'B_{\mathrm{batch}}')
        for i in (906,): alias(body[i], 'B_{i}^{−}', r'B_{\mathrm{SAC},i}^{-}')
        old.append(body[907], old.math(r'B_{\mathrm{SAC},i}^{-}'), '为第', old.math('i'),
                   '条后继样本上的软价值估计；', old.math(r'\bar\ell_{\mathrm{sym}}^+'),
                   '为该批后继状态的平均导数误差，', old.math(r'B_{\mathrm{batch}}'), '为训练小批量大小。')

        for i in (1189, 1190): named_atom(body[i], 'e', r'e_{\mathrm{IQL}}')
        alias(body[1192], 'w_{A}', r'w_{\mathrm{IQL}}')
        for i in range(1480, 1488):
            alias(body[i], 'a_{n}^{2}', r'(a^n)^2')
            alias(body[i], 'a_{n}', r'a^n')
            alias(body[i], 'g_{θ}', r'g_{\mathrm{gate},\theta}')
            named_atom(body[i], 'z', r'\zeta')
        for i in (1475,): alias(body[i], 'd_{k}', r'd_k^{\mathrm{term}}')
        for i in (1526, 1531):
            for acc in body[i].iter(M + 'acc'):
                if re.fullmatch(r'[yQw](?:_\{[^}]+\})?(?:\^\{[^}]+\})?', text(acc.find(M + 'e')) or ''):
                    chr = acc.find(M + 'accPr/' + M + 'chr')
                    if chr is not None and chr.get(M + 'val') in {'̅', '‾'}:
                        chr.set(M + 'val', '̃')
        for script in body[1527].iter():
            if script.tag in {M + 'sSub', M + 'sSubSup'} and text(script.find(M + 'e')) == 'c' and text(script.find(M + 'sub')) == '0':
                sub = script.find(M + 'sub')
                for child in list(sub): sub.remove(child)
                sub.extend(list(old.math(r'\mathrm{const}')))
        for i in (1593,): alias(body[i], 'y_{Δ}', r'y^\Delta')
        alias(body[81], '̃{y}_{Δ}', r'\tilde y^\Delta')
        alias(body[1531], 'ℓ_{Hub}', r'l_{\mathrm{Hub}}')
        old.replace_text(body[1601], '形成加权平均：', '形成加权平均，具体见式(6-14)。')
        for i in (1606, 1608, 1609): named_atom(body[i], 'ξ', r'\xi_{\mathrm{time}}')
        alias(body[622], 'c_{0}', r'c_{\mathrm{shift}}')
        # Align appendix D's local aliases to the actual model in Chapter 4.
        for i in range(2517, 2522):
            alias(body[i], 'l_{sym}', r'\ell_{\mathrm{sym}}')
            alias(body[i], 'Γ_{B^{+}}', r'\bar\ell_{\mathrm{sym}}^+')
            alias(body[i], "V_{i}^{'}", r'B_{\mathrm{SAC},i}^-')
            alias(body[i], "̃{a}_{i}^{'}", r'a_i^{\prime}')
        old.replace_text(body[2518], '第4.3.1节的策略基础目标', '第4.2.3节式(4-17)的策略基础目标')
        batch = body[2519].find('.//' + M + 'oMath')
        assert text(batch) == 'B'
        batch.getparent().remove(batch)

        # Missing definitions introduced by the new model subsections.
        for latex, meaning in [
            (r'x_{\mathrm{obs},k},c_{\mathrm{obs}}', '归一化前的公交观测与各坐标的固定归一化尺度'),
            (r'B_{\mathrm{SAC}},B_{\mathrm{mix}},B_{\mathrm{SAC},i}^-', '后继软价值、与离线状态价值组合后的后继价值及第i条样本的目标软价值'),
            (r'e_{\mathrm{IQL}},w_{\mathrm{IQL}}', 'IQL期望分位回归残差及数据内动作的优势权重'),
            (r'\xi_{\mathrm{time}},I_f,I_b', '运营进度与前后邻车存在标记'),
            (r'\ell_{\mathrm{sym}},\bar\ell_{\mathrm{sym}}^+', '策略与符号参考的归一化导数误差及一批后继状态上的均值'),
            (r'\zeta', '混合驻站策略连续分支的高斯潜变量，与运行模式z区分'),
        ]:
            old.symbol_row(body[81], latex, meaning)
        # Keep this notation-table note together and at the table's 10.5 pt size.
        properties = body[82].find(W + 'pPr')
        if properties.find(W + 'keepLines') is None:
            E.SubElement(properties, W + 'keepLines')
        for run in body[82].iter(W + 'r'):
            pr = run.find(W + 'rPr')
            if pr is None:
                pr = E.Element(W + 'rPr'); run.insert(0, pr)
            for tag in ('sz', 'szCs'):
                size = pr.find(W + tag)
                if size is None: size = E.SubElement(pr, W + tag)
                size.set(W + 'val', '21')

        # These four statistical references lost their prose anchors in restructuring.
        old.append(body[1641], '连续控制基准的设计与比较见文献[137]。')
        old.prepend(body[1651], '强化学习结果会受训练随机性影响，算法比较应明确独立重复和统计单位[138][139][140]。')
        literal_links = citations(body)

        spans = body_spans(raw)
        changed = [i for i,p in enumerate(body) if E.tostring(p) != before[i]]
        data = raw
        journal = []
        for i in reversed(changed):
            start, end = spans[i]
            journal.append({'block': i, 'before': text(E.fromstring(before[i])), 'after': text(body[i])})
            data = data[:start] + E.tostring(body[i], encoding='utf-8', with_tail=False) + data[end:]
        parts = [(deepcopy(info), data if info.filename == 'word/document.xml' else source.read(info.filename)) for info in source.infolist()]
    with ZipFile(OUT, 'w') as dest:
        for info, data in parts:
            dest.writestr(info, data)
    (HERE / '逐处修改记录.json').write_text(json.dumps(list(reversed(journal)), ensure_ascii=False, indent=2))
    print(json.dumps({'output': str(OUT), 'changed_blocks': len(changed), 'literal_citation_links': literal_links}, ensure_ascii=False))


if __name__ == '__main__':
    main()
