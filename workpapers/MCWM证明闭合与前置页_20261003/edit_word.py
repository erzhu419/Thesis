"""Personalize template front matter and synchronize the closed Appendix E proofs."""
from copy import deepcopy
import json
from pathlib import Path
import re
import sys
from zipfile import ZipFile, ZIP_DEFLATED

from lxml import etree as E

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / 'workpapers/全文深度校审_20261003'))
from extract_word import NS, content
from revise_word import q, run, mr, sub
sys.path.insert(0, str(ROOT / 'workpapers/模板迁移_20261002'))
from verify_format import Doc

BASE = ROOT / 'output/排版完成.docx'
TEMPLATE = next(ROOT.glob('中南大学博士学位论文_gty*docx'))


def properties(tag, data):
    node = E.Element(q('w:' + tag))
    for key, value in data.items():
        if key in ('pStyle', 'rStyle', 'sectPr'):
            continue
        for attrs in value if isinstance(value, list) else [value]:
            child = E.SubElement(node, q('w:' + key))
            for k, v in attrs.items():
                if k != 'children':
                    child.set(q('w:' + k), str(v))
            for c in attrs.get('children', []):
                n = E.SubElement(child, q('w:' + c['tag']))
                for k, v in c.items():
                    if k != 'tag':
                        n.set(q('w:' + k), str(v))
    return node


def fonts_from_source(source, rp):
    rp = deepcopy(rp)
    hint = rp.get('rFonts', {}).get('hint')
    rp['rFonts'] = {k: v for k, v in source.fonts(rp).items() if v}
    if hint:
        rp['rFonts']['hint'] = hint
    rp.setdefault('b', {'val': '0'})
    rp.setdefault('i', {'val': '0'})
    return rp


def template_front(source, style_id='ThesisFrontMatter'):
    with ZipFile(TEMPLATE) as z:
        original = E.fromstring(z.read('word/document.xml')).find('w:body', NS)
    front = [deepcopy(n) for n in list(original)[:62]]
    source_ps = []
    for node in list(source.body)[:62]:
        source_ps.extend([node] if node.tag.endswith('}p') else node.findall('.//w:p', NS))
    copied_ps = []
    for node in front:
        copied_ps.extend([node] if node.tag == q('w:p') else node.findall('.//w:p', NS))
    assert len(source_ps) == len(copied_ps)
    for original_p, p in zip(source_ps, copied_ps):
        _, pp, rp, _, _ = source.resolve(original_p)
        pp = deepcopy(pp)
        pp.setdefault('outlineLvl', {'val': '9'})
        pp.setdefault('keepNext', {'val': '0'})
        pp.setdefault('keepLines', {'val': '0'})
        pp.setdefault('contextualSpacing', {'val': '0'})
        pp.setdefault('spacing', {}).setdefault('before', '0')
        pp['spacing'].setdefault('after', '0')
        pr = p.find('w:pPr', NS)
        if pr is not None:
            p.remove(pr)
        pr = properties('pPr', pp)
        sid = E.Element(q('w:pStyle'))
        sid.set(q('w:val'), style_id)
        pr.insert(0, sid)
        from template_analysis import props
        from verify_format import blend
        mark = blend(rp, props(original_p.find(q('w:pPr') + '/' + q('w:rPr'))))
        pr.append(properties('rPr', fonts_from_source(source, mark)))
        p.insert(0, pr)
        source_runs = list(original_p.iter(q('w:r')))
        copied_runs = list(p.iter(q('w:r')))
        assert len(source_runs) == len(copied_runs)
        # Preserve the actual template font cascade, not similarly named styles.
        from template_analysis import props
        from verify_format import blend
        for old_r, r in zip(source_runs, copied_runs):
            raw = props(old_r.find(q('w:rPr')))
            effective = blend(source.apply(deepcopy(rp), raw.get('rStyle', {}).get('val'), 'rPr'), raw)
            old_pr = r.find('w:rPr', NS)
            if old_pr is not None:
                r.remove(old_pr)
            r.insert(0, properties('rPr', fonts_from_source(source, effective)))
    for node in front:
        for mark in node.xpath('.//w:bookmarkStart | .//w:bookmarkEnd | .//w:lastRenderedPageBreak', namespaces=NS):
            mark.getparent().remove(mark)
        for style in node.xpath('.//w:tblStyle', namespaces=NS):
            style.getparent().remove(style)
        assert not node.xpath('.//@r:embed | .//@r:id', namespaces=NS)
    return front


def set_text(p, text):
    prototype = next((r for r in p.findall('w:r', NS) if r.find('w:t', NS) is not None), None)
    prototype = deepcopy(prototype) if prototype is not None else run('')
    for n in list(p):
        if n.tag != q('w:pPr'):
            p.remove(n)
    for n in list(prototype):
        if n.tag != q('w:rPr'):
            prototype.remove(n)
    t = E.SubElement(prototype, q('w:t'))
    t.set('{http://www.w3.org/XML/1998/namespace}space', 'preserve')
    t.text = text
    p.append(prototype)


def prose(p, text, math_symbols=None):
    spans = {}
    field, depth = [], 0
    for n in p:
        begin = n.xpath('.//w:fldChar[@w:fldCharType="begin"]', namespaces=NS)
        end = n.xpath('.//w:fldChar[@w:fldCharType="end"]', namespaces=NS)
        if begin:
            depth += len(begin)
        if depth:
            field.append(deepcopy(n))
        if end:
            depth -= len(end)
            if depth == 0:
                label = ''.join(content(v) for v in field)
                if label in text:
                    spans[label] = field
                field = []
    assert depth == 0
    for n in p.findall('w:hyperlink', NS):
        spans[content(n)] = [deepcopy(n)]
    assert all(label in text for label in spans)
    for n in list(p):
        if n.tag not in (q('w:pPr'), q('w:bookmarkStart'), q('w:bookmarkEnd')):
            p.remove(n)
    pattern = r'(\{\{[^{}]+\}\}' + ''.join('|' + re.escape(label) for label in spans) + ')'
    for part in re.split(pattern, text):
        if part in spans:
            p.extend(deepcopy(spans[part]))
        elif part.startswith('{{'):
            token = part[2:-2]
            if math_symbols and token in math_symbols:
                p.append(deepcopy(math_symbols[token]))
            else:
                n = E.SubElement(p, q('m:oMath'))
                n.append(sub(*token.split('_', 1)) if '_' in token else mr(token))
        elif part:
            if part.startswith('证明。'):
                bold = run('证明。')
                pr = E.Element(q('w:rPr'))
                E.SubElement(pr, q('w:b'))
                bold.insert(0, pr)
                p.append(bold)
                part = part[3:]
            p.append(run(part))


def main():
    with ZipFile(BASE) as z:
        parts = {n: z.read(n) for n in z.namelist()}
    with ZipFile(HERE / '排版完成_修改前.docx') as z:
        assert parts['word/document.xml'] == z.read('word/document.xml'), 'Reread changed Word first.'
    doc = E.fromstring(parts['word/document.xml'])
    body = doc.find('w:body', NS)
    blocks = list(body)
    assert content(blocks[0]).replace('\n', '') == '从仿真到现实的公交系统驻站控制方法研究'
    source = Doc(TEMPLATE)
    front = template_front(source)
    for i in (6, 27):
        set_text(front[i], '从仿真到现实的')
    for i in (7, 28):
        set_text(front[i], '公交系统驻站控制方法研究')
    for i in (8, 29):
        set_text(front[i], 'Research on Bus System Holding Control Methods')
    for i in (9, 30):
        set_text(front[i], 'from Simulation to Reality')
    for i in (19, 42):
        set_text(front[i], '    年    月')
    set_text(front[21], '中图分类号                 学校代码  10533')
    set_text(front[22], 'UDC                        学位类别  学术学位')
    set_text(front[37], '论文答辩日期     年   月   日    答辩委员会主席              ')
    for table_index in (13, 33):
        for row in front[table_index].findall('w:tr', NS):
            cells = row.findall('w:tc', NS)
            label = re.sub(r'\s+', '', content(cells[0]))
            replacement = {
                '一级学科': '交通运输工程', '二级学科': '交通运输规划与管理',
                '作者姓名': '张异凡', '指导教师': '郑亮',
                '研究方向': '', '二级培养单位': '交通运输工程学院',
            }[label]
            set_text(cells[1].find('w:p', NS), replacement)
    body.remove(blocks[0])
    body.remove(blocks[1])
    for i, n in enumerate(front):
        body.insert(i, n)
    # Keep the existing unnumbered cover section and all subsequent sections.
    styles = E.fromstring(parts['word/styles.xml'])
    style = E.SubElement(styles, q('w:style'))
    style.set(q('w:type'), 'paragraph')
    style.set(q('w:styleId'), 'ThesisFrontMatter')
    E.SubElement(style, q('w:name')).set(q('w:val'), '学位论文前置页')
    style.append(properties('rPr', fonts_from_source(source, source.apply(source.dr, source.default, 'rPr'))))
    edits = {
        1858: '残差模型上的价值评价因而具有固定算子的收缩性质。有限状态空间在最大范数下完备，故固定策略对应唯一价值固定点，且从任意初值出发的评价迭代均收敛。附录E.1和E.2分别证明随机核合法性、固定点存在与唯一性及迭代误差界；附录E.3分析该固定点与目标价值之间的差异。模型拟合的更新过程是另一映射，其固定点结论以附录E.5的完整映射收缩条件为前提。',
        1868: '代入并将 {{γ}}{{ϵ_V}} 移至左端，即得所述上界。固定合法策略将状态动作转移核加权为状态转移核：非负性与行和为1均保持，统一的奖励误差界和总变差界也不会因这一加权增大。封闭范围条件则保证两种模型的后继状态仍处于施加误差界的区域。引理E.2给出总变差与期望差的联系，定理E.2据此证明相同奖励下的价值误差界；附录E.3.3加入奖励误差得到本节的完整形式，E.3.4与E.4的具体反例说明为何局部预测准确不能推出全局价值准确。',
        3003: '固定策略下，先对动作概率求和，得到状态转移核与期望即时奖励。策略和转移核均非负且归一化，因此该状态核也非负且每行和为1，期望奖励仍受式(E-2)约束。价值函数空间在最大范数下完备；由定理E.1及 {{γ}}<1，收缩映射不动点定理给出唯一固定点 {{fixedvalue}}。从任意初值 {{initialvalue}} 出发，令 {{valueiteration}}。连续两次迭代的差由折扣因子的幂控制，后续差值之和可用收敛几何级数界定，故迭代构成柯西序列。其极限满足固定点方程；再将每次迭代与固定点比较，得到',
        3028: '于是正数1−{{γ}}乘以价值误差，不超过式(E-20)中不含价值误差的项，除以1−{{γ}}>0即得式(E-18)。固定策略对各动作的非负加权不会扩大式(E-17)中的统一总变差界，因而上述状态核论证同样适用于原状态动作核。证毕。',
        3047: '证明。 价值误差非负，并由式(E-23)中两个非负项之和控制。固定折扣条件下，两项的系数均为有限常数；当奖励误差和转移误差分别趋于零时，上界趋于零。由夹逼定理得到式(E-25)。证毕。',
        3068: '证明。 以有限索引集上的实值函数表示残差，最大范数使该空间完备。式(E-29)给出完整映射的收缩常数，小于1，故从任意初值迭代得到柯西序列。收缩映射连续，迭代极限满足固定点方程。若有两个固定点，式(E-29)会使它们的距离不超过该距离与小于1的常数之积，因此距离为零，固定点唯一。最后，将第n次迭代与固定点比较并反复使用式(E-29)，得到式(E-30)。证毕。',
        3082: '这些等式是以下排序定理的明确前提。对同方差高斯密度，在等类别先验下，两类后验的几率等于密度比；共同的密度归一化项相消，取对数即可得到式(E-33)，其中 {{τ_rank}}为两倍方差。式(E-32)则来自所指定的指数型误差信号。以下证明使用这两条等式及正值条件，不将任意训练所得分类器或置信评价器等同于上述理想关系。',
        3124: '原算子同理；行和小于1只会减弱后继项的传播，不影响这一收缩界。两个算子均作用于有限维完备空间，因此分别存在唯一固定点。将原固定点等式两侧乘以1−{{γ_Δ}}，并利用有限概率加权对评价函数的线性性，得到',
    }
    changes = []
    for i, text in edits.items():
        before = content(blocks[i])
        assert blocks[i].tag == q('w:p') and before, (i, before)
        math_symbols = None
        if i == 3003:
            original = blocks[i].xpath('.//m:oMath', namespaces=NS)
            math_symbols = dict(zip(['fixedvalue', 'initialvalue', 'valueiteration'], original[1:4]))
        prose(blocks[i], text, math_symbols)
        changes.append({'old_block': i, 'old': before, 'new': text})
    assert len(doc.xpath('.//w:sectPr', namespaces=NS)) == 16
    assert not any(word in ''.join(content(n) for n in front)
                   for word in ['高天宇', '王中钢', '2025', 'U270.2', '629', '周丹', 'TPMS'])
    parts['word/document.xml'] = E.tostring(doc, xml_declaration=True, encoding='UTF-8', standalone=True)
    parts['word/styles.xml'] = E.tostring(styles, xml_declaration=True, encoding='UTF-8', standalone=True)
    with ZipFile(BASE, 'w', ZIP_DEFLATED) as z:
        for name, data in parts.items():
            z.writestr(name, data)
    (HERE / 'word_changes.json').write_text(json.dumps(changes, ensure_ascii=False, indent=2))
    print(json.dumps({'front_pages': 3, 'prose_changes': len(changes), 'sections': 16}, ensure_ascii=False))


if __name__ == '__main__':
    main()
