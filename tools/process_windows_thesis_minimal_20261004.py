from __future__ import annotations

import copy
import re
import zipfile
from pathlib import Path
from xml.etree import ElementTree as ET


SRC = Path('/home/erzhu419/mine_code/Thesis/output/排版完成_Windows修复.docx')
OUT = Path('/home/erzhu419/mine_code/Thesis/output/排版完成_批注修订_保留原文_单驻站主线.docx')

W_NS = 'http://schemas.openxmlformats.org/wordprocessingml/2006/main'
XML_NS = 'http://www.w3.org/XML/1998/namespace'
NS = {'w': W_NS}
ET.register_namespace('w', W_NS)
ET.register_namespace('r', 'http://schemas.openxmlformats.org/officeDocument/2006/relationships')
ET.register_namespace('mc', 'http://schemas.openxmlformats.org/markup-compatibility/2006')


def q(local: str) -> str:
    return f'{{{W_NS}}}{local}'


def p_text(p: ET.Element) -> str:
    return ''.join((x.text or '') for x in p.findall('.//w:t', NS))


def p_style(p: ET.Element) -> str:
    node = p.find('./w:pPr/w:pStyle', NS)
    return node.attrib.get(q('val'), '') if node is not None else ''


def body_paragraphs(body: ET.Element):
    return [x for x in list(body) if x.tag == q('p')]


def find_p(body: ET.Element, prefix: str, style: str | None = None) -> ET.Element:
    for p in body_paragraphs(body):
        if p_text(p).startswith(prefix) and (style is None or p_style(p) == style):
            return p
    raise KeyError(f'paragraph not found: {prefix!r} style={style!r}')


def accept_revisions(root: ET.Element) -> None:
    def walk(parent: ET.Element) -> None:
        for child in list(parent):
            local = child.tag.rsplit('}', 1)[-1]
            if local == 'del':
                parent.remove(child)
                continue
            if local == 'ins':
                pos = list(parent).index(child)
                parent.remove(child)
                for item in list(child):
                    parent.insert(pos, item)
                    pos += 1
                continue
            walk(child)
    walk(root)


def strip_comment_markers(root: ET.Element) -> None:
    marker_names = {'commentRangeStart', 'commentRangeEnd', 'commentReference'}
    def walk(parent: ET.Element) -> None:
        for child in list(parent):
            if child.tag.rsplit('}', 1)[-1] in marker_names:
                parent.remove(child)
            else:
                walk(child)
    walk(root)


def make_run(text: str) -> ET.Element:
    run = ET.Element(q('r'))
    t = ET.SubElement(run, q('t'))
    if text.startswith(' ') or text.endswith(' ') or '  ' in text:
        t.set(f'{{{XML_NS}}}space', 'preserve')
    t.text = text
    return run


def set_text(p: ET.Element, text: str) -> None:
    keep = []
    for child in list(p):
        local = child.tag.rsplit('}', 1)[-1]
        if local == 'pPr' or local in {'bookmarkStart', 'bookmarkEnd'}:
            keep.append(child)
    p[:] = keep
    p.append(make_run(text))


def clone_body_paragraph(template: ET.Element, text: str) -> ET.Element:
    p = ET.Element(q('p'))
    ppr = template.find('./w:pPr', NS)
    if ppr is not None:
        p.append(copy.deepcopy(ppr))
    p.append(make_run(text))
    return p


def insert_after(body: ET.Element, anchor: ET.Element, paragraphs: list[ET.Element]) -> None:
    pos = list(body).index(anchor) + 1
    for p in paragraphs:
        body.insert(pos, p)
        pos += 1


def replace_visible_text(root: ET.Element, replacements: dict[str, str], skip: set[int] | None = None) -> None:
    skip = skip or set()
    for t in root.findall('.//w:t', NS):
        if id(t) in skip:
            continue
        if not t.text:
            continue
        value = t.text
        for old, new in replacements.items():
            value = value.replace(old, new)
        t.text = value


def replace_across_text_runs(root: ET.Element, replacements: dict[str, str]) -> None:
    """Replace short visible phrases even when Word split them across runs."""
    for p in root.findall('.//w:p', NS):
        text_nodes = p.findall('.//w:t', NS)
        for old, new in replacements.items():
            while True:
                pieces = [x.text or '' for x in text_nodes]
                joined = ''.join(pieces)
                start = joined.find(old)
                if start < 0:
                    break
                end = start + len(old)
                cursor = 0
                first = last = None
                first_offset = last_offset = 0
                for i, piece in enumerate(pieces):
                    next_cursor = cursor + len(piece)
                    if first is None and start < next_cursor:
                        first = i
                        first_offset = start - cursor
                    if end <= next_cursor:
                        last = i
                        last_offset = end - cursor
                        break
                    cursor = next_cursor
                if first is None or last is None:
                    break
                if first == last:
                    pieces[first] = pieces[first][:first_offset] + new + pieces[first][last_offset:]
                else:
                    pieces[first] = pieces[first][:first_offset] + new
                    for i in range(first + 1, last):
                        pieces[i] = ''
                    pieces[last] = pieces[last][last_offset:]
                for node, value in zip(text_nodes, pieces):
                    node.text = value


def preserve_root_namespaces(original: bytes, updated: bytes, local_name: str) -> bytes:
    pattern = rb'<w:' + re.escape(local_name.encode('ascii')) + rb'\b[^>]*>'
    old = re.search(pattern, original)
    new = re.search(pattern, updated)
    if old is None or new is None:
        return updated
    declaration_pattern = rb'\sxmlns(?::[A-Za-z0-9_.-]+)?="[^"]*"'
    old_declarations = re.findall(declaration_pattern, old.group(0))
    new_tag = new.group(0)
    present = set(re.findall(declaration_pattern, new_tag))
    missing = [x for x in old_declarations if x not in present]
    if missing:
        new_tag = new_tag[:-1] + b''.join(missing) + b'>'
    return updated[:new.start()] + new_tag + updated[new.end():]


def clean_package(document: ET.Element, rels: ET.Element, content_types: ET.Element) -> None:
    with zipfile.ZipFile(SRC, 'r') as zin, zipfile.ZipFile(OUT, 'w', zipfile.ZIP_DEFLATED) as zout:
        skip = {
            'word/document.xml', 'word/comments.xml', 'word/commentsExtended.xml',
            'word/commentsExtensible.xml', 'word/commentsIds.xml',
            'word/_rels/document.xml.rels', '[Content_Types].xml',
        }
        header_replacements = {
            '公交在途控制': '公交运行过程控制',
            '在途控制': '运行过程控制',
            '公交在途时长': '公交运行时长',
            '在途时长': '运行时长',
        }
        for info in zin.infolist():
            if info.filename in skip:
                continue
            data = zin.read(info.filename)
            if info.filename.startswith('word/header') or info.filename.startswith('word/footer'):
                try:
                    root = ET.fromstring(data)
                    replace_visible_text(root, header_replacements)
                    updated = ET.tostring(root, encoding='utf-8', xml_declaration=True)
                    data = preserve_root_namespaces(data, updated, root.tag.rsplit('}', 1)[-1])
                except ET.ParseError:
                    pass
            zout.writestr(info, data)
        original = zin.read('word/document.xml')
        updated = ET.tostring(document, encoding='utf-8', xml_declaration=True)
        updated = preserve_root_namespaces(original, updated, 'document')
        zout.writestr('word/document.xml', updated)
        zout.writestr('word/_rels/document.xml.rels', ET.tostring(rels, encoding='utf-8', xml_declaration=True))
        zout.writestr('[Content_Types].xml', ET.tostring(content_types, encoding='utf-8', xml_declaration=True))


def remove_comment_relationships(rels: ET.Element, content_types: ET.Element) -> None:
    for rel in list(rels):
        if 'comments' in rel.attrib.get('Type', '').lower() or 'comments' in rel.attrib.get('Target', '').lower():
            rels.remove(rel)
    for item in list(content_types):
        if 'comments' in item.attrib.get('PartName', '').lower():
            content_types.remove(item)


def main() -> None:
    with zipfile.ZipFile(SRC, 'r') as zin:
        document = ET.fromstring(zin.read('word/document.xml'))
        rels = ET.fromstring(zin.read('word/_rels/document.xml.rels'))
        content_types = ET.fromstring(zin.read('[Content_Types].xml'))

    accept_revisions(document)
    strip_comment_markers(document)
    body = document.find('.//w:body', NS)
    if body is None:
        raise RuntimeError('document body missing')

    # Keep the author's abstract addition verbatim. Only the comment-specific
    # wording around it is adjusted; the sentence about real-world deployment
    # is not replaced by an explanatory paraphrase.
    abstract_first = find_p(body, '随机客流和路况变化使公交车头时距失衡', 'abstractcnfirst')
    set_text(abstract_first,
        '随机客流和路况变化使公交车头时距失衡，增加串车风险和乘客等待时间。附加停站可调节站间时距，但受到车辆与站点差异、环境条件及动作上限的制约。强化学习可依据状态选择动作，却难以在实际运营中反复试错，仿真策略又面临训练与使用环境的分布偏移，故以往强化学习难以部署在现实环境。本文围绕从仿真到现实的驻站控制，先研究预先缺少目标域数据时的平稳随机鲁棒学习与非平稳运行适应，再研究获得目标离线数据后的跨保真训练及模型修正。')
    set_text(find_p(body, '本文以车辆到站为决策时刻', 'abstractcnbody'),
        '本文以车辆完成必要上下客后的到站事件为决策时刻，将车辆与站点信息、前后车头时距、附加停站动作及乘客时间成本纳入统一控制模型。多线路实验沿用现有长沙公交路网、线路、乘客起讫点需求及时刻表资料。自建简易仿真用于低成本训练，城市交通微观仿真平台（Simulation of Urban MObility，SUMO）用于表示更细致的道路交通和车辆相互作用，作为现实部署环境的仿真代理。跨保真迁移采用SUMO离线数据与简易仿真中在线交互产生的新经验训练策略，再于SUMO新运行场景中评价。本文正式主实验统一采用附加停站这一单一控制动作。主要研究内容与结果如下。')

    # Abstract comment resolutions. These are deliberately local and do not
    # rewrite the author's unrelated tracked edits elsewhere in the thesis.
    set_text(find_p(body, '（1）针对不同车辆与站点的差异', 'abstractcnbody'),
        '（1）针对不同车辆与站点的类别差异及随机运行中的动作价值估计波动，提出结合类别嵌入与连续状态的共享停站策略，并提出鲁棒集成软演员评论家方法（Robust Ensemble Soft Actor-Critic，RE-SAC）。本文将广义乘客时间定义为候车时间的两倍与车内时间之和。在与参数和设置都相同的主流算法比较中，学习嵌入相对独热编码在正常运行、客流激增和快速环境切换下，分别降低5.51%、1.41%和4.91%的人均广义乘客时间。单一驻站动作、车头时距目标下，RE-SAC相对主流对照在所测随机环境中具有更低的计划时距偏差和更稳定的价值评价。结果表明，在高不确定性环境中，允许一定估计偏差以换取更稳定的价值评价，可以带来更好的控制效果；不同运营目标仍存在取舍。')
    set_text(find_p(body, '（2）针对运行规律变化与局部控制权限有限的问题', 'abstractcnbody'),
        '（2）针对运行规律随时间变化、而附加停站只能局部改变车辆离站时刻的问题，提出贝叶斯自适应策略正则化（Bayesian Adaptive Policy Regularization，BAPR），通过环境识别和信念调节形成条件适应；进一步提出因果符号贝叶斯自适应策略正则化（Causal-Symbolic Bayesian Adaptive Policy Regularization，CS-BAPR）。在正常需求、单次与重复环境变化、持续高需求及短时极端需求下，训练稳定化方法相对前序BAPR均降低平均时距奖励惩罚，单一驻站动作下的降幅为58.82%至69.79%。与RE-SAC的匹配比较表明，CS-BAPR在运行规律发生变化时具有更强的条件适应能力。本文所称结构化输出，是指由预设算术单元组成的策略输出；该输出并非在每种条件下都带来额外收益。改用乘客时间目标后，服务成本差异缩小，说明时距目标收益不能直接解释为同等幅度的乘客时间收益。')
    set_text(find_p(body, '（3）针对现实环境中难以通过试错持续采集数据的问题', 'abstractcnbody'),
        '（3）针对目标环境难以持续通过试错采集训练数据的问题，在混合离线与在线强化学习（Hybrid Offline-and-Online Reinforcement Learning，H2O）及其扩展H2O+基础上建立公交跨保真训练流程。固定SUMO离线数据与简易仿真交互共同支持策略训练，同时到站车辆编码、允许车辆不额外停站或增加附加停站时长的单一驻站动作，以及运行末段载客较少且后续接续车辆不足时的完成保障，共同构成控制器。在十个新环境块、三档需求和九种控制器的1350次配对评价中，完成保障相对相同学习策略的直接父控制器，使窗口内人均广义乘客时间减少92.12秒、下降3.71%，完成率提高0.466个百分点。同更新预算比较支持混合训练方案相对纯离线方案的收益。')
    set_text(find_p(body, '（4）针对源目标转移关系失配', 'abstractcnbody'),
        '（4）针对源目标转移关系失配，提出基于模型修正的世界模型方法（Model-Corrected World Model，MC-WM），以留出预测误差选择残差修正或直接目标模型，并以对数置信评价加权一步模型样本。公交扩展采用固定目标离线数据与混合事件模型，并以归一化置信备份缓解模型作用衰减。仅驻站、乘客时间目标下，在正常标定、联合50%和联合75%源参数偏差下，相对共同完成保障的Daganzo式规则，人均广义乘客时间分别减少57.67、70.67和76.26秒。这里的联合50%和联合75%源参数偏差，是同时改变道路平均速度、乘客OD、路口延误和基本停站服务时间等转移参数。车头时距目标下，高偏差条件下MC-WM相对H2O+也显示出改善。结果支持利用目标离线数据修正源仿真训练，并显示这一作用可服务于两种运营目标。')

    # Body comments outside the abstract.
    set_text(find_p(body, '仿真中有效的策略不一定能在实际使用环境中取得相同效果', 'body'),
        '仿真中有效的策略不一定能在实际部署环境中取得相同效果。本文将承担训练交互的仿真称为源环境，将策略最终接受运行评价的环境称为目标环境。二者可能在道路速度、客流起讫点分布、路口通行延误和停站服务时间等方面存在差异，使相同状态和动作产生不同的后续运行结果。本文以分布偏移（distribution shift）描述训练与部署时相关概率分布的变化，其中相同状态动作下的后继分布变化称为动力学偏移。动力学随机化研究通过在仿真训练中变化模型参数改善迁移，说明训练分布的构造能够影响现实适应；但这一路径需要确定哪些变化值得纳入训练。[8] 当目标离线数据已经存在时，还可以利用其中关于目标环境的证据，约束或修正源仿真中的学习过程。')
    set_text(find_p(body, '混合离线与在线强化学习（Hybrid Offline-and-Online Reinforcement Learning', 'body'),
        '混合离线与在线强化学习（Hybrid Offline-and-Online Reinforcement Learning，H2O）及其扩展H2O+将目标环境中预先采集的离线数据与训练仿真中新产生的交互经验结合，提供了不同于纯离线学习和目标在线微调的数据利用方式。[9,10] 对公交驻站控制而言，这一思想具有明确吸引力：保留仿真试错的低成本，同时利用目标记录减少学习对源模型的完全依赖。然而，把通用学习框架用于实际公交运营，还需要处理异步事件、零停站动作、运行时域以及未完成服务等具体问题。')
    set_text(find_p(body, '本论文以“从仿真到现实的公交系统驻站控制方法研究”为题', 'body'),
        '本论文以“从仿真到现实的公交系统驻站控制方法研究”为题，以目标环境信息的逐步增加组织研究：缺少预先目标域数据时，依靠仿真中的随机化、保守学习和条件适应，提高策略对潜在变化的适用性；获得目标离线数据后，进一步用这些数据调整低成本源仿真样本在目标转移关系下的权重与可信度，并修正源环境与目标环境的预测关系。自建简易仿真承担低成本训练，SUMO作为更细致运行环境的仿真代理，连接算法研究与面向实际运营的迁移目标。')
    set_text(find_p(body, '全文按环境假设与目标数据可用性递进', 'body'),
        '全文按环境假设与目标数据可用性递进。第3章从平稳随机环境出发研究鲁棒强化学习，第4章在保留随机波动的基础上增加运行环境规律随时间变化的假设；这两章主要利用源环境交互得到的数据，研究目标环境数据不可得时的控制问题。第5章引入目标环境离线数据，利用这些信息约束在源环境中训练得到的策略；第6章进一步把源目标差异纳入预测模型及其使用方式。四章共同服务于仿真到现实迁移，分别回答如何面对随机性、如何识别变化、如何利用目标经验以及如何修正模型。以下先给出共用定义，后续各章重点展开相对于这些定义的新问题、新假设和算法。')

    # Remove the old term from visible prose, while leaving bookmark names and
    # internal field targets untouched.
    replace_visible_text(document, {
        '公交在途控制': '公交运行过程控制',
        '在途控制': '运行过程控制',
        '公交在途时长': '公交运行时长',
        '在途时长': '运行时长',
        '面向有限使用环境数据': '面向有限目标环境数据',
        '面向训练与使用环境差异': '面向训练与部署环境差异',
        '训练与使用环境': '训练与部署环境',
        '目标域使用': '目标域部署',
        '待使用的环境': '待部署的环境',
        '本章的实际使用环境': '本章的实际部署环境',
        '目标数据提供使用环境': '目标数据提供目标环境',
        '训练环境与使用环境': '训练环境与部署环境',
        '难以持续在使用环境中': '难以持续在部署环境中',
        '对真实转移使用环境诊断': '对真实转移的环境诊断',
        '实际使用环境': '实际部署环境',
        '使用环境': '部署环境',
    })
    replace_across_text_runs(document, {
        '面向有限使用环境数据': '面向有限目标环境数据',
        '面向训练与使用环境差异': '面向训练与部署环境差异',
        '训练与使用环境': '训练与部署环境',
        '目标域使用': '目标域部署',
        '本章的实际使用环境': '本章的实际部署环境',
        '目标数据提供使用环境': '目标数据提供目标环境',
        '训练环境与使用环境': '训练环境与部署环境',
        '难以持续在使用环境中': '难以持续在部署环境中',
        '对真实转移使用环境诊断': '对真实转移的环境诊断',
        '实际使用环境': '实际部署环境',
        '使用环境': '部署环境',
    })

    # The author's abstract sentence is intentionally restored after the
    # terminology pass because its wording is an explicit tracked addition.
    set_text(abstract_first,
        '随机客流和路况变化使公交车头时距失衡，增加串车风险和乘客等待时间。附加停站可调节站间时距，但受到车辆与站点差异、环境条件及动作上限的制约。强化学习可依据状态选择动作，却难以在实际运营中反复试错，仿真策略又面临训练与使用环境的分布偏移，故以往强化学习难以部署在现实环境。本文围绕从仿真到现实的驻站控制，先研究预先缺少目标域数据时的平稳随机鲁棒学习与非平稳运行适应，再研究获得目标离线数据后的跨保真训练及模型修正。')

    # Chapter-opening introductions requested separately from comment handling.
    template = find_p(body, '公交车队驻站控制的基本任务', 'body')
    openings = {
        '第3章 平稳随机环境下基于单智能体鲁棒强化学习的公交车队驻站': [
            '第2章给出了公交事件决策、驻站动作和评价目标的共同定义。本章从目标环境数据尚不可得的条件出发，研究在平稳随机运行规律下如何学习可跨车辆复用、并能承受客流与路况随机波动的驻站策略。',
            '本章沿“问题—模型—算法—结果分析”展开：先说明平稳随机假设和单智能体共享决策的适用边界，再建立类别嵌入、事件触发动作和计划对齐目标，随后提出RE-SAC并以单一驻站动作完成表征、价值评价和服务后果的对照。结果重点回答随机环境中的价值评价稳定性，而不把局部时距改善直接等同于乘客时间改善。',
        ],
        '第4章 非平稳环境下基于单智能体鲁棒强化学习的公交车队驻站控制': [
            '第3章假定给定运行条件下的随机规律保持不变，本章进一步考虑客流和道路条件随时间漂移、突变或重复变化的情形。目标环境数据仍不在训练前直接提供，控制器只能根据运行历史逐步判断当前条件，因而问题从平稳随机鲁棒学习推进到非平稳条件适应。',
            '本章依次讨论隐含运行模式、信念调节和CS-BAPR的训练稳定化与策略结构，并在统一的单一驻站动作接口下比较正常需求、环境突变和重复变化。结果同时报告时距目标与乘客时间目标，区分方法对非平稳时距控制的收益、输出结构的附加作用及其服务指标边界。',
        ],
        '第5章 跨保真运行环境下基于迁移强化学习的公交车队驻站控制': [
            '第3、4章研究的是目标环境数据尚不可得时的源仿真学习。本章增加固定目标离线数据这一信息条件，但仍限制目标环境中的持续试错，将研究重点转向目标经验与源仿真交互如何在同一驻站控制接口下协同。',
            '本章先区分纯离线学习、目标在线微调和跨保真续训的数据条件，再建立H2O+公交流程，分析离线初始化、源环境续训、样本加权及完成保障的作用。正式结果以SUMO离线数据与简易仿真交互为训练基础，以新的SUMO运行场景为评价环境，重点回答固定目标经验能否支持低成本源仿真训练，以及执行保障带来的服务收益是否应与学习算法收益分开归因。',
        ],
        '第6章 分布偏移下基于世界模型的公交车队驻站控制': [
            '第5章在经验与价值更新层面结合目标离线数据和源仿真交互，本章进一步把目标数据用于学习和修正源环境的转移关系。当源仿真中的道路速度、客流到达、路口延误和停站服务时间存在标定偏差时，问题不再只是如何混合两类经验，还包括哪些源预测值得保留，以及预测信息应以多大强度参与策略学习。',
            '本章提出MC-WM，以留出目标数据在残差模型和直接目标模型之间作选择，再以归一化置信备份调节模型样本对价值更新的影响。受控任务用于解释模型类别选择，公交任务在固定SUMO离线数据、简易仿真交互和单一驻站动作下检验模型辅助对乘客服务的作用，并通过多训练种子和消融比较区分模型拟合、模型使用强度与实际控制收益。',
        ],
    }
    for heading, paragraphs in openings.items():
        insert_after(body, find_p(body, heading, 'chapter'), [clone_body_paragraph(template, x) for x in paragraphs])

    remove_comment_relationships(rels, content_types)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    clean_package(document, rels, content_types)
    print(OUT)


if __name__ == '__main__':
    main()
