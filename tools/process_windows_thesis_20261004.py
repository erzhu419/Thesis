from __future__ import annotations

import copy
import re
import shutil
import zipfile
from pathlib import Path
from xml.etree import ElementTree as ET


SRC = Path('/home/erzhu419/mine_code/Thesis/output/排版完成_Windows修复.docx')
OUT = Path('/home/erzhu419/mine_code/Thesis/output/排版完成_批注修订接受_单驻站主线.docx')

W_NS = 'http://schemas.openxmlformats.org/wordprocessingml/2006/main'
XML_NS = 'http://www.w3.org/XML/1998/namespace'
REL_NS = 'http://schemas.openxmlformats.org/package/2006/relationships'
CT_NS = 'http://schemas.openxmlformats.org/package/2006/content-types'
NS = {'w': W_NS}
ET.register_namespace('w', W_NS)
ET.register_namespace('r', 'http://schemas.openxmlformats.org/officeDocument/2006/relationships')
ET.register_namespace('mc', 'http://schemas.openxmlformats.org/markup-compatibility/2006')


def q(local: str) -> str:
    return f'{{{W_NS}}}{local}'


def text_of(p: ET.Element) -> str:
    return ''.join((t.text or '') for t in p.findall('.//w:t', NS)) + ''.join(
        (t.text or '') for t in p.findall('.//w:delText', NS)
    )


def style_of(p: ET.Element) -> str:
    node = p.find('./w:pPr/w:pStyle', NS)
    return node.attrib.get(q('val'), '') if node is not None else ''


def accept_revisions(root: ET.Element) -> None:
    """Keep inserted content and discard deleted content in all XML descendants."""
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
    names = {'commentRangeStart', 'commentRangeEnd', 'commentReference'}

    def walk(parent: ET.Element) -> None:
        for child in list(parent):
            if child.tag.rsplit('}', 1)[-1] in names:
                parent.remove(child)
            else:
                walk(child)

    walk(root)


def make_run(text: str) -> ET.Element:
    r = ET.Element(q('r'))
    t = ET.SubElement(r, q('t'))
    if text.startswith(' ') or text.endswith(' ') or '  ' in text:
        t.set(f'{{{XML_NS}}}space', 'preserve')
    t.text = text
    return r


def set_paragraph_text(p: ET.Element, text: str) -> None:
    """Replace ordinary prose while retaining paragraph formatting and bookmarks."""
    keep = []
    for child in list(p):
        local = child.tag.rsplit('}', 1)[-1]
        if local == 'pPr' or local in {'bookmarkStart', 'bookmarkEnd'}:
            keep.append(child)
    p[:] = keep
    p.append(make_run(text))


def replace_t_text(root: ET.Element, replacements: dict[str, str]) -> None:
    for t in root.findall('.//w:t', NS):
        if t.text:
            value = t.text
            for old, new in replacements.items():
                value = value.replace(old, new)
            t.text = value


def all_body_paragraphs(body: ET.Element):
    return [x for x in list(body) if x.tag == q('p')]


def find_paragraph(body: ET.Element, prefix: str, style: str | None = None) -> ET.Element:
    candidates = []
    for p in all_body_paragraphs(body):
        if text_of(p).startswith(prefix) and (style is None or style_of(p) == style):
            candidates.append(p)
    if not candidates:
        raise KeyError(f'paragraph not found: {prefix!r} style={style!r}')
    return candidates[0]


def find_heading_index(body: ET.Element, prefix: str, style: str) -> int:
    for i, child in enumerate(list(body)):
        if child.tag == q('p') and style_of(child) == style and text_of(child).startswith(prefix):
            return i
    raise KeyError(f'heading not found: {prefix!r}')


def remove_section(body: ET.Element, start_prefix: str, next_h2_prefix: str) -> int:
    start = find_heading_index(body, start_prefix, 'h3')
    end = find_heading_index(body, next_h2_prefix, 'h2')
    children = list(body)[start:end]
    for child in children:
        body.remove(child)
    return len(children)


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


def set_by_prefix(body: ET.Element, prefix: str, text: str, style: str | None = None) -> None:
    set_paragraph_text(find_paragraph(body, prefix, style), text)


def remove_toc_entry(body: ET.Element, prefix: str) -> None:
    for p in list(body):
        if p.tag == q('p') and style_of(p).startswith('TOC') and text_of(p).startswith(prefix):
            body.remove(p)


def replace_visible_text(root: ET.Element, replacements: dict[str, str]) -> None:
    for t in root.findall('.//w:t', NS):
        if t.text:
            value = t.text
            for old, new in replacements.items():
                value = value.replace(old, new)
            t.text = value


def preserve_root_start_tag(original: bytes, updated: bytes, local_name: str) -> bytes:
    """Merge original and generated namespace declarations after XML edits."""
    pattern = rb'<w:' + re.escape(local_name.encode('ascii')) + rb'\b[^>]*>'
    original_match = re.search(pattern, original)
    updated_match = re.search(pattern, updated)
    if original_match is None or updated_match is None:
        return updated
    declaration_pattern = rb'\sxmlns(?::[A-Za-z0-9_.-]+)?="[^"]*"'
    original_declarations = re.findall(declaration_pattern, original_match.group(0))
    updated_tag = updated_match.group(0)
    updated_declarations = set(re.findall(declaration_pattern, updated_tag))
    missing = [item for item in original_declarations if item not in updated_declarations]
    if missing:
        updated_tag = updated_tag[:-1] + b''.join(missing) + b'>'
    return updated[:updated_match.start()] + updated_tag + updated[updated_match.end():]


def clean_package(src: Path, out: Path, document: ET.Element, rels: ET.Element, content_types: ET.Element) -> None:
    header_replacements = {
        '公交在途控制': '公交运行过程控制',
        '在途控制': '公交运行过程控制',
        '公交在途时长': '公交运行时长',
        '在途时长': '运行时长',
    }
    with zipfile.ZipFile(src, 'r') as zin, zipfile.ZipFile(out, 'w', zipfile.ZIP_DEFLATED) as zout:
        for item in zin.infolist():
            name = item.filename
            if name in {
                'word/document.xml',
                'word/comments.xml',
                'word/commentsExtended.xml',
                'word/commentsExtensible.xml',
                'word/commentsIds.xml',
                'word/_rels/document.xml.rels',
                '[Content_Types].xml',
            }:
                continue
            data = zin.read(name)
            if name.startswith('word/header') or name.startswith('word/footer'):
                try:
                    header_root = ET.fromstring(data)
                    replace_visible_text(header_root, header_replacements)
                    updated = ET.tostring(header_root, encoding='utf-8', xml_declaration=True)
                    root_name = header_root.tag.rsplit('}', 1)[-1]
                    data = preserve_root_start_tag(data, updated, root_name)
                except ET.ParseError:
                    pass
            zout.writestr(item, data)
        original_document = zin.read('word/document.xml')
        updated_document = ET.tostring(document, encoding='utf-8', xml_declaration=True)
        updated_document = preserve_root_start_tag(original_document, updated_document, 'document')
        zout.writestr('word/document.xml', updated_document)
        zout.writestr('word/_rels/document.xml.rels', ET.tostring(rels, encoding='utf-8', xml_declaration=True))
        zout.writestr('[Content_Types].xml', ET.tostring(content_types, encoding='utf-8', xml_declaration=True))


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

    # The revised abstract is rewritten as a coherent unit so that accepted
    # track changes cannot leave fragments such as the deleted wording behind.
    set_by_prefix(body, '随机客流和路况变化使公交车头时距失衡',
        '随机客流和路况变化会使公交车头时距失衡，增加串车风险和乘客等待时间。附加停站可以调节站间时距，但其作用受车辆、站点、运行条件和动作上限约束。强化学习能够依据运行状态选择控制动作，却不能在实际车队中反复试错；仿真训练得到的策略还可能面对与训练不同的部署动力学。本文围绕从仿真到现实的公交驻站控制，先研究目标环境数据不可得时的平稳随机鲁棒学习与非平稳条件适应，再研究获得目标离线数据后的跨保真训练和模型修正。', 'abstractcnfirst')
    set_by_prefix(body, '本文以车辆到站为决策时刻',
        '本文以车辆完成必要上下客后的到站事件为决策时刻，将车辆与站点信息、前后车头时距、附加停站动作及乘客时间成本纳入统一控制模型。多线路实验沿用现有长沙公交路网、线路、乘客起讫点需求及时刻表资料。自建简易仿真用于低成本训练，城市交通微观仿真平台（Simulation of Urban MObility，SUMO）用于表示更细致的道路交通和车辆相互作用，作为目标环境的仿真代理。跨保真实验使用SUMO离线数据和简易仿真交互训练策略，再在SUMO新运行场景中评价。本文实验统一采用附加停站这一单一控制动作，速度倍率固定为1。主要研究内容如下。', 'abstractcnbody')
    set_by_prefix(body, '（1）针对不同车辆与站点的差异',
        '（1）针对不同车辆、站点和运行阶段的类别差异，以及随机客流和路况导致的动作价值估计波动，提出结合类别嵌入与连续状态的共享驻站策略，并提出鲁棒集成软演员评论家方法（Robust Ensemble Soft Actor-Critic，RE-SAC）。在保持训练预算和网络规模一致的主流算法对照中，学习嵌入相对独热编码在正常运行、客流激增和快速环境切换下，分别降低5.51%、1.41%和4.91%的人均广义乘客时间。本文将广义乘客时间定义为候车时间的两倍与车内时间之和，用于同时反映站外等待和车内停留。单一驻站动作、车头时距目标下，RE-SAC在计划时距偏差和价值评价稳定性方面优于相应主流对照；在高波动条件下，允许一定价值偏差以换取更稳定的评价，有利于保持控制效果。', 'abstractcnbody')
    set_by_prefix(body, '（2）针对运行规律变化与局部控制权限有限的问题',
        '（2）针对运行规律随时间变化且动作只能局部改变车辆离站时刻的问题，提出贝叶斯自适应策略正则化（Bayesian Adaptive Policy Regularization，BAPR），通过环境识别和信念调节形成条件适应；进一步提出因果符号贝叶斯自适应策略正则化（Causal-Symbolic Bayesian Adaptive Policy Regularization，CS-BAPR）。在正常需求、单次和重复环境变化、持续高需求及短时极端需求条件下，训练稳定化方法相对前序BAPR降低平均时距奖励惩罚；单一驻站动作下的降幅为58.82%至69.79%。与RE-SAC的同一多线路对照显示，CS-BAPR在需求规律发生变化时具有更强的条件适应能力；带因果符号约束的策略输出并非在每种条件下都带来额外收益。改用乘客时间目标后，服务成本差异缩小，说明时距目标的改善不能直接等同于乘客时间的同比改善。', 'abstractcnbody')
    set_by_prefix(body, '（3）针对现实环境中难以通过试错持续采集数据的问题',
        '（3）针对目标环境难以持续通过试错采集训练数据的问题，在混合离线与在线强化学习（Hybrid Offline-and-Online Reinforcement Learning，H2O）及其扩展H2O+基础上建立公交跨保真训练流程。固定SUMO离线数据与简易仿真交互共同支持策略训练；同时到站车辆编码、允许车辆不额外停站或执行正的附加停站时长的动作模型，以及运行末段低载且缺少后续接续车辆时的完成保障，共同构成控制器。在十个新环境块、三档需求和九种控制器的1350次配对评价中，完成保障相对相同学习策略的直接父控制器，使窗口内人均广义时间减少92.12秒、下降3.71%，完成率提高0.466个百分点。同更新预算比较支持混合训练方案相对所测纯离线方案的收益。', 'abstractcnbody')
    set_by_prefix(body, '（4）针对源目标转移关系失配',
        '（4）针对源仿真与目标环境转移规律失配，提出基于模型修正的世界模型方法（Model-Corrected World Model，MC-WM），以留出预测误差选择残差修正或直接目标模型，并以对数置信评价加权一步模型样本。公交扩展采用固定目标离线数据与混合事件模型，并以归一化置信备份缓解模型作用衰减。在单一驻站、乘客时间目标下，相对采用相同完成保障的Daganzo式规则，人均广义时间在正常标定、联合50%和联合75%源参数偏差下分别减少57.67、70.67和76.26秒。这里的联合偏差同时改变道路平均速度、客流OD、路口延误和基本停站服务时间等转移参数。结果支持利用目标离线数据修正源仿真训练，并显示模型辅助收益在源环境标定不足时更明显。', 'abstractcnbody')
    set_by_prefix(body, '本文以目标环境信息的可获得条件及其利用方式',
        '本文以目标环境信息的可获得条件及其利用方式，串联平稳随机鲁棒控制、非平稳条件适应、跨保真迁移和世界模型修正，形成从仿真到现实的公交驻站控制研究路径。', 'abstractcnbody')

    # Keep the English abstract semantically aligned with the Chinese abstract.
    set_by_prefix(body, 'Random passenger demand and changing road traffic disrupt bus headways',
        'Random passenger demand and changing road traffic disrupt bus headways, promote bunching, and increase passenger waiting. Holding control regulates service by delaying departure from a stop, but its effectiveness depends on vehicle and stop characteristics, operating conditions, and the maximum permitted holding time. Reinforcement learning can select actions from operational observations, yet repeated trial-and-error learning on an operating fleet is impractical, and a policy trained in simulation may face different deployment dynamics. This thesis studies sim-to-real bus holding control as target-domain information becomes available: robust learning under stationary stochastic operations and adaptation to non-stationary conditions first address settings without pre-collected target-domain training data; cross-fidelity learning and world-model correction then incorporate target offline data.', 'abstractenfirst')
    set_by_prefix(body, 'An arrival-event-based bus control model connects vehicle and stop information',
        'An arrival-event-based bus control model connects vehicle and stop information, forward and backward headways, holding actions, and passenger time costs. Multi-line experiments use an existing Changsha road network, bus routes, origin-destination demand, and timetables. An in-house simplified simulator captures the main vehicle and passenger-service relationships for low-cost training. Simulation of Urban MObility (SUMO) additionally represents road traffic and vehicle interactions and serves as a simulated proxy for the deployment environment. Cross-fidelity transfer here means using previously collected SUMO data, continuing training in the simplified simulator, and evaluating the resulting policy in new SUMO operating scenarios. The main experiments use holding as the single control action; speed control is outside the main thesis comparisons.', 'abstractenbody')
    set_by_prefix(body, 'This thesis proposes a shared holding policy combining categorical embeddings',
        'This thesis proposes a shared holding policy combining categorical embeddings with continuous observations and develops Robust Ensemble Soft Actor-Critic (RE-SAC) for conservative action-value learning. With a common learner, learned embeddings reduce generalized passenger time relative to one-hot encoding by 5.51%, 1.41%, and 4.91% under routine operations, demand surges, and rapid environmental changes. Generalized passenger time is defined as twice waiting time plus onboard time. Under the single holding action and a headway objective, RE-SAC improves planned-headway deviation and evaluation stability relative to the matched mainstream baseline. The results indicate that accepting a controlled amount of value bias can improve stability when operating uncertainty is high.', 'abstractenbody')
    set_by_prefix(body, 'To address changing operating laws and bounded control authority',
        'To address changing operating conditions and the limited ability of holding to change a vehicle departure time, this thesis proposes Bayesian Adaptive Policy Regularization (BAPR), using environment identification and belief-based adjustment, and subsequently develops Causal-Symbolic Bayesian Adaptive Policy Regularization (CS-BAPR). Across routine demand, single and repeated environmental changes, sustained high demand, and a brief extreme demand condition, the stabilized method reduces the mean headway-reward penalty relative to BAPR by 58.82% to 69.79% for the single holding action. A symbol-constrained policy output does not provide an additional benefit in every condition, and switching to a passenger-time objective narrows the service-cost difference. A matched comparison with RE-SAC shows the value of conditional adaptation when operating laws change.', 'abstractenbody')
    set_by_prefix(body, 'A bus-specific cross-fidelity workflow builds on Hybrid Offline-and-Online Reinforcement Learning',
        'A bus-specific cross-fidelity workflow builds on Hybrid Offline-and-Online Reinforcement Learning (H2O) and its extension H2O+, combining fixed SUMO offline data with new simplified-simulator experience. Simultaneous-arrival encoding, a single holding action that allows zero or positive additional stop time, and a low-load completion reserve form the complete controller. In 1,350 paired evaluations covering 10 new environment blocks, three demand levels, and nine controllers, the reserve reduces within-window generalized time by 92.12 seconds, or 3.71%, and increases completion by 0.466 percentage points relative to the direct parent with the same learned policy. An update-budget-matched comparison supports hybrid training over the tested offline-only scheme.', 'abstractenbody')
    set_by_prefix(body, 'The Model-Corrected World Model (MC-WM) selects residual correction',
        'The Model-Corrected World Model (MC-WM) selects residual correction or direct target modeling using held-out prediction errors and weights one-step model samples through log-confidence evaluation. In the bus extension, fixed target offline data and hybrid event models support model-assisted training, while a normalized confidence backup limits attenuation of model updates. With a single holding action and a passenger-time objective, mean generalized time decreases relative to a Daganzo-style rule with the same completion reserve by 57.67, 70.67, and 76.26 seconds under calibrated, joint 50%, and joint 75% source-parameter mismatch. The mismatch jointly changes road speed, passenger OD demand, intersection delay, and basic stop-service time. The improvement is larger when source calibration is insufficient, supporting target-data-informed correction of source simulation training.', 'abstractenbody')
    set_by_prefix(body, 'The thesis connects stationary robust control',
        'The thesis connects stationary robust control, non-stationary adaptation, cross-fidelity transfer, and world-model correction through the availability and use of target-domain information. The main bus experiments retain a single holding action so that algorithmic and data-transfer effects are not confounded with an enlarged action interface.', 'abstractenbody')

    # Comment resolutions that affect the rest of the manuscript.
    set_by_prefix(body, '仿真中有效的策略不一定能在实际使用环境中取得相同效果',
        '仿真中有效的策略不一定能在实际部署环境中取得相同效果。本文将承担训练交互的仿真称为源环境，将策略最终接受运行评价的环境称为目标环境。二者可能在道路速度、客流起讫点分布、路口通行延误和停站服务时间等方面存在差异，使相同状态和动作产生不同的后续运行结果。本文以分布偏移（distribution shift）描述训练与部署时相关概率分布的变化，其中相同状态动作下的后继分布变化称为动力学偏移。动力学随机化研究通过在仿真训练中变化模型参数改善迁移，说明训练分布的构造能够影响现实适应；但这一路径需要确定哪些变化值得纳入训练。当目标离线数据已经存在时，还可以利用其中关于目标环境的证据，约束或修正源仿真中的学习过程。', 'body')
    set_by_prefix(body, '混合离线与在线强化学习（Hybrid Offline-and-Online Reinforcement Learning',
        '混合离线与在线强化学习（Hybrid Offline-and-Online Reinforcement Learning，H2O）及其扩展H2O+将目标环境中预先采集的离线数据与训练仿真中新产生的交互经验结合，提供了不同于纯离线学习和目标在线微调的数据利用方式。[9,10] 对公交驻站控制而言，这一思想具有明确吸引力：保留仿真试错的低成本，同时利用目标记录减少学习对源模型的完全依赖。然而，把通用学习框架用于实际公交运营，还需要处理异步事件、零附加停站动作、运行时域以及未完成服务等具体问题。', 'body')
    set_by_prefix(body, '本论文以“从仿真到现实的公交系统驻站控制方法研究”为题',
        '本论文以“从仿真到现实的公交系统驻站控制方法研究”为题，以目标环境信息的逐步增加组织研究：缺少预先目标域数据时，依靠仿真中的随机化、鲁棒学习和条件适应，提高策略对潜在变化的适用性；获得目标离线数据后，进一步用这些数据约束跨保真训练，并修正源环境与目标环境之间的可信关系。自建简易仿真承担低成本训练，SUMO作为更细致的目标仿真环境，连接算法研究与面向实际运营的部署目标。', 'body')
    set_by_prefix(body, '全文按环境假设与目标数据可用性递进',
        '全文按环境假设与目标数据可用性递进。第3章从平稳随机环境出发研究鲁棒强化学习，第4章在保留随机波动的基础上增加环境规律随时间变化的假设；这两章主要利用源环境交互数据，研究目标环境数据不可得时的控制问题。第5章引入目标环境离线数据，利用这些信息约束源环境中的策略训练；第6章进一步把源目标差异纳入预测模型及其部署方式。四章共同服务于仿真到现实迁移，分别回答如何面对随机性、如何识别变化、如何利用目标经验以及如何修正模型。以下先给出共用定义，后续各章重点展开相对于这些定义的新问题、新假设和算法。', 'body')
    set_by_prefix(body, '本节的为计划目标间隔',
        '本节定义计划对齐的车头时距奖励。为避免把解析响应与乘客服务指标混为一谈，后文分别报告计划时距偏差、车头时距不均衡和广义乘客时间。', 'body')

    # The Windows revision still contains several planning paragraphs from the
    # earlier optional speed-control campaign.  The completed thesis matrix is
    # intentionally single-action; rewrite these descriptions before deleting
    # the optional result sections so no stale cross-reference survives.
    set_by_prefix(body, '本文研究公交驻站策略如何从可反复训练的仿真走向信息有限',
        '本文研究公交驻站策略如何从可反复训练的仿真走向信息有限、运行条件可能变化的目标环境。共同控制任务是：车辆完成必要上下客后，根据当前运行信息选择有界附加驻站时长，以改善车头时距或乘客服务。线路、班次及供给计划构成给定条件，候车时间、车内时间、运行间隔和行程完成情况用于评价控制后果。全文实验统一采用附加驻站这一单一控制动作，速度倍率固定为1。', 'body')
    set_by_prefix(body, '具体输入维度取决于各研究的控制接口。多线路公交离线学习将15个原始观测量',
        '具体输入维度取决于各研究的控制接口。多线路公交离线学习将15个原始观测量与上一驻站动作共同用于输入。H2O+最多组织16个同时事件槽位，原始打包状态为245维，规范化后为229维；有效槽位标记及批内已选驻站量共同描述决策上下文。本文主矩阵均沿用这一单一驻站接口，不把速度命令作为策略输出。', 'body')
    set_by_prefix(body, '联合动作仍采用归一化编码',
        '附加驻站量采用归一化编码，再映射至0至60秒的可执行范围。策略保留精确的零驻站概率，正驻站值用于调节车辆离站时刻；经验与价值网络始终接收同一动作编码，不把执行单位与训练单位混用。', 'body')
    set_by_prefix(body, '速度调节与驻站作用于不同物理过程',
        '道路速度和驻站分别属于环境转移因素与策略控制量。本文多线路离线学习、公交H2O+和MC-WM主实验均只学习附加驻站，策略输出不包含速度倍率；实际车速仍受道路限速、信号和跟驰条件影响。用于标定偏差的道路平均速度属于环境转移条件，与策略动作分别定义，因此环境偏差实验不会把改变路况误写为新增控制权限。', 'body')
    set_by_prefix(body, 'SUMO平台从道路网络、站点、公交路线和班次时刻表构建车辆服务任务',
        'SUMO平台从道路网络、站点、公交路线和班次时刻表构建车辆服务任务，再叠加信号配时、社会交通与乘客OD。车辆在道路中按微观运行机制推进，控制接口在指定服务事件处读取状态并写入附加驻站时长。由此，路网和交通输入决定道路传播，容量与上下客决定站点服务，学习器决定驻站控制，三部分在执行中保持对应。', 'body')
    set_by_prefix(body, '自建平台使用事件驱动方式推进公交运行',
        '自建平台使用事件驱动方式推进公交运行。走廊配置以路段运行参数和分时需求产生车辆与乘客事件，用于第3、4章反复比较相同条件下的学习方法；多线路配置读取与SUMO一致的线路、站点、时刻表和乘客行程，以标定的路段运行时间替代逐秒微观车辆运动，用于第5、6章的源环境训练。两种配置共同保留到站、上下客、容量约束、附加停站和离站的因果顺序，具体场景不混作同一实验。', 'body')
    set_by_prefix(body, '两种奖励与两种动作的补充比较不覆盖已有检查点',
        '不同奖励目标的补充比较不覆盖已有检查点。本文正式结果统一使用单一驻站动作；第3、4章从各自车辆续接接口训练，第5、6章使用多线路事件接口。各章结果按训练数据、奖励目标和评价环境分别汇总，不将不同状态表示或事件结算产生的数值合并。', 'body')
    set_by_prefix(body, '零与正驻站是同一个物理量的两部分',
        '零与正驻站是同一个物理量的两部分，不是两个控制维度。本文所有实验均保持速度倍率为1，策略只输出有界附加驻站时长；动作分布保留精确零驻站概率，正驻站值对应可执行的离站延迟。', 'body')
    set_by_prefix(body, '这些规则使用决策时可获得的运行量',
        '这些规则使用决策时可获得的运行量。该前向规则体现Daganzo的车头时距反馈思想，所比较对象是式(5-12)的具体比例规则。不同章节根据可观测的前后车间隔分别定义规则输入，不将不同信息接口下的规则实现合并为一个固定控制器。', 'body')
    set_by_prefix(body, '控制收益还取决于运营目标和动作权限',
        '控制收益还取决于运营目标。车头时距规则主要调节运行间隔，低载尾部保障为特定状态保留服务机会。本文在统一的单一驻站动作接口下分别优化时距与乘客时间，说明同一训练方法可以服务于不同目标，但不能以某项评价指标替代实际奖励。', 'body')
    set_by_prefix(body, '输出结构与行为克隆交叉比较中的四组均已采用',
        '输出结构与行为克隆交叉比较中的四组均已采用第4.3.2节的训练稳定化，因而表4-9检验的是稳定化之后的组件增量。该比较保持单一驻站动作、相同训练预算和相同评价目标，结果只用于解释输出结构与行为克隆的相对作用。', 'body')
    set_by_prefix(body, '第4.4.1至4.4.6节及第4.4.7节的稳定化主比较',
        '第4.4.1至4.4.6节的稳定化主比较没有启用符号发现和不变风险最小化约束。基于稀疏符号参考的补充比较单独分析，其收益不与主配置合并，也不把NAU/NMU输出视为恢复了公交因果结构。附录D给出的策略响应界仍以完整映射的函数与导数条件为前提。', 'body')
    set_by_prefix(body, '本章原有公交比较仅输出附加驻站',
        '本章公交比较统一采用附加驻站这一单一控制动作，执行端速度倍率固定为1。偏差实验改变的道路平均速度、客流OD、路口延误及基本服务时间属于环境转移关系，不等于策略动作；同组方法保持相同驻站上限与完成保障。', 'body')
    set_by_prefix(body, '驻站—速度联合控制进一步检验两种运营目标',
        '单一驻站控制进一步检验两种运营目标。以乘客时间训练时，MC-WM在正常标定、联合50%和联合75%源偏差下相对共同保障的Daganzo式规则分别减少57.67、70.67和76.26秒人均广义乘客时间；以车头时距训练时，MC-WM在高偏差条件下相对H2O+的优势更明显，而正常标定下二者差距较小。两种目标的结果共同表明，世界模型的公交应用不局限于一种奖励；其作用取决于预测关系、预测进入价值更新的强度与事件结算的一致性。', 'body')
    set_by_prefix(body, '本节将模型选择与贝尔曼可信度用于公交事件过程',
        '本节将模型选择与贝尔曼可信度用于公交事件过程，在固定SUMO离线数据和新增源仿真经验下评价模型辅助。事件预测模型与归一化置信算法已在前文给出。第6.4.3节保留模型类别、损失系数及置信机制的比较，用于解释模型信息如何进入价值更新，并报告单一驻站的重复运行结果；本节进一步以H2O+和Daganzo式规则为主要参照，交叉考察两种奖励。早期单次接入、预算与偏差探索见附录F，按各自训练条件解释。', 'body')
    set_by_prefix(body, '联合动作扩展使预测器同时接收驻站和速度两维条件',
        '公交世界模型接收与驻站事件相匹配的状态和动作条件，预测物理边界后的事件变化。乘客奖励由预测的非负时间成本形成，时距奖励由预测后继事件的时距量按同一计划对齐函数计算。模型仅增加训练期的一步预测信息，不在部署时生成额外控制命令或接入目标数据。', 'body')
    set_by_prefix(body, '为进一步判断服务收益是否超过反馈规则',
        '为进一步判断服务收益是否超过反馈规则，本节在相同开发和测试情形上评价Daganzo式规则和车头时距保障规则。两者与学习方法均采用0至60秒的附加驻站动作范围及共同完成保障，不因源偏差或本轮结果调整参数。规则不使用源仿真，同一组目标测试成本在三种源条件下复用。', 'body')
    set_by_prefix(body, '不同训练和评价批次分别报告',
        '不同训练和评价批次分别报告，历史单一驻站结果按对应训练条件独立汇总。各节按独立重复运行和共同目标情形取平均，相关测试情形已在开发阶段使用，属于已有场景下的对照评价，不作为未见城市的泛化证明。结果表征完整模型辅助配置在对应数据和执行条件下的平均效果。', 'body')

    # Remove the optional joint holding-speed sections from the main thesis.
    for start, end in [
        ('3.4.9 ', '3.5 '),
        ('4.4.7 ', '4.5 '),
        ('5.2.7 ', '5.3 '),
        ('5.4.4 ', '5.5 '),
        ('6.4.4 ', '6.5 '),
    ]:
        remove_section(body, start, end)
    for toc in ['3.4.9 ', '4.4.7 ', '5.2.7 ', '5.4.4 ', '6.4.4 ']:
        remove_toc_entry(body, toc)

    # Remove or narrow references left behind by the deleted optional sections.
    replace_t = {
        '对公交在途控制而言': '对公交驻站控制而言',
        '公交在途控制': '公交运行过程控制',
        '在途控制': '公交运行过程控制',
        '公交在途时长': '公交运行时长',
        '在途时长': '运行时长',
        '第3.4.9、第4.4.7、第5.4.4和第6.4.4节补充驻站—速度协同控制。每个算法的比较组具有相同动作范围，不能把增加速度权限本身解释为算法收益。': '本文主实验统一采用附加驻站动作；速度控制不纳入本论文主矩阵。',
        '第3.4.9节一致': '前文关于时距目标与乘客时间目标的讨论一致',
        '第4.4.7节另与未采用整套稳定化的前序BAPR比较，并扩展奖励和动作范围，两种比较分别对应完整配置收益与输出结构的附加作用。': '本节仅比较完整训练稳定化与输出结构的组件作用，未将符号约束收益扩展为普遍结论。',
        '第4.4.1至4.4.6节及第4.4.7节的稳定化主比较': '第4.4.1至4.4.6节的稳定化主比较',
        '零与正驻站是同一个物理量的两部分，不等于驻站与速度两个动作维度。原有单动作实验保持速度倍率为1；第5.2.7节再增加独立的有界速度输出。两类模型均保留精确零驻站概率，联合模型不会把零驻站改写成难以达到的连续边界。': '零附加停站和正的附加停站时长属于同一个驻站动作的不同取值。本文所有主实验均保持这一单一动作接口，并保留精确的零停站概率；不再引入独立的速度输出。',
        '本节采用第5.4.4节的联合动作接口，固定目标离线数据，完成50000次离线初始化后，以简易仿真采集约25000个事件并新增20000次策略与评论家更新。比较正常标定和联合75%源偏差两类训练条件；后者将道路平均速度设为原值的0.25倍、OD与基本服务时间设为1.75倍、额外路口遭遇密度设为0.75。目标SUMO始终不变，也不参与训练交互。两档目标需求分别采用1.00倍和1.25倍，每档五个共同场景，在24000秒时域内评价固定策略。': '本章的跨保真主实验固定目标离线数据，在简易仿真中续训，并在新的SUMO场景中评价固定驻站策略。正常标定与联合源参数偏差分别用于检验标定充分和标定不足两类条件；联合偏差同时改变道路平均速度、客流OD、路口延误和基本停站服务时间，目标SUMO不参与训练交互。',
        '第5.4.4节在共同的双动作接口中分别优化时距与乘客时间，说明同一跨保真训练方法可以服务于不同目标，但不能以某项评价指标替代实际奖励。': '在单一驻站接口下分别优化时距和乘客时间，说明同一跨保真训练方法可以服务于不同运营目标，但不能以某项评价指标替代实际奖励。',
        '第6.4.4节进一步采用第5.2.7节的驻站—速度联合动作，MC-WM与H2O+共用状态和动作接口。世界模型辅助不赋予任何比较方法额外控制权限。偏差实验改变的道路平均速度、客流OD、路口延误及基本服务时间属于环境转移关系，不等于策略可选速度倍率；同组方法还保持相同驻站上限与完成保障。': '本章公交比较统一采用单一附加驻站动作。偏差实验改变的道路平均速度、客流OD、路口延误及基本服务时间属于环境转移关系，不等于策略动作；各方法保持相同驻站上限与完成保障。',
        '第6.4.4节进一步在驻站—速度协同下，以H2O+和Daganzo式规则为主要参照，交叉考察两种奖励。': '本章仅在单一驻站接口下，以H2O+和Daganzo式规则为主要参照，分别考察时距目标和乘客时间目标。',
        '第6.4.4节按独立重复运行和共同目标情形取平均，相关测试情形已在开发阶段使用，属于已有场景下的对照评价，不作为未见城市的泛化证明。': '本章按独立重复运行和共同目标情形取平均；相关测试情形已在开发阶段使用，属于已有场景下的对照评价，不作为未见城市的泛化证明。',
        '双动作主比较固定共同的数据来源、动作范围和执行保障，时距训练还共同使用梯度范数约束，因而不能把仅对某一算法施加的训练稳定化误写为模型收益。': '主比较固定共同的数据来源、单一驻站动作范围和执行保障，因而不能把仅对某一算法施加的训练稳定化误写为模型收益。',
        '双动作扩展中，乘客时间目标下相对H2O+的降幅为3.48%和4.81%，相对Daganzo式规则为1.94%和4.13%；时距目标下相对规则的平均惩罚降幅为4.87%和3.18%。联合75%偏差下，时距训练的MC-WM还比同样施加梯度约束的H2O+降低3.61%的乘客成本。模型辅助可用于两种公交目标，但乘客目标下的策略不必具有最低时距偏差。': '单一驻站重复运行确认表明，固定模型项系数0.25在正常标定、联合50%和75%源偏差下，相对共同保障的Daganzo式规则分别减少57.67、70.67和76.26秒人均广义时间。模型辅助可以改善所测两类公交目标，但乘客目标下的策略不必具有最低时距偏差。',
    }
    replace_t_text(document, replace_t)

    # Chapter-level opening prose, matching the body style used after Chapter 2.
    template = find_paragraph(body, '公交车队驻站控制的基本任务', 'body')
    chapter_openers = {
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
    for heading, paragraphs in chapter_openers.items():
        h = find_paragraph(body, heading, 'chapter')
        insert_after(body, h, [clone_body_paragraph(template, t) for t in paragraphs])

    # Replace chapter summaries with conclusions that match the retained action
    # interface and the completed experiment evidence.
    set_by_prefix(body, '本章围绕缺少预先目标域数据时，平稳随机环境中的公交车队驻站学习问题展开',
        '本章围绕目标环境数据尚不可得时的平稳随机公交驻站学习问题展开。模型层面，以车辆、站点、时段和方向等类别信息结合连续运行状态，形成由一个学习器处理多车辆事件的共享策略；算法层面，提出RE-SAC，以固定目标偏移、集成价值评价和分歧约束组织保守动作价值学习。附录A给出固定参数理想算子的收缩性质，正文第3.3节说明这些条件与实际更新之间的边界。', 'body')
    set_by_prefix(body, '双向走廊仿真支持共享控制方案的可行性',
        '在单一附加驻站动作下，类别嵌入相对独热编码在正常运行、客流激增和快速环境切换下分别降低5.51%、1.41%和4.91%的人均广义乘客时间；恢复原项目关键优化设置后的匹配多线路实验中，RE-SAC在时距目标下六种运行条件均优于相应SAC对照，计划偏差也整体更低。乘客时间目标下两类方法存在指标取舍，说明时距奖励、乘客时间和价值评价稳定性不能合并为单一结论。', 'body')
    set_by_prefix(body, '长期训练和动作续接实验进一步表明',
        '长期训练和独立动作后果实验进一步表明，探索强度、价值排序和后期温度变化都会影响控制收益，单看价值数值误差不足以判断策略质量。以上结论仍限于给定计划、车辆供给和有限驻站范围；客流或路况规律持续改变时，原有动作映射可能不再适合。下一章在相同单智能体和单一驻站权限下引入隐含运行模式的漂移与切换，研究如何由历史证据识别变化并调整控制响应。', 'body')

    set_by_prefix(body, '本章将非平稳公交运行写为隐含条件识别与有界控制响应问题',
        '本章将非平稳公交运行写为隐含条件识别与有界驻站控制响应问题，由BAPR的信念调节进一步研究CS-BAPR的训练稳定化和策略结构。在统一的单一驻站动作下，稳定化方法相对前序BAPR在六种需求条件下均降低平均时距奖励惩罚，降幅为58.82%至69.79%；与RE-SAC的共同对照支持其在运行规律变化下的条件适应优势。带因果符号约束的策略输出并非在所有条件下都带来额外收益，改用乘客时间目标后服务成本差异缩小。', 'body')
    set_by_prefix(body, '第3、4章由源仿真中的保守学习和条件适应构成未预先获得目标域数据时的两步研究',
        '第3、4章由源仿真中的鲁棒学习和条件适应构成目标环境数据不可得时的两步研究。第5章进一步引入固定目标离线数据，使训练能够利用目标环境提供的经验，并结合低成本源仿真继续交互；第6章在这一信息条件上进一步研究转移模型的修正。论文由此沿同一sim2real问题逐步增加可用信息和需要处理的环境差异。', 'body')

    set_by_prefix(body, '本章围绕目标离线数据可获得、目标交互受限的公交驻站控制问题',
        '本章围绕目标离线数据可获得、目标交互受限的公交驻站控制问题，区分源目标访问分布、条件转移和有限采样误差，并在统一驻站接口下分别研究目标数据利用与源仿真续训。固定SUMO离线数据能够支持离线策略学习；在简易仿真中追加交互可进一步改善部分控制回报，但运行规律性与乘客时间存在不同权衡。', 'body')
    set_by_prefix(body, '后一任务以广义乘客时间为目标，公交H2O+',
        '跨保真主任务以广义乘客时间为目标，公交H2O+将固定SUMO离线数据与简易仿真新增经验结合，采用同时到站车辆编码、单一附加驻站动作和运行末段完成保障。十个独立环境块的1350次配对评价表明，完成保障相对相同学习策略的直接父控制器使窗口内人均广义时间减少92.12秒、下降3.71%，完成率提高0.466个百分点。该收益属于执行保障，应与混合训练相对纯离线方案的作用分开归因。', 'body')
    set_by_prefix(body, '360次学习与执行交叉评价表明',
        '学习与执行交叉评价表明，完成保障在多种基础控制器上均有作用；同保障间隔规则与混合方案的稳定差异未分辨。五训练种子的同更新预算比较支持混合方案相对所测纯离线方案的收益，但尚未建立其相对纯源训练的稳定优势。结果限于规定SUMO评价窗口和单一驻站动作接口，下一章进一步利用目标数据学习转移模型，以处理源仿真标定不足。', 'body')

    set_by_prefix(body, '本章针对源仿真与目标环境的转移规律差异，提出MC-WM',
        '本章针对源仿真与目标环境的转移规律差异提出MC-WM。在模型层面，以源预测为基础建立残差与直接目标两类候选，并按任务分别表示连续状态变化和公交事件转移；在算法层面，利用留出目标数据选择模型，以可信度加权的一步预测支持价值更新，并通过归一化置信调节模型使用强度。受控任务检验模型类别选择，公交任务在固定目标离线数据和单一驻站动作下检验模型辅助对乘客服务的作用。', 'body')
    set_by_prefix(body, '在三个受控动力学任务、各10个训练种子的独立比较中',
        '在受控动力学任务中，MC-WM相对仅目标数据SAC的回报改善与模型类别选择结果支持其一般机制；公交多种子实验则把模型项系数、置信备份和共同执行保障分开比较。固定模型项系数0.25在正常标定、联合50%和75%源偏差下，相对共同保障的Daganzo式规则分别减少57.67、70.67和76.26秒人均广义乘客时间；偏差越大时，模型辅助相对无模型骨干的收益更明显。', 'body')
    set_by_prefix(body, '公交扩展以固定目标离线数据和简易仿真训练为条件',
        '机制分析表明，预测误差基本稳定时，未归一化的置信传播会使后期模型作用减弱，归一化置信备份能够恢复部分模型权重和梯度贡献；去掉模型梯度或提高模型项系数均不能在所有标定条件下稳定增益。20个训练种子的固定配置与上下文扩展共同说明，模型拟合、模型使用强度和上下文调节是不同环节，不能把任一环节的局部改善写成MC-WM始终优于所有参照。', 'body')
    set_by_prefix(body, '机制记录将上述收益限制落实到具体更新过程',
        '这些结果把MC-WM的收益限制在可观察的机制上：目标数据帮助判断源结构是否应保留，归一化置信备份决定模型样本能否持续影响价值更新，而模型项系数仍需与源目标偏差匹配。模型增加训练开销，部署阶段仍只执行冻结的驻站策略网络。', 'body')
    set_by_prefix(body, '进一步的仅驻站重复运行确认表明',
        '进一步的单一驻站重复运行确认表明，固定模型项系数0.25在正常标定、联合50%和75%源偏差下，相对共同保障的Daganzo式规则分别减少57.67、70.67和76.26秒人均广义乘客时间。高偏差下模型辅助相对无模型骨干的改善更明显，但正常标定下MC-WM与H2O+的差距较小，说明其主要价值在于缓解标定不足，而不是在所有条件下替代已有跨保真方法。', 'body')
    set_by_prefix(body, '本章的主要认识是，源结构是否保留应由目标数据检验',
        '本章的主要认识是，源结构是否保留应由目标数据检验，模型进入策略更新的方式则决定其预测能否产生实际控制作用。受控类别干预说明不适当复用源结构会限制控制，公交多种子确认则把目标离线数据、模型使用强度与共同完成保障下的乘客服务增量连接起来。MC-WM在源仿真标定不足时具有更明确的价值，但该结论受固定数据覆盖、参数失配方式、评价窗口和单一驻站动作约束。', 'body')

    # Align the introduction and Chapter 7 synthesis with the retained
    # single-action experiment matrix and remove references to deleted figures.
    set_by_prefix(body, '国内公交研究也从乘客服务和具体控制权限出发处理运行失衡',
        '国内公交研究也从乘客服务和具体控制权限出发处理运行失衡。何胜学针对双线单换乘区研究驻站学习控制，将换乘引起的停站不确定性与线路稳定性联系起来。[22] 赵琥等以重叠线路为对象，通过车速诱导调节进入共同区间的车辆时序；李利华等针对站点群体聚集性客流，联合调整驻站时间和路段速度，在车头时距与乘客行程时间之间进行多目标优化。[23,24] 这些研究说明，需求变化、线路交互和控制权限需要共同进入问题定义。本文以车辆完成必要上下客后的附加驻站为共同基础，正式实验统一采用单一驻站动作；各组保持既定线路、班次及供给计划，学习方法在相同动作权限下比较，传统驻站规则作为无需训练的运行参照。', 'body')
    set_by_prefix(body, '本文由BAPR的条件识别和信念调节进一步提出CS-BAPR',
        '本文由BAPR的条件识别和信念调节进一步提出CS-BAPR，在既定计划下研究训练稳定化与策略输出结构的作用。单一驻站动作下的两种奖励比较表明，稳定化感知机相对前序BAPR在六种条件下均改善平均时距奖励惩罚，降幅为58.82%至69.79%。持续高需求和重复冲击均包含在评价中，收益并非只由一次短时扰动体现。算术输出未在所有条件下增加收益；改用乘客时间奖励后，服务成本接近，说明方法的时距目标优势需要与乘客指标分别解释。', 'body')
    set_by_prefix(body, '7.2.4 面向训练与使用环境差异的世界模型修正',
        '7.2.4 面向训练与部署环境差异的世界模型修正', 'h3')
    set_by_prefix(body, '针对源结构可能不适用于目标环境的问题，提出MC-WM',
        '针对源结构可能不适用于目标环境的问题，提出MC-WM，将源预测的保留写为残差与直接模型的显式选择，并以对数置信评价调节一步模型样本的损失权重。公交扩展按类别事件、计数、连续状态和持续时间组织预测，使固定目标离线数据参与事件模型学习；归一化置信备份则依据累积尺度性质维持模型作用。表6-5支持避免不适当源结构复用的价值；表6-13支持单一驻站条件下相对规则的服务收益，表6-16和表6-17进一步给出两种奖励下相对H2O+与Daganzo式规则的结果。理论分析、预测使用和控制评价由此形成连接。', 'body')
    set_by_prefix(body, '世界模型与上下文调节并非越复杂',
        '世界模型与上下文调节并非越复杂越好。固定模型项系数0.25形成主要公交结果，上下文修复未获得一致的额外收益。模型拟合、置信更新和样本生成增加训练成本，部署演员结构则保持不变。各奖励实验采用相应的数据和转移定义，跨章节数值不能脱离需求强度、评价时域及统计分母直接排名。受控任务中相对直接目标模型方法的回报优势仍未分辨。', 'body')

    # Remove any remaining literal reference to the optional action matrix in
    # prose that survived section deletion.
    for p in all_body_paragraphs(body):
        txt = text_of(p)
        if '双动作' in txt or '驻站—速度' in txt:
            # Do not touch ordinary formula paragraphs; remaining occurrences
            # are prose references and can be narrowed safely.
            new = txt.replace('双动作', '单一驻站动作').replace('驻站—速度协同', '驻站控制')
            if new != txt:
                set_paragraph_text(p, new)

    # Remove comment relationships and comment content types from the package.
    for rel in list(rels):
        typ = rel.attrib.get('Type', '')
        target = rel.attrib.get('Target', '')
        if 'comments' in typ.lower() or 'comments' in target.lower():
            rels.remove(rel)
    for override in list(content_types):
        part = override.attrib.get('PartName', '')
        if 'comments' in part.lower():
            content_types.remove(override)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    clean_package(SRC, OUT, document, rels, content_types)
    print(OUT)


if __name__ == '__main__':
    main()
