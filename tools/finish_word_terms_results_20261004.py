from __future__ import annotations

import json
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import process_windows_thesis_minimal_20261004 as base


SRC = Path('/home/erzhu419/mine_code/Thesis/output/排版完成_批注修订_保留原文_单驻站主线.docx')
OUT = Path('/home/erzhu419/mine_code/Thesis/output/排版完成_术语统一_最新结果补全_20261004.docx')


def insert_before(body, anchor, paragraphs):
    pos = list(body).index(anchor)
    for paragraph in paragraphs:
        body.insert(pos, paragraph)
        pos += 1


def append_text(paragraph, text):
    paragraph.append(base.make_run(text))


def h4(body, text, template_prefix):
    template = base.find_p(body, template_prefix, 'h4')
    paragraph = base.clone_body_paragraph(template, text)
    ppr = paragraph.find('./base.w:pPr', base.NS) if False else paragraph.find('./w:pPr', base.NS)
    if ppr is not None and ppr.find('./w:keepNext', base.NS) is None:
        ppr.append(base.ET.Element(base.q('keepNext')))
    return paragraph


def body_p(body, text, template_prefix='为检验第3.4.3节的规模排序'):
    template = base.find_p(body, template_prefix, 'body')
    return base.clone_body_paragraph(template, text)


def mean(rows, key):
    return statistics.mean(row[key] for row in rows)


def mcwm_guarded_summary():
    path = Path('/home/erzhu419/mine_code/Thesis/experiments/mcwm_headway_holding_v19/analysis_summary.json')
    data = json.loads(path.read_text(encoding='utf-8'))
    out = {}
    for scene in ('calibrated', 'joint75'):
        rows = [r for r in data['aggregates'] if r['scene'] == scene and r['layer'] == 'guarded']
        paired = [r for r in data['paired'] if r['scene'] == scene and r['layer'] == 'guarded' and r['reference'] == 'holding_h2oplus']
        mc = {key: mean(rows, key) for key in ('cost', 'reward', 'plan', 'imbalance')}
        delta = {f'{key}_delta': mean(paired, f'{key}_delta') for key in ('cost', 'reward', 'plan', 'imbalance')}
        h2o = {key: mc[key] - delta[f'{key}_delta'] for key in mc}
        out[scene] = {'mc': mc, 'h2o': h2o, 'delta': delta}
    return out


def main():
    base.SRC = SRC
    base.OUT = OUT
    with __import__('zipfile').ZipFile(SRC, 'r') as zin:
        document = base.ET.fromstring(zin.read('word/document.xml'))
        rels = base.ET.fromstring(zin.read('word/_rels/document.xml.rels'))
        content_types = base.ET.fromstring(zin.read('[Content_Types].xml'))

    body = document.find('.//base.w:body', base.NS) if False else document.find('.//w:body', base.NS)
    if body is None:
        raise RuntimeError('document body missing')
    base.accept_revisions(document)
    base.strip_comment_markers(document)

    # Keep the English abstract aligned with the author's revised Chinese
    # abstract: holding alone is the formal main line; joint speed control is
    # retained in the body as an extension, not as a competing abstract claim.
    base.set_text(base.find_p(body, 'This thesis proposes a shared holding policy', 'abstractenbody'),
        'This thesis proposes a shared holding policy combining categorical embeddings with continuous observations and develops Robust Ensemble Soft Actor-Critic (RE-SAC) for conservative action-value learning. With a common learner, learned embeddings reduce generalized passenger time relative to one-hot encoding by 5.51%, 1.41%, and 4.91% under routine operations, demand surges, and rapid switching. Generalized time is twice waiting time plus onboard time. In a matched multi-line review with holding alone and a headway objective, RE-SAC achieves mean headway reward of −101.43 versus −113.88 for SAC; planned-headway deviation and forward-backward headway imbalance decrease by 5.00 and 7.36 seconds, respectively. The passenger-time objective does not yield a uniform RE-SAC advantage, so the result is bounded by the operating objective.')
    base.set_text(base.find_p(body, 'To address changing operating laws and bounded control authority', 'abstractenbody'),
        'To address changing operating laws and bounded control authority, this thesis proposes Bayesian Adaptive Policy Regularization (BAPR), using environment identification and belief-based adjustment, and subsequently develops Causal-Symbolic Bayesian Adaptive Policy Regularization (CS-BAPR). The main bus configuration uses holding alone and combines training stabilization with alternative policy-output structures. Across routine demand, single and repeated shocks, sustained high demand, and a brief extreme shock, the stabilized perceptron configuration reduces the mean headway-reward penalty relative to the preceding BAPR configuration by 58.82% to 69.79%. Arithmetic outputs do not provide an additional benefit in every condition. When training instead targets passenger time, service costs remain close, so headway-reward gains cannot be interpreted as equal reductions in passenger time.')
    base.set_text(base.find_p(body, 'A bus-specific cross-fidelity workflow builds on Hybrid Offline-and-Online Reinforcement Learning', 'abstractenbody'),
        'A bus-specific cross-fidelity workflow builds on Hybrid Offline-and-Online Reinforcement Learning (H2O) and its extension H2O+, combining fixed SUMO offline data with new simplified-simulator experience. Simultaneous-arrival encoding, a single holding action that either adds no extra stopping time or adds a positive holding duration, and a low-load completion reserve form the complete controller. In 1,350 paired evaluations covering 10 new environment blocks, three demand levels, and nine controllers, the reserve reduces within-window generalized time by 92.12 seconds, or 3.71%, and increases completion by 0.466 percentage points relative to the direct parent with the same learned policy. An update-budget-matched comparison supports hybrid training over the tested offline-only scheme.')
    base.set_text(base.find_p(body, 'The Model-Corrected World Model (MC-WM)', 'abstractenbody'),
        'The Model-Corrected World Model (MC-WM) selects residual correction or direct target modeling using held-out prediction errors and weights one-step model samples through log-confidence evaluation. Across three controlled dynamics shifts in Multi-Joint dynamics with Contact (MuJoCo), with target interaction, mean return improves over target-only Soft Actor-Critic (SAC) by 1,925.7, 1,184.5, and 133.4. Under actuator shift, direct modeling gains 1,529.9 return over forced residual correction. The bus extension uses fixed target offline data and hybrid event models, with a normalized confidence backup limiting attenuation of model updates. For holding alone with a passenger-time objective, mean generalized time decreases relative to a Daganzo-style rule with the same completion reserve by 57.67, 70.67, and 76.26 seconds under calibrated, joint 50%, and joint 75% source-parameter mismatch. A separate holding-only headway check with model coefficient 0.125 yields a near tie with H2O+ under calibration and a 13.36-second reduction in generalized time per passenger under joint 75% mismatch. These results support using target offline data to correct source-simulator training, while distinguishing passenger-time and headway objectives.')

    # Replace only the two terms that the user identified as unnatural. The
    # replacements are ordered from the most specific phrase to the generic
    # phrase so that headings and ordinary prose remain grammatical.
    terms = {
        '低载尾部车辆的行程完成保障': '低载且无后车车辆的行程完成保障',
        '低载尾部服务保障': '低载且无后车时的服务保障',
        '低载尾部保障对象': '低载且无后车时的保障对象',
        '低载尾部车辆': '低载且无后车车辆',
        '低载尾部状态': '低载且无后车状态',
        '低载尾部保障': '低载且无后车时的服务保障',
        '低载尾部': '低载且无后车时',
        '零停站或正停站时长': '不额外停站或增加驻站时长',
        '零、正停站混合动作': '不额外停站或增加驻站时长的动作',
        '零、正停站混合策略': '不额外停站或增加驻站时长的混合策略',
        '零、正停站': '不额外停站或增加驻站时长',
        '零与正停站': '不额外停站与增加驻站时长',
        '零与正停': '不额外停站与增加驻站时长',
        # Repair grammar created after the generic phrase replacements.
        '低载且无后车时车辆': '低载且无后车车辆',
        '低载且无后车时保障对象': '低载且无后车时的保障对象',
        '低载且无后车时服务保障': '低载且无后车时的服务保障',
    }
    base.replace_visible_text(document, terms)
    base.replace_across_text_runs(document, terms)

    # The completed native-recipe comparison belongs beside the existing RE-
    # SAC component and scale comparisons, before the objective/action matrix.
    v8_heading = h4(body, '3.4.8.8 原项目优化配方在多线路环境中的复核', '3.4.8.7 集成规模的五种子重复与计算代价')
    v8_text = [
        '前述RE-SAC比较是在本章既定训练设置下完成的。为确认观察到的差异不是由多线路适配时改变优化配方造成，本文补充恢复原项目关键设置的匹配复核：共享类别嵌入、学习率10⁻⁵、批量2048、十个评论家（归档SAC参照采用两个）、不放回采样和每次更新的五个已结算转移均恢复；多线路状态、归一化、同车反馈、物理动作、评价日和100回合预算保持不变。该组共32条训练和576条评价，覆盖两种动作设置、两种运营目标及四个价值评价通道。这里报告的是设置组合复核，不把收益归因于某一个超参数。',
        '在单一驻站、车头时距目标下，RE-SAC的平均车头时距奖励为−101.43，SAC为−113.88，前者提高12.46个奖励单位；计划时距偏差由106.47秒降至101.47秒，前后时距不均衡由126.79秒降至119.44秒。关闭两类不确定性通道后的平均奖励分别为−103.55和−134.51，说明在本组设置中同时保留两类评价通道更合适。这里的“提高”针对车头时距奖励，不能等同于广义乘客时间下降。',
        '乘客时间目标下，RE-SAC的平均广义乘客时间为15710.36秒/人，SAC为15680.72秒/人，未形成RE-SAC的统一优势；双动作、车头时距目标和双动作、乘客时间目标也没有得到普遍改善。因此，本章将RE-SAC的明确结论限定为单一驻站、车头时距目标下的匹配比较，并把不同运营目标的取舍保留为实验边界。',
    ]
    v8_paragraphs = [body_p(body, t) for t in v8_text]
    first_v8_ppr = v8_paragraphs[0].find('./w:pPr', base.NS)
    if first_v8_ppr is not None and first_v8_ppr.find('./w:keepLines', base.NS) is None:
        first_v8_ppr.append(base.ET.Element(base.q('keepLines')))
    insert_before(body, base.find_p(body, '3.4.9 两种运营目标与两种动作权限的交叉比较', 'h3'), [v8_heading] + v8_paragraphs)

    # The final holding-only MC-WM headway check is separate from the earlier
    # dual-action matrix. Compute its paired H2O+ values directly from the
    # server-side summary instead of copying rounded values by hand.
    mc = mcwm_guarded_summary()
    cal = mc['calibrated']
    high = mc['joint75']
    v19_heading = h4(body, '6.4.3.9 单一驻站车头时距目标下模型项系数的复核', '6.4.3.8 开发集选择与上下文调节的扩展比较')
    v19_text = [
        '前述公交世界模型结果还包含驻站—速度联合动作和不同模型项系数的比较。为单独检验单一驻站主线下的时距目标，本文固定模型项系数为0.125、观测项系数为0.75，保留目标离线数据、源仿真交互和完成保障，分别在正常标定与联合75%源参数偏差下复核两个独立初始化。该组共4条训练和80条评价，不能与前面的双动作结果或乘客时间结果合并统计。',
        f'在完成保障执行层下，正常标定时MC-WM的平均广义乘客时间为{cal["mc"]["cost"]:.2f}秒/人，H2O+为{cal["h2o"]["cost"]:.2f}秒/人，MC-WM低{abs(cal["delta"]["cost_delta"]):.2f}秒/人，差异接近持平；联合75%偏差下，两者分别为{high["mc"]["cost"]:.2f}和{high["h2o"]["cost"]:.2f}秒/人，MC-WM低{abs(high["delta"]["cost_delta"]):.2f}秒/人。后一个条件的改善约为0.49%，说明模型修正在标定不足时更有价值，但不能外推为所有标定条件下都稳定优于H2O+。',
        '同一复核中的直接执行层与完成保障执行层方向不同，说明执行保障和模型样本使用强度会共同影响最终服务指标。因而，本文把完成保障层作为主比较的部署执行协议，同时将直接层保留为诊断，不用跨执行层的数值比较支持更强的算法结论。',
    ]
    insert_before(body, base.find_p(body, '6.4.4 驻站—速度协同下两种奖励的对照结果', 'h3'), [v19_heading] + [body_p(body, t) for t in v19_text])

    # Carry the two final checks into the conclusion without changing the
    # author's existing conclusion paragraphs.
    p_resac = base.find_p(body, '本文提出的RE-SAC以集成评价、目标偏移及分歧惩罚', 'body')
    append_text(p_resac, ' 恢复原项目关键优化设置的多线路复核进一步表明，在单一驻站、车头时距目标下，RE-SAC相对SAC的平均奖励提高12.46个单位，计划偏差和前后时距不均衡分别降低5.00秒和7.36秒；乘客时间目标下未形成统一优势。')
    p_mcwm = base.find_p(body, '驻站—速度联合控制进一步检验两种运营目标', 'body')
    append_text(p_mcwm, f' 补充的单一驻站时距复核将模型项系数固定为0.125；在完成保障执行下，正常标定时相对H2O+低{abs(cal["delta"]["cost_delta"]):.2f}秒/人，联合75%偏差时低{abs(high["delta"]["cost_delta"]):.2f}秒/人，前者接近持平，后者改善更清楚。')

    base.remove_comment_relationships(rels, content_types)
    base.clean_package(document, rels, content_types)
    print(OUT)


if __name__ == '__main__':
    main()
