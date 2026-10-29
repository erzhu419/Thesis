"""Merge context-checked table corrections; leave all main-text issues recorded only."""
from pathlib import Path
import json

A = Path(__file__).resolve().parent
original = json.loads((A / 'symbol_table.json').read_text())
plan = {r['row']: {'source_row': r['row'], 'action': 'keep',
                   'original_symbol': r['symbol'], 'original_definition': r['definition'],
                   'correct_tex': r['symbol'], 'correct_definition': r['definition'],
                   'reason': '实际使用及定义已核对，保留。'} for r in original}

def clean(tex):
    if tex is None:
        return None
    return tex.replace('$', '').replace('；', ',').replace('\\\\', '\\').strip()

def patch(row, tex, meaning, reason):
    plan[row].update(action='update' if tex else 'delete', correct_tex=clean(tex),
                     correct_definition=meaning, reason=reason)

c13 = json.loads((A / 'review_ch1_3.json').read_text())
for p in c13['table_patches']:
    if p.get('row'):
        patch(p['row'], p['suggested_symbol_tex'], p['suggested_definition'],
              '按第1–3章及相应附录的实际记号调整；详见逐项证据。')
c46 = json.loads((A / 'review_ch4_6.json').read_text())
for p in c46['table_row_notes']:
    if p['action'] != 'keep':
        patch(p['row'], p['correct_tex'], p['correct_definition'], p['reason'])
app = json.loads((A / 'review_appendices.json').read_text())
for p in app['table_patches']:
    if p['row'] not in (38, 86, 97):
        patch(p['row'], p['correct_tex'], p['correct_definition'], '主文与附录的实际记号核对。')
for p in json.loads((A / 'review_uncovered_rows_patch.json').read_text())['table_row_notes']:
    if p['action'] != 'keep':
        patch(p['row'], p['correct_tex'], p['correct_definition'], p['reason'])

# Resolve the small overlaps between chapter reviewers from the actual source.
patch(1, r'\mathcal L_{od},l', 'OD对(o,d)的可服务线路集合及线路索引（式5-5）',
      '保留第5章确实使用的线路集合，删除表中未对应的泛称。')
patch(5, r't^{d},t^{s},y_{i,j}^{a},y_{i,j}^{d}',
      '本站计划离站时刻、完成必要上下客的时刻；班次i在站点j的实际到站与离站时刻（第2、3章）',
      '采用原公式实际时刻记号，上标字体随原定义。')
plan[10]['correct_definition'] = plan[10]['correct_definition'].replace('第3.2.1节', '第3.2.2节')
patch(11, r'\lambda,\eta^{board}',
      '局部乘客到达率与单位乘客上车服务时间（第3、4章；OD分时到达率另依式5-5定义）',
      '保留第4章确实使用的上车服务时间，去掉未用的站点索引别名。')
patch(23, r'\xi,\bar\xi',
      '第6章对数置信评价网络及其目标网络的参数；第2章的同形ξ另为CVaR成本阈值',
      '原ξ_π高斯噪声无对应，改列实际使用的置信网络参数及其作用域。')
patch(24, None, None, 'ε_fp在当前正文和附录无对应，删除旧别名。')
patch(26, r'\gamma,\Gamma_k,d_t,d_k,d_i,d',
      '回报折扣因子、事件延续因子与回合终止标记；终止标记索引随对应事件定义',
      '使用已有终止记号，删除从未采用的term上标。')
patch(34, r'\boldsymbol{x},\boldsymbol{o},E_j',
      '拼接后的网络输入、连续运行观测与第j类离散变量的嵌入矩阵（式3-4至3-6）',
      '与类别变量行分工，避免把第6章模型误差e_i解释为类别编码映射。')
patch(37, r'\mathcal L_{bc}', '行为克隆辅助项的未加权动作平方误差（式4-18）',
      '归一化动作已在原第9行说明，此处不重复列ã。')
patch(38, r'\mathcal Z,z,b(z),\rho(z),G_z',
      '候选运行模式集合、模式元素、依据历史的模式信念b(z)、冻结备份的模式权重ρ(z)及模式条件执行增益',
      '补入实际模式集合与正文信念写法，保留附录使用的ρ(z)。')
patch(54, r'\mathcal l_H,\tau_{IQL},\tau_e',
      'Huber损失及IQL期望分位水平；IQL在第2章记τ_e、第5章记τ_IQL，第6章τ_e另为预测误差尺度',
      '补足同形τ_e的两个明确作用域，不改变任何原公式。')
patch(55, r'\|\cdot\|_\infty,\|\cdot\|_2,\|\cdot\|_F',
      '最大范数、欧氏范数与矩阵Frobenius范数', '补入雅可比误差反复使用的Frobenius范数。')
patch(57, r'\tau,\tau_h',
      'τ为乘客积分时刻（第5章）或目标网络软更新系数（第3章）；τ_h为式3-11奖励衰减尺度，后续实验简写为τ',
      '删除未用预算别名，明确原文中τ的分域用法及奖励尺度别名，正文不改。')
patch(59, r'f_{sh},\varepsilon_{sh}',
      '跨环境共享真实映射与其相对于拟合符号模型的导数误差上界（附录D）',
      '拟合符号模型f_c_coef统一在原第73行说明，清理重复。')
patch(64, r'c_j,n_j,h,d_j,n,d',
      '类别取值、类别数、隐藏层宽度与嵌入维数；单类别分析以n、d分别记类别数与嵌入维数（第3章）',
      '按实际定义对应每个量，删除未用语义下标。')
plan[63]['correct_definition'] = plan[63]['correct_definition'].replace('第3.2.2节', '第3.2.3节')
patch(78, r'\mu_\theta(s)', 'DDPG中参数化的确定性策略（式2-9）',
      '删除不存在的混合零动作门控释义，保留实际确定性策略。')
patch(86, r'B,K_{\max},n_{rem}',
      '训练小批量大小、有限事件数上界及剩余事件数；附录E.5中B另为回归拟合映射的Lipschitz常数',
      '以B代替未用的B_batch，同时说明附录的合法局部复用。')
patch(85, r'\lambda_{ale},\lambda_{epi},\kappa_k,\kappa_k^{\prime},\kappa_{ale}',
      '范数项与集成方差项系数；第3章κ_k为目标网络权重范数，第4章κ_k、κ′_k分别为当前与目标网络范数；附录A的κ_ale为加权固定正则量',
      '按实际上下文区分在线、目标与加权固定范数项，不统一改动正文。')
patch(97, r'\mathcal l_{sym},\mathcal L_{JC},\bar{\mathcal L}_{JC}^{\prime}',
      '逐状态归一化雅可比导数误差、当前训练批次均值及后继状态批次均值（式4-15、4-16及D-4c至D-4e）',
      '采用实际均值下标、大小写及撇号，不再列旧的bar(l_sym)^+。')
patch(98, r'\zeta,\zeta(u)',
      '第2章CVaR的非负重加权密度；4.3.3节连接基准点与评价点的线段（式4-23），两者作用域不同',
      '删除不存在的连续分支潜变量释义。')

extra = []
def add(tex, definition, reason, after):
    extra.append({'source_row': None, 'action': 'add', 'original_symbol': '',
                  'original_definition': '', 'correct_tex': clean(tex),
                  'correct_definition': definition, 'reason': reason, 'after_row': after})

add(r'\mathcal L_Q,\mathcal L_\pi,\mathcal L_V',
    '动作价值、策略与状态价值的训练损失；各章按相应目标定义', '补入全书反复使用的训练损失记号。', 25)
add('K', '动作价值网络集成的成员数，各章按具体方法设置', '补入核心集成规模，区别于事件数上界。', 35)
for p in c46['proposed_new_rows']:
    after = 38 if p['id'] in ('N4-2', 'N4-3') else 97 if p['id'] in ('N4-1', 'N4-4') else 98
    if p['id'] == 'N4-6':
        after = 72
    elif p['id'] == 'N6-1':
        after = 45
    elif p['id'] == 'N6-2':
        after = 62
    add(p['correct_tex'], p['correct_definition'], p['reason'], after)

# Additional small-scope reviews, when present, provide actual notations for old experiment aliases.
supplement = A / 'supplement_table_patches.json'
if supplement.exists():
    for p in json.loads(supplement.read_text()):
        patch(p['row'], p['correct_tex'], p['correct_definition'], p['reason'])

rows = []
for number in range(1, 99):
    rows.append(plan[number])
    rows.extend(x for x in extra if x['after_row'] == number)
intro = ('数学记号按字母、上下标及字体识别。下标sim、tgt分别表示源仿真和目标环境。'
         '帽号表示估计或预测；目标网络的撇号及目标参数的横线按各节定义识别。'
         '星号表示相应问题的最优解或计划目标。同一字母在不同章节或局部证明中的含义依其作用域区分，'
         '求和、积分及定理中临时定义的局部变量不逐一列入。')
(A / 'symbol_table_plan.json').write_text(json.dumps({'intro': intro, 'rows': rows},
                                                   ensure_ascii=False, indent=2), encoding='utf-8')
print({name: sum(r['action'] == name for r in rows) for name in ('keep', 'update', 'delete', 'add')})
