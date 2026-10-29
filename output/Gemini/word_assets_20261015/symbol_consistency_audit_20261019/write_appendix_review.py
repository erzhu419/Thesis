"""Record the manually reviewed appendix definitions and native notation evidence."""
from pathlib import Path
from zipfile import ZipFile
from lxml import etree as E
import json

A = Path(__file__).resolve().parent
W = '{http://schemas.openxmlformats.org/wordprocessingml/2006/main}'
M = '{http://schemas.openxmlformats.org/officeDocument/2006/math}'
formulas = json.loads((A / 'formulas.json').read_text())
by_id = {x['math_id']: x for x in formulas}
paragraphs = json.loads((A / 'paragraphs.json').read_text())
symbols = json.loads((A / 'symbol_table.json').read_text())
with ZipFile(A / 'source_snapshot.docx') as z:
    root = E.fromstring(z.read('word/document.xml'))
maths = list(root.iter(M + 'oMath'))
tree = root.getroottree()


def text(n):
    return ''.join(t.text or '' for t in n.iter(M + 't'))


def loc(i, letters=None):
    result = {k: by_id[i][k] for k in ('math_id', 'paragraph', 'formula_number', 'tex')}
    result['mathXPath'] = tree.getpath(maths[i - 1])
    if letters is not None:
        result['native_runs'] = []
        for j, run in enumerate(maths[i - 1].iter(M + 'r'), 1):
            if text(run) not in letters:
                continue
            pr = run.find(M + 'rPr')
            result['native_runs'].append({
                'run': j, 'text': text(run), 'runXPath': tree.getpath(run),
                'math_properties': {E.QName(c).localname: c.get(M + 'val', '1') for c in pr} if pr is not None else {},
            })
    return result


def row(n):
    s = symbols[n - 1]
    return {'row': n, 'symbol': s['symbol'], 'definition': s['definition'],
            'locations': [loc(i) for i in s['math_ids']]}


findings = [
    {'id': 'APP-01', 'kind': 'symbol_table_unintroduced_alias', 'table': row(97),
     'finding': '表中后继状态批均值写作bar(l_sym)^+，但主文和附录D均明确定义为bar(L_JC)\'；大小写、下标与上标均不对应，且没有给出二者同义的定义。',
     'actual_notation': "\\bar{\\mathcal L}_{JC}'", 'actual_locations': [loc(i) for i in (1715, 1718, 4204, 4205, 4389)],
     'native_evidence': {'table': loc(120, {'l', 'sym', '+'}), 'D_4d': loc(4204, {'L', 'JC', "'"})},
     'minimal_correction': '将表97第二个符号改为正文及式(D-4d)实际使用的bar(L_JC)\'；不重命名正文公式。'},
    {'id': 'APP-02', 'kind': 'symbol_table_unintroduced_alias', 'table': row(86),
     'finding': '训练小批量大小在附录D明确写为B=256，式(D-4c)、(D-4d)均使用B；表列B_batch没有在本节引入为别名。',
     'actual_notation': 'B', 'actual_locations': [loc(4199), loc(4201), loc(4204)],
     'native_evidence': {'table': loc(109, {'B', 'batch'}), 'D_4d': loc(4204, {'B'})},
     'minimal_correction': '符号表采用B并注明训练小批量作用域；附录E.5中B为另行定义的回归拟合Lipschitz常数，不据此重命名正文。'},
    {'id': 'APP-03', 'kind': 'symbol_table_math_variant_mismatch', 'table': row(39),
     'finding': '同为模式推断所用的历史H_k，表39的H原生m:scr=sans-serif且m:sty=p；主文及附录B的H无m:scr、无m:sty，是默认数学字母。差异不是Pandoc跨组宏造成。',
     'actual_notation': 'H_k（默认数学字母）', 'actual_locations': [loc(i) for i in (1554, 3930, 3932, 3951, 3954)],
     'native_evidence': {'table': loc(50, {'H'}), 'B_29': loc(3930, {'H'}), 'B_definition': loc(3951, {'H'})},
     'minimal_correction': '让符号表H的数学字体与已定义并实际使用的H_k一致。'},
    {'id': 'APP-04', 'kind': 'symbol_table_math_variant_mismatch', 'table': row(83),
     'finding': '概率残差实向量空间在E.5和式(E-27)实际为X_delta（X原生无m:scr）；表83却将同一X设m:scr=script，即mathcal X_delta。字母角色和下标相同，数学字体不一致。',
     'actual_notation': 'X_delta（默认数学字母）', 'actual_locations': [loc(4573), loc(4574)],
     'native_evidence': {'table': loc(106, {'X'}), 'E_definition': loc(4573, {'X'}), 'E_27': loc(4574, {'X'})},
     'minimal_correction': '让表83的X字体与附录E.5现用X_delta一致。'},
    {'id': 'APP-05', 'kind': 'repeated_core_symbol_missing_from_table', 'related_table_rows': [18, 32, 38],
     'finding': '候选运行模式集合mathcal Z在主文和附录C、D反复出现，但主要符号表只列模式元素z、信念rho和增益G_z，未收该模式集合。表32的带beta_tail下标Z是风险包络，并不能代替模式集合条目。',
     'actual_notation': '\\mathcal Z（候选模式集合）',
     'actual_locations': [loc(i) for i in (1573, 1580, 4021, 4029, 4034, 4042, 4043, 4368)],
     'minimal_correction': '在模式符号行补入mathcal Z及有限候选模式集合含义；保留与风险包络Z_beta_tail的区别。'},
]

accepted = [
    {'table_rows': [27, 28, 29, 30, 31, 32, 42], 'scope': '附录A',
     'finding': 'alpha_ent与正文alpha在P3386明确同义；支持函数、Wasserstein半径ipm、轨迹成本traj及CVaR的tail/risk下标均实际使用。kappa_ale在式(A-2)明确定义为含lambda_ale的标量，不与表85未加权kappa_j强行等同。',
     'locations': [loc(i) for i in (3642, 3665, 3733, 3753, 3769, 3771, 3772)]},
    {'table_rows': [66], 'scope': '附录B.4',
     'finding': 'H_obs、h_obs、P_obs,1和P_obs,2确为原文所用，不是表中新加别名；式(B-24)至(B-29)完整沿用。',
     'actual_locations': [loc(i) for i in (3913, 3917, 3918, 3919, 3920, 3930)]},
    {'table_rows': [74, 75, 76, 88, 89], 'scope': '附录B',
     'finding': 'exec/nom/ref/beh/act/phys/aug及路径上标O,path均有实际定义；候选脚本对V^{O,path}/Q^{O,path}的零匹配是因实际还带n,k下标，不构成缺用。',
     'actual_locations': [loc(i) for i in (3801, 3825, 3851, 3930, 3964, 3965, 3985, 4015)]},
    {'table_rows': [38, 40, 58, 68], 'scope': '附录C',
     'finding': 'rho、Omega_prior、O_belief,n、c_fb与delta_post实际使用；先验旧/新与后验新/旧比的方向在P3937明示互为倒数，表40定义相符。',
     'actual_locations': [loc(i) for i in (4025, 4135, 4137, 4138, 4141, 4148)]},
    {'table_rows': [59, 69, 70, 71, 72, 73, 90, 91, 92], 'scope': '附录D',
     'finding': 'f_sh、f_c_coef、c_coef/i、eps_val/der/path、X_eval/in/out、e_map/path、eps_sym/pol/IRM/emp和Delta_gen均实际出现且与表义相符；mu_det,theta的det宏包装不同不代表没用。',
     'actual_locations': [loc(i) for i in (4164, 4186, 4241, 4254, 4262, 4288, 4416, 4417, 4420, 4436, 4442, 4447, 4451, 4455)]},
    {'table_rows': [69, 87, 97], 'scope': '附录D.1、D.4',
     'finding': 'P3985–3995明确给出mu_det,theta与正文tilde a、f_ref与符号参考动作、J_mu/J_f与正文Dpi/Dpi_sym的对应；NAU权重c_NAU,i在(D-30)–(D-31)实际使用。没有把NAU性质套给不含NAU配置。',
     'actual_locations': [loc(i) for i in (4173, 4185, 4193, 4194, 4195, 4196, 4306, 4309)]},
    {'table_rows': [47, 48, 49, 50, 51, 53, 83, 86], 'scope': '附录E',
     'finding': 'eps_P/r/V、L_adj/occ、K_max/n_rem、gamma_Delta、c_const/c_min^Delta、原及归一化T_Delta/Q^Delta均与定义相符。gamma=1的有限事件界与gamma_Delta<1的置信递推明确分开。表83的X字体差异单列APP-04。',
     'actual_locations': [loc(i) for i in (4500, 4513, 4521, 4531, 4593, 4651, 4661, 4668, 4678, 4693)]},
    {'scope': '附录E.1与正文状态残差',
     'finding': 'hat delta是转移概率加性残差，表45的hat Delta_s/r是连续状态/奖励预测残差；P4319已明确二者对象不同，属于有说明的不同记号，不能互相替换。',
     'paragraphs': [4319], 'actual_locations': [loc(4471), loc(4473)]},
    {'scope': '附录各局部证明',
     'finding': 'A中的f、D中的a/b/n/m/g和E.6中的e_real/e_resid/w_rank/tau_rank等均在相应命题或小节局部定义；求和索引和局部证明变量的复用不作为一符多义或核心漏收。D批量B与E.5回归映射Lipschitz常数B分属明确作用域。',
     'paragraphs': [99, 3999, 4008, 4436, 4442]},
    {'scope': '附录F', 'finding': '仅实验预算、需求倍率、源样本权重w_sim、成本及置信区间等局部量；未发现与表内定义冲突的数学记号。',
     'paragraphs': [4530, 4531, 4535, 4536, 4569, 4570, 4571]},
]
coverage = []
for letter in 'ABCDEF':
    ms = [f for f in formulas if f['chapter'].startswith('附录' + letter)]
    ps = [p for p in paragraphs if p['chapter'].startswith('附录' + letter)]
    coverage.append({'appendix': letter, 'math_objects_read': len(ms),
                     'math_id_first_last': [ms[0]['math_id'], ms[-1]['math_id']],
                     'paragraph_first_last': [ps[0]['paragraph'], ps[-1]['paragraph']]})
report = {'read_only': True, 'source': str((A / 'source_snapshot.docx').resolve()),
          'scope': '主要符号表98行与附录A–F全文数学定义；必要处核正文对应定义及原生OMML。',
          'coverage': coverage, 'findings': findings, 'accepted': accepted,
          'handoff_to_other_scopes': ['表39的K_z实际主文采用Psi(z\'|z)，由第4–6章审查合并。',
                                      '表67的dual对应第2章delta/rho/V及特征向量定义问题、表71iter、表77learn由第1–6章范围复核合并，不在此重复制造附录缺用问题。'],
          'changes_made_to_thesis': False}
(A / 'review_appendices.json').write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n')
print(json.dumps({'appendices_read': 6, 'appendix_math_objects': sum(x['math_objects_read'] for x in coverage),
                  'findings': len(findings), 'accepted_groups': len(accepted)}, ensure_ascii=False))
