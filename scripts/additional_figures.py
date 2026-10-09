"""Chinese result figures derived solely from completed local experiments."""

import csv
import json
from collections import defaultdict

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np


def mcwm_risk_control(b):
    source = 'MC-WM/paper/generated/final_exact_routing/exact_routing_counterfactuals.csv'
    data = b.rows(source, 'mcwm_exact_paired.csv')
    summaries = []
    fig, axes = plt.subplots(3, 2, figsize=(5.75, 6.1), sharex='row')
    fig.subplots_adjust(left=.13, right=.97, bottom=.17, top=.92,
                        hspace=.82, wspace=.42)
    for row_index, (env, label) in enumerate(b.ENVIRONMENTS):
        rows = sorted((r for r in data if r['environment'] == env),
                      key=lambda r: int(r['seed']))
        assert len(rows) == len({r['seed'] for r in rows}) == 10
        assert all(r['selected_arm_metrics_exact'] == 'True' for r in rows)
        risk = np.array([float(r['risk_margin']) for r in rows])
        selected = np.array([float(r['selected_risk']) for r in rows])
        alternative = np.array([float(r['alternative_risk']) for r in rows])
        np.testing.assert_allclose(risk, alternative - selected, rtol=0, atol=1e-12)
        returns = np.array([float(r['automatic_minus_alternative_return']) for r in rows])
        violations = np.array([float(r['automatic_minus_alternative_violations']) for r in rows])
        family = rows[0]['selected_family']
        assert all(r['selected_family'] == family for r in rows)
        summaries.append({
            'environment': env, 'selected_family': family, 'n_training_seeds': len(rows),
            'risk_margin_mean': float(risk.mean()),
            'risk_margin_range': [float(risk.min()), float(risk.max())],
            'return_higher_equal_lower': [int((returns > 0).sum()),
                                          int((returns == 0).sum()), int((returns < 0).sum())],
            'violation_lower_equal_higher': [int((violations < 0).sum()),
                                             int((violations == 0).sum()), int((violations > 0).sum())],
            'per_seed': [{
                'seed': int(r['seed']), 'selected_risk': float(r['selected_risk']),
                'alternative_risk': float(r['alternative_risk']),
                'risk_margin': float(r['risk_margin']),
                'return_difference': float(r['automatic_minus_alternative_return']),
                'violation_difference': float(r['automatic_minus_alternative_violations']),
            } for r in rows],
        })
        for col, (values, good, ylabel) in enumerate([
            (returns / 1000, returns > 0, '回报差 / 千'),
            (violations, violations < 0, '每回合违例数差'),
        ]):
            ax = axes[row_index, col]
            colors = [b.GRAY if v == 0 else b.GREEN if g else b.RED
                      for v, g in zip(values, good)]
            ax.scatter(risk, values, c=colors, s=27, edgecolor='white',
                       linewidth=.45, zorder=3)
            ax.axhline(0, color=b.GRAY, ls='--', lw=.8)
            ax.set(xlim=(0, float(risk.max()) * 1.13), ylabel=ylabel)
            ax.margins(y=.20)
            ax.xaxis.set_major_locator(b.matplotlib.ticker.MaxNLocator(4))
            ax.yaxis.set_major_locator(b.matplotlib.ticker.MaxNLocator(4))
            ax.grid(lw=.5)
            ax.set_title(f'({chr(97 + row_index * 2 + col)}) {label}',
                         loc='left', fontsize=9)
    for ax in axes[-1]:
        ax.set_xlabel('校准集预测误差之差', fontsize=9)
    fig.legend(handles=[Line2D([], [], marker='o', ls='', color=c, markersize=4, label=t)
                        for c, t in [(b.GREEN, '该指标改善'), (b.RED, '该指标退化'),
                                     (b.GRAY, '该指标相同')]],
               loc='lower center', ncol=3, fontsize=8.5, bbox_to_anchor=(.55, .005))
    (b.DATA / 'mcwm_risk_control.json').write_text(json.dumps({
        'source': source, 'unit': 'independent training seed; 20 test episodes averaged per policy',
        'analysis': 'descriptive paired outcomes; no fitted correlation or new routing threshold',
        'environments': summaries,
    }, ensure_ascii=False, indent=2) + '\n')
    b.save(fig, 'fig6_model_calibration',
           '较低校准集预测误差与控制收益不逐种子等价，Ant存在平均回报收益', [source],
           '每任务10个配对训练种子；纵轴为所选减未选类别的20回合测试均值差；'
           '横轴为含状态与奖励的标准化校准集预测误差之差；同任务两列横轴一致，跨任务不合并；无拟合线')


def transit_resources_and_correlation(b):
    source = 'TransitDuet/transit_duet/results_remote/revision3_heldout/main_results_heldout.csv'
    with (b.PROJECTS / source).open() as handle:
        data = list(csv.DictReader(handle))
    names = {'H_timetable_v5_stable_300': 'TransitDuet', 'H_hiro_final': '仅目标传递',
             'H_haar_final': '强化学习代理', 'H_fixed_timetable_300_final': '固定300秒',
             'H_fixed_timetable_360_final': '固定360秒',
             'baseline_rule_daganzo': 'Daganzo型代理', 'baseline_rule_xuan': 'Xuan型代理'}
    fig, axes = b.pair_panels([names[r['method']] for r in data], height=4.5, left=.28)
    for ax, key, label in zip(axes, ['overshoot', 'avg_holding_sec'],
                              ['(a) 峰值车辆超额', '(b) 末次班次停站']):
        for i, row in enumerate(data):
            ax.errorbar(float(row[key + '_mean']), i,
                        xerr=float(row[key + '_seed_sd']), fmt='o',
                        color=b.BLUE if i == 0 else b.GRAY, capsize=2, ms=4)
        ax.set_title(label, fontsize=9)
    axes[0].set_xlabel('车辆数')
    axes[1].set_xlabel('时间 / 秒')
    b.save(fig, 'fig4_transit_resources', '峰值车辆超额与末次班次停站分别描述资源和局部执行',
           [source], '10组种子，每组50个独立测试回合；均值与种子组均值标准差；规则无训练重复；停站非全日均值')

    correlations, sources = [], []
    for seed in [42, 123, 456, 789, 1001, 1002, 1003, 1004, 1005, 1006]:
        path = f'TransitDuet/transit_duet/logs/H_timetable_v5_stable_300_seed{seed}/diagnostics.csv'
        sources.append(path)
        with (b.PROJECTS / path).open() as handle:
            rows = [r for r in csv.DictReader(handle) if float(r['ep']) < 9000][-30:]
        for row in rows:
            correlations.append([seed, float(row['planned_dispatch_headway_std']),
                                 float(row['headway_cv']), float(row['actual_dispatch_headway_cv'])])
    d = np.array(correlations)
    assert len(d) == 300
    (b.DATA / 'transit_planning_correlation.json').write_text(
        json.dumps(correlations, ensure_ascii=False, indent=2) + '\n')
    fig, axes = plt.subplots(1, 2, figsize=(5.75, 3.35))
    fig.subplots_adjust(left=.13, right=.97, bottom=.23, top=.87, wspace=.42)
    for ax, col, title, ylabel in zip(axes, [2, 3], ['(a) 终点时距横截面', '(b) 全日实际发车'],
                                      ['终点时距变异系数', '实际发车时距变异系数']):
        selected = d[(d[:, 1] > 0) & (d[:, col] >= 0)]
        x, y = selected[:, 1], selected[:, col]
        ax.scatter(x, y, s=8, alpha=.35, color=b.BLUE, rasterized=True)
        slope, intercept = np.polyfit(x, y, 1)
        xx = np.array([x.min(), x.max()])
        ax.plot(xx, slope * xx + intercept, color=b.RED, lw=1)
        ax.text(.05, .91, f'r = {np.corrcoef(x, y)[0, 1]:.3f}', transform=ax.transAxes, fontsize=8.5)
        ax.set(title=title, ylabel=ylabel)
        ax.set_xlabel('计划时距标准差 / 秒', fontsize=8.5)
        ax.grid(color='#E8E8E8', linewidth=.6)
    b.save(fig, 'fig4_transit_correlation', '计划波动与全日发车相关较强，与终点时距横截面相关较弱',
           sources, '10个训练种子各最后30个训练回合；300点作描述性相关，不视为300次独立训练；左图非全日站点CV')


def resac_training_curves(b):
    src='RE-SAC/results/'
    configs=[('sac','SAC','baseline_sac/rewards.npy'),
             *[(f'resac_K{k}',f'K={k}',f'ensemble_{k}/logs/rewards.npy') for k in [2,5,10,20,40]],
             ('aleatoric_only','仅权重范数','aleatoric_only/logs/rewards.npy'),
             ('epistemic_only','仅集成分歧','epistemic_only/logs/rewards.npy')]
    curves={}
    sources=[]
    for key,label,first in configs:
        paths=[src+f'bus_seed_extension/logs/{key}_seed{s}/rewards.npy' for s in [32,40]]
        values=np.stack([np.load(b.PROJECTS/p) for p in paths])/1000
        assert values.shape==(2,500), (key,values.shape)
        smoothed=values.copy()
        for t in range(1,500):smoothed[:,t]=.95*smoothed[:,t-1]+.05*values[:,t]
        curves[key]={'label':label,'mean':smoothed.mean(0).tolist(),'sd':smoothed.std(0,ddof=1).tolist()}
        sources.extend(paths)
    (b.DATA / 'resac_training_curves.json').write_text(
        json.dumps(curves, ensure_ascii=False, indent=2) + '\n')
    fig,axes=plt.subplots(2,1,figsize=(5.75,5.4),sharex=True)
    fig.subplots_adjust(left=.15,right=.97,bottom=.11,top=.94,hspace=.40)
    colors = {'sac': b.GRAY, 'resac_K2': b.GOLD, 'resac_K5': b.PURPLE,
              'resac_K10': b.BLUE, 'resac_K20': b.GREEN, 'resac_K40': '#B2754B',
              'aleatoric_only': b.GOLD, 'epistemic_only': b.RED}
    for ax,keys,title in [(axes[0],[x[0] for x in configs[:6]],'(a) 集成规模'),(axes[1],['sac','resac_K10','aleatoric_only','epistemic_only'],'(b) 惩罚通道消融')]:
        for key in keys:
            c = colors[key]
            d=curves[key];m=np.array(d['mean']);sd=np.array(d['sd'])
            ax.plot(np.arange(1,501),m,color=c,lw=1,label=d['label'])
            ax.fill_between(np.arange(1,501),m-sd,m+sd,color=c,alpha=.09,lw=0)
        ax.set(title=title,ylabel='回合回报 / 千')
        ax.grid(color='#E8E8E8', linewidth=.6)
        ax.legend(ncol=3,fontsize=8,frameon=False,loc='lower right')
    axes[-1].set_xlabel('训练回合')
    b.save(fig,'fig3_training_curves','相同演员宽度下比较集成规模与惩罚通道',sources,
           '扩展种子32、40；演员隐藏宽度64；各序列采用0.95系数指数平滑后汇总；均值及样本标准差；不混入32宽度历史SAC')


def render(b):
    def read(rel):
        with (b.PROJECTS / rel).open() as f:
            return list(csv.DictReader(f))

    def snapshot(name, data):
        (b.DATA / (name+'.json')).write_text(json.dumps(data, ensure_ascii=False, indent=2)+'\n')

    def grid(ax):
        ax.grid(color='#E8E8E8', linewidth=.6)
        ax.set_axisbelow(True)

    def finish(fig, name, claim, sources, stats):
        b.save(fig, name, claim, sources, stats)

    def interval(ax, y, summary, color=b.BLUE, scale=1):
        mean = summary['mean']/scale
        lo,hi = np.asarray(summary['bootstrap_95_ci'])/scale
        ax.errorbar(mean,y,xerr=[[mean-lo],[hi-mean]],fmt='o',color=color,ms=4,capsize=2)

    resac_training_curves(b)

    root='RE-SAC/jax_experiments/diagnostics/bus_parallel_followup_20260920/temperature/'
    temp={};sources=[]
    for arm in ['adaptive','frozen']:
        grouped=defaultdict(lambda:defaultdict(list))
        for seed in range(7800,7805):
            path=root+f'{arm}_{seed}/evaluations.csv';sources.append(path)
            for r in read(path):
                grouped[int(r['day'])][seed].append([float(r['environment_reward'])/1000,float(r['reference_alpha'])])
        temp[arm]=[]
        for day,values in sorted(grouped.items()):
            v=np.array([np.mean(x,axis=0) for x in values.values()]);assert len(v)==5
            temp[arm].append([day,*v.mean(0),*v.std(0,ddof=1)])
    snapshot('temperature_trajectory',temp)
    fig,axes=plt.subplots(2,1,figsize=(5.75,4.6),sharex=True)
    fig.subplots_adjust(left=.15,right=.97,bottom=.13,top=.91,hspace=.40)
    for arm,label,c in [('adaptive','自适应温度',b.GRAY),('frozen','第100日后冻结温度',b.BLUE)]:
        d=np.array(temp[arm]);d=d[d[:,0]>=50]
        for ax,col in zip(axes,[1,2]):
            ax.plot(d[:,0],d[:,col],label=label,color=c,lw=1.2)
            ax.fill_between(d[:,0],d[:,col]-d[:,col+2],d[:,col]+d[:,col+2],color=c,alpha=.13,lw=0)
    for ax in axes:ax.axvline(100,color=b.RED,ls=':',lw=1);grid(ax)
    axes[0].set(title='(a) 中后期策略评价',ylabel='评价回报 / 千')
    axes[1].set(title='(b) 熵温度变化',ylabel='熵温度',xlabel='训练运行日')
    fig.legend(*axes[0].get_legend_handles_labels(),ncol=2,loc='upper center',frameon=False,fontsize=9)
    finish(fig,'fig3_temperature_trajectory','冻结温度后策略变化趋缓但仍存在后期回报下降',sources,'5个配对训练种子；先平均同一种子的重复评价，再计算均值与样本标准差；仅展示第50日起')

    source='CS-BAPR/paper/source_data/reported_tables.csv';data=read(source)
    methods=[('bapr-pristine','原始BAPR'),('bapr','早期BAPR'),('csbapr-no-nau','ReLU'),('csbapr','NAU/NMU'),('csbapr-tanh','tanh'),('csbapr-kan','KAN')]
    def val(table,method,metric):
        return float(next(r['value'] for r in data if r['table']==str(table) and r['method']==method and r['metric']==metric))
    fig,ax=plt.subplots(figsize=(5.75,3.6));fig.subplots_adjust(left=.24,right=.97,bottom=.18,top=.95)
    for i,((m,l),c) in enumerate(zip(methods,b.COLORS)):
        ax.errorbar(val(1,m,'ID total return mean'),i,xerr=val(1,m,'ID total return sd'),fmt='o',capsize=3,color=c)
    ax.set(yticks=range(6),yticklabels=[l for m,l in methods],ylim=(5.6,-.6),xlabel='训练分布内回报 / 千');grid(ax)
    finish(fig,'fig4_id_results','训练分布内结果与极端需求下的结构排序不同',[source],'10个种子，均值及标准差')

    from mechanism_figures import csbapr_contrast
    csbapr_contrast(b)

    thresholds=['< -1200K crash rate','< -1500K crash rate','< -1800K crash rate','< -2000K crash rate']
    matrix=np.array([[val(5,m,t) for t in thresholds] for m,l in methods])
    fig,axes=plt.subplots(1,2,figsize=(5.75,4),gridspec_kw={'width_ratios':[1.45,1]})
    fig.subplots_adjust(left=.23,right=.97,bottom=.25,top=.87,wspace=.27)
    im=axes[0].imshow(matrix,cmap='YlOrRd',vmin=0,vmax=100,aspect='auto')
    for (i,j),v in np.ndenumerate(matrix):axes[0].text(j,i,f'{v:.0f}%',ha='center',va='center',fontsize=8,color='white' if v>=60 else '#222222')
    axes[0].set(xticks=range(4),xticklabels=['−1200','−1500','−1800','−2000'],yticks=range(6),yticklabels=[l for m,l in methods],xlabel='回报阈值 / 千',title='(a) 低于阈值的种子比例')
    axes[0].tick_params(axis='x',labelrotation=50,labelsize=8)
    for i,((m,l),c) in enumerate(zip(methods,b.COLORS)):axes[1].scatter(val(5,m,'worst seed'),i,color=c,s=26)
    axes[1].set(yticks=[],ylim=(5.5,-.5),xlabel='最差回报 / 千',title='(b) 最差种子')
    axes[0].set_title('(a) 低回报比例',fontsize=9);axes[1].set_title('(b) 最差种子',fontsize=9);grid(axes[1])
    finish(fig,'fig4_tail_risk','降低常见低回报比例不保证排除严重尾部损失',[source],'20倍需求，10个种子；阈值比例与最差单种子回报并列，不由均值推断尾部')

    transit_resources_and_correlation(b)

    ablations=[['滚动计划',6.05,1.634],['仅目标传递',6.58,1.687],['最优固定间隔',6.90,1.846],['扩大调整范围',6.18,1.760],['连续发车动作',7.70,1.856],['去除停站上下文',9.45,2.048],['独立训练后组合',7.27,1.929]]
    snapshot('transit_training_ablations',ablations)
    fig,axes=b.pair_panels([r[0] for r in ablations],height=4.1,left=.29)
    for ax,col,title,xlabel in zip(axes,[1,2],['(a) 平均等待时间','(b) 复合运行成本'],['时间 / 分钟','归一化复合成本']):
        for i,r in enumerate(ablations):ax.scatter(r[col],i,color=b.BLUE if i==0 else b.GRAY,s=25)
        ax.set(title=title,xlabel=xlabel);ax.set_title(title,fontsize=9)
    finish(fig,'fig4_transit_ablation','去除停站上下文或独立组合策略使训练后期表现下降',['TransitDuet/paper/sections/5_experiments.tex, timetable ablation'],'训练后期窗口汇总，与独立测试分开；原表未提供跨种子离散程度，不添加误差线')

    source='offline-sumo/paper/generated/figure2_learning_curves_data.csv';data=read(source)
    fig,ax=plt.subplots(figsize=(5.75,3.9));fig.subplots_adjust(left=.15,right=.97,bottom=.24,top=.96)
    for method,c in zip(dict.fromkeys(r['method'] for r in data),b.COLORS):
        rr=sorted([r for r in data if r['method']==method],key=lambda r:float(r['epoch']))
        x=np.array([float(r['epoch'])/10 for r in rr]);m=np.array([float(r['return_mean'])/1000 for r in rr]);sd=np.array([float(r['return_std'])/1000 for r in rr])
        ax.plot(x,m,marker='o',ms=3,lw=1,color=c,label=method.replace('OnlineSAC','在线SAC'))
        ax.fill_between(x,m-sd,m+sd,color=c,alpha=.1,lw=0)
    ax.set(xlabel='新增目标控制决策 / 千次',ylabel='完整回合回报 / 千');grid(ax)
    fig.legend(*ax.get_legend_handles_labels(),loc='lower center',ncol=3,frameon=False,fontsize=8.5)
    finish(fig,'fig5_offline_learning','不同初始化与混合学习方式随目标交互预算呈现不同学习轨迹',[source],'5个训练种子，各种子平均10个完整18,000秒SUMO回合；均值与跨种子标准差')

    source='offline-sumo/paper/generated/figure4_corrected_return_data.csv';data=read(source)
    groups=defaultdict(list)
    for r in data:
        assert not r['error'] and float(r['sim_time'])>=18000
        groups[(r['method'],r['checkpoint_label'],r['train_seed'])].append(float(r['return']))
    per_checkpoint=defaultdict(list)
    for (method,label,seed),v in groups.items():per_checkpoint[(method,label)].append(float(np.mean(v)))
    summary={}
    for method,label in per_checkpoint:
        v=per_checkpoint[method,label];summary.setdefault(method,[]).append([label,np.mean(v),np.std(v,ddof=1),len(v)])
    snapshot('offline_sensitivity',summary)
    fig,axes=plt.subplots(2,1,figsize=(5.75,5.3),sharex=True)
    fig.subplots_adjust(left=.15,right=.97,bottom=.13,top=.94,hspace=.40)
    for ax,methods,title in [(axes[0],['RLPD-0.25','RLPD-0.50','RLPD-0.75'],'(a) 离线回放比例'),(axes[1],['RLPD-E10','RLPD-E10-UTD4'],'(b) 十评论家配置的更新数据比')]:
        for method,c in zip(methods,b.COLORS):
            rr=sorted(summary[method],key=lambda r:10000 if r[0]=='final' else int(r[0]));rr=[r for r in rr if r[0]!='final']
            x=np.array([int(r[0])/10 for r in rr]);m=np.array([r[1]/1000 for r in rr]);sd=np.array([r[2]/1000 for r in rr])
            ax.plot(x,m,marker='o',ms=3,lw=1,color=c,label=method+f' (n={rr[0][3]})');ax.fill_between(x,m-sd,m+sd,color=c,alpha=.13,lw=0)
        ax.set(title=title,ylabel='回合回报 / 千');grid(ax);ax.legend(frameon=False,fontsize=8,ncol=2,loc='lower right')
    axes[-1].set_xlabel('新增目标控制决策 / 千次')
    finish(fig,'fig5_offline_sensitivity','增加离线比例或更新次数的收益有限且伴随种子差异',[source],'按训练种子先汇总SUMO环境回合；均值与样本标准差；n=3或5见图例；仅绘数值检查点，避免重复绘制final别名')

    source='offline-sumo/paper/generated/figure3_operational_tradeoff_data.csv';data=read(source)
    fig,ax=plt.subplots(figsize=(5.75,4.6));fig.subplots_adjust(left=.14,right=.97,bottom=.40,top=.96)
    for i,r in enumerate(data):
        learned=r['controller_group']=='Learned controllers';c=b.BLUE if learned else b.GREEN
        x=float(r['headway_abs_dev_mean_s_mean']);y=float(r['passenger_total_mean_s_mean'])
        ax.scatter(x,y,color=c,marker='o' if learned else 's',s=24)
        ax.annotate(str(i+1),(x,y),xytext=(4,4),textcoords='offset points',fontsize=7.5)
    ax.set(xlabel='车头时距绝对偏差 / 秒',ylabel='平均乘客总时间 / 秒');grid(ax)
    rule_names={'OnlineSAC':'在线SAC','Adaptive headway':'自适应间隔','Daganzo-inspired':'Daganzo型规则','NoHold':'零停站','Reference heuristic':'参考启发式','Schedule holding':'计划停站'}
    labels=[f"{i+1}  {rule_names.get(r['display_method'],r['display_method'])}" for i,r in enumerate(data)]
    handles=[Line2D([],[],color=b.BLUE if r['controller_group']=='Learned controllers' else b.GREEN,marker='o' if r['controller_group']=='Learned controllers' else 's',lw=0,ms=4) for r in data]
    fig.legend(handles,labels,loc='lower center',ncol=3,frameon=False,fontsize=7.5,columnspacing=1)
    finish(fig,'fig5_operational_frontier','学习控制与规则控制在规律性和乘客时间之间呈现不同权衡',[source],'各控制器的汇总均值；学习控制器n=5训练种子，规则控制器n=10环境种子；描述性散点不用于显著性判断')

    source='sumo-rl/H2Oplus/reproducibility/h2oplus_v20_formal_confirmation_results/formal_confirmation_analysis_v1.json'
    h=json.loads((b.PROJECTS/source).read_text())['formal_confirmation_scientific_result']
    names=[('h2oplus-v19-completion-safety-quarter-load-sixty','完成保障控制'),('h2oplus-v19-completion-safety-baseline','直接参照控制'),('h2oplus-v14-terminal-quarter','未加执行变换'),('h2oplus','基础H2O+'),('sac-online','在线SAC'),('h2oplus-v10-delayed-actor','延迟演员更新'),('h2oplus-v11-actor-hold','冻结演员续训'),('h2oplus-v12-actor-ramp','渐进演员更新'),('h2oplus-v13-proximal-strong','近端约束续训')]
    snapshot('h2o_confirmatory_means',{'means':h['algorithm_means'],'blocks':h['candidate_vs_direct_parent_by_formal_environment_block']})
    fig,axes=b.pair_panels([n for m,n in names],height=4.7,left=.28)
    for ax,key,mult,title,xlabel in zip(axes,['primary','completion'],[1,100],['(a) 每出发乘客时间','(b) 行程完成率'],['广义时间 / 秒','完成率 / %']):
        for i,(method,label) in enumerate(names):ax.scatter(h['algorithm_means'][method][key]*mult,i,s=25,color=b.BLUE if i==0 else b.GRAY)
        ax.set(title=title,xlabel=xlabel);ax.set_title(title,fontsize=9)
    finish(fig,'fig5_h2o_controllers','完成保障在统一评价中同时减少乘客时间并提高完成率',[source],'9个控制器的汇总均值；5个训练种子、10个独立环境块与3个需求水平；此图不将150个配对单元当作独立样本')
    blocks=list(h['candidate_vs_direct_parent_by_formal_environment_block'].values())
    fig,ax=plt.subplots(figsize=(5.75,3.5));fig.subplots_adjust(left=.15,right=.97,bottom=.19,top=.94)
    label_offsets=[(-13,3),(5,-11),(7,-12),(5,6),(5,-11),(5,5),(5,-11),(7,9),(-15,-9),(-15,6)]
    for i,r in enumerate(blocks):
        x=-r['primary_change_seconds'];y=100*r['completion_change']
        ax.scatter(x,y,color=b.BLUE,s=35,edgecolor='white',lw=.6,zorder=3)
        ax.annotate(str(i+1),(x,y),xytext=label_offsets[i],textcoords='offset points',fontsize=8)
    ax.axhline(0,color=b.GRAY,lw=.8);ax.axvline(0,color=b.GRAY,lw=.8)
    ax.set(xlabel='每出发乘客广义时间减少 / 秒',ylabel='行程完成率提高 / 百分点',xlim=(78,112),ylim=(.40,.56))
    ax.text(.04,.93,'数字表示独立环境块',transform=ax.transAxes,fontsize=8,color=b.GRAY)
    grid(ax)
    finish(fig,'fig5_h2o_environment_blocks','服务改善方向在10个独立环境块上一致',[source],'先在每个环境块内汇总5个训练种子和3个需求水平，再展示配对均值；不添加块内伪独立误差线')

    methods=[('mcwm_calibration_risk_target_done','校准选择MC-WM'),('mcwm_core','固定残差核心'),('direct_target_mbpo','直接目标MBPO'),('target_sac','目标SAC'),('darc','DARC'),('raw_sim','原始源策略')]
    source='MC-WM/paper/generated/final_confirmation/algorithm_confirmation_results.json';result=json.loads((b.PROJECTS/source).read_text())
    fig,axes=plt.subplots(3,2,figsize=(5.75,6.4));fig.subplots_adjust(left=.28,right=.97,bottom=.09,top=.94,hspace=.55,wspace=.3)
    for i,(env,label) in enumerate(b.ENVIRONMENTS):
        for ax,key in zip(axes[i],['return','violations_per_episode']):
            for y,(method,name) in enumerate(methods):interval(ax,y,result['environment_results'][env]['endpoints'][method][key],b.BLUE if y==0 else b.GRAY)
            ax.set(ylim=(5.6,-.6),yticks=range(6));grid(ax)
            ax.xaxis.set_major_locator(b.matplotlib.ticker.MaxNLocator(3))
        axes[i,0].set_yticklabels([name for method,name in methods],fontsize=8)
        axes[i,1].set_yticklabels([]);axes[i,0].set_title(f'({chr(97+i)}) {label}',loc='left',fontsize=9)
    axes[-1,0].set_xlabel('留出测试回报');axes[-1,1].set_xlabel('每回合违例数')
    finish(fig,'fig6_final_endpoints','独立测试同时比较策略回报与约束违例',[source],'每环境每方法10个训练种子；均值及已计算的95%自助法区间；各环境分别设置回报坐标范围')

    source='MC-WM/paper/generated/final_confirmation/learning_curve_summary.csv';data=read(source)
    fig,axes=plt.subplots(3,1,figsize=(5.75,6.0),sharex=True);fig.subplots_adjust(left=.15,right=.97,bottom=.21,top=.95,hspace=.42)
    for i,(ax,(env,label)) in enumerate(zip(axes,b.ENVIRONMENTS)):
        for (method,name),c in zip(methods,b.COLORS):
            rr=sorted([r for r in data if r['environment']==env and r['method']==method],key=lambda r:int(r['online_step']))
            x=b.mcwm_progress(rr,method)
            m,lo,hi=[np.array([float(r[k]) for r in rr]) for k in ['violation_mean','violation_ci_low','violation_ci_high']]
            ax.plot(x,m,color=c,lw=1,label=name);ax.fill_between(x,lo,hi,color=c,alpha=.10,lw=0)
        ax.set(title=f'({chr(97+i)}) {label}',ylabel='每回合违例数');grid(ax)
    axes[-1].set(xlabel='预设训练预算完成比例 / %',xticks=[25,50,75,100],xlim=(10,102))
    fig.legend(*axes[0].get_legend_handles_labels(),loc='lower center',ncol=2,frameon=False,fontsize=9)
    finish(fig,'fig6_violation_learning','控制适应过程中的约束违例需与回报同步观察',[source],'10个训练种子，验证均值及95%自助法区间；8个预设检查点按在线适应40000目标步或纯源训练50000源步分别归一化；不含验证成本，未插值')

    source='MC-WM/paper/generated/final_confirmation/target_interaction_ledger_summary.csv';data=read(source)
    fig,axes=plt.subplots(3,1,figsize=(5.75,6.0),sharex=True);fig.subplots_adjust(left=.28,right=.97,bottom=.14,top=.95,hspace=.55)
    for i,(ax,(env,label)) in enumerate(zip(axes,b.ENVIRONMENTS)):
        rr=[next(r for r in data if r['environment']==env and r['method']==m) for m,n in methods];left=np.zeros(6)
        for key,name,c in [('real_training_steps_mean','训练',b.BLUE),('real_validation_steps_mean','验证',b.GREEN),('real_test_steps_mean','测试',b.GOLD)]:
            v=np.array([float(r[key])/1000 for r in rr]);ax.barh(range(6),v,left=left,height=.65,color=c,label=name);left+=v
        ax.set(yticks=range(6),yticklabels=[n for m,n in methods],ylim=(5.6,-.6),title=f'({chr(97+i)}) {label}');ax.tick_params(axis='y',labelsize=8);grid(ax)
    axes[-1].set_xlabel('目标环境交互 / 千步');fig.legend(*axes[0].get_legend_handles_labels(),loc='lower center',ncol=3,frameon=False,fontsize=9)
    finish(fig,'fig6_target_access','同一训练预算不等于相同的训练验证测试总访问量',[source],'10个种子的平均实际步数；三类交互分开累计；离线固定日志不重复计为新增访问')

    mcwm_risk_control(b)

    source='MC-WM/paper/generated/sparse_dense_residual_results.json';result=json.loads((b.PROJECTS/source).read_text())
    fig,axes=plt.subplots(2,2,figsize=(5.75,4.9));fig.subplots_adjust(left=.24,right=.97,bottom=.14,top=.91,hspace=.70,wspace=.30)
    specs=[('dense_minus_sparse','return_difference','(a) 留出测试回报差'),('dense_minus_sparse','violation_difference','(b) 每回合违例数差'),('secondary_diagnostics','dense_minus_sparse_calibration_risk','(c) 校准集预测误差之差'),('secondary_diagnostics','dense_minus_sparse_state_parameters','(d) 状态模型参数量差')]
    labels=['Ant执行器','重力变化','摩擦变化']
    for ax,(group,key,title) in zip(axes.flat,specs):
        for y,(env,label) in enumerate(b.ENVIRONMENTS):interval(ax,y,result['environment_results'][env][group][key])
        ax.axvline(0,color=b.GRAY,ls='--',lw=.7);ax.set(yticks=range(3),yticklabels=labels,ylim=(2.5,-.5),title=title,xlabel='稠密残差 − 稀疏残差');ax.set_title(title,fontsize=9);ax.tick_params(axis='x',labelsize=8);grid(ax)
    for ax in axes[:,1]:ax.set_yticklabels([])
    finish(fig,'fig6_sparse_dense','稀疏残差减少参数量，但未形成一致的控制收益优势',[source],'每环境10个配对种子，固定残差部署；均值及95%自助法区间；四种指标使用各自单位')
