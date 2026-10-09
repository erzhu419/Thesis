"""Reward geometry and directly interpretable mechanism contrasts."""

import csv
import json
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


def ridge(front, back):
    df, db = np.abs(front - 360), np.abs(back - 360)
    weight = df / (df + db + 1e-6)
    return (-weight * df - (1-weight) * db - .5*np.abs(front-back)
            - 20*((df > 180) | (db > 180)))


def reward_geometry(b):
    grid = np.linspace(90, 720, 127)
    front, back = np.meshgrid(grid, grid)
    fig = plt.figure(figsize=(5.75, 5.9))
    gs = fig.add_gridspec(2, 2, left=.12, right=.96, bottom=.17, top=.96,
                          height_ratios=[1.35, 1], hspace=.60, wspace=.40)
    for i, (z, title, cmap) in enumerate([
            (1+ridge(front, back)/360, '(a) 计划对齐的脊形奖励', 'Blues'),
            (np.exp(-np.abs(front-back)/60), '(b) 指数时距均衡项', 'YlGn')]):
        ax = fig.add_subplot(gs[0, i], projection='3d')
        ax.plot_surface(front/360, back/360, z, cmap=cmap, rcount=90, ccount=90,
                        linewidth=0, antialiased=True, alpha=.94)
        ax.view_init(elev=27, azim=-126)
        ax.set_box_aspect((1, 1, .80))
        ax.set(xlim=(.25,2), ylim=(.25,2), zlim=(-1.8,1.1),
               xticks=[.5,1,2], yticks=[.5,1,2], zticks=[-1,0,1])
        ax.set_title(title, fontsize=9, pad=10)
        ax.set_xlabel(r'$h^{\mathrm{f}}/h_0$', labelpad=-1)
        ax.set_ylabel(r'$h^{\mathrm{b}}/h_0$', labelpad=-1)
        ax.tick_params(labelsize=7, pad=0)
        ax.plot(grid/360, grid/360,
                1+ridge(grid,grid)/360 if i == 0 else np.ones_like(grid),
                color=b.RED, lw=1.3, zorder=10)
        ax.scatter([1],[1],[1],s=18,color=b.RED,depthshade=False)
        for axis in [ax.xaxis,ax.yaxis,ax.zaxis]:
            axis.pane.fill = False
    ax = fig.add_subplot(gs[1,0])
    ax.plot(grid, 1+ridge(grid,grid)/360, color=b.BLUE, label='脊形奖励')
    ax.plot(grid, np.ones_like(grid), color=b.GREEN, ls='--', label='指数均衡')
    ax.axvline(360,color=b.GRAY,lw=.7,ls=':')
    ax.set(title='(c) 两侧均匀，但整体偏离计划', xlabel='共同车头时距 / 秒',
           ylabel='展示尺度下的奖励', xticks=[120,360,600], ylim=(-.9,1.15))
    ax.grid(axis='y')
    ax = fig.add_subplot(gs[1,1])
    delta = np.linspace(-600,600,1201)
    ax.plot(delta,1+ridge(360+delta/2,360-delta/2)/360,color=b.BLUE)
    ax.plot(delta,np.exp(-np.abs(delta)/60),color=b.GREEN,ls='--')
    ax.plot(delta,np.exp(-np.abs(delta)),color=b.GOLD,ls=':',label='指数均衡（1秒尺度）')
    ax.set(title='(d) 平均间隔固定，局部失衡',xlabel='前后时距差 / 秒',
           xticks=[-600,0,600],ylim=(-.9,1.15),yticklabels=[])
    ax.grid(axis='y')
    fig.legend(handles=[Line2D([],[],color=b.BLUE,label='脊形奖励：1+r/360'),
                        Line2D([],[],color=b.GREEN,ls='--',label='指数项：60秒尺度'),
                        Line2D([],[],color=b.GOLD,ls=':',label='指数项：1秒尺度')],
               loc='lower center',ncol=1,fontsize=8,bbox_to_anchor=(.53,.003))
    b.save(fig,'fig3_holding_geometry','均衡奖励保留整条对角脊线，计划对齐奖励区分共同偏移',
           ['Categorical Feature公交奖励定义', 'https://lijunsun.github.io/files/papers/2020-TRC-Bus.pdf, Eq.3'],
           '解析函数，无抽样统计；h0=360秒；脊形奖励仅作正仿射显示变换1+r/360；指数尺度1秒与60秒为明确设定，不宣称原文尺度；固定动作项不改变状态曲面的形状')


def bapr_comparison(b):
    root = b.PROJECTS / 'BAPR/reproducibility/v32'
    data = json.loads((root / 'analysis.json').read_text())
    arms = ['resac_b0_8p4m', 'canonical_v5_map_compensation']
    values = []
    for seed in data['training_seeds']:
        audit = json.loads((root / f'audits/seed_{seed}/audit.json').read_text())
        values.append([np.mean([event[arm]['return_mean'] for event in audit['switching_holdout'].values()])
                       for arm in arms])
    values = np.array(values)
    comparison = data['comparisons']['causal_vs_equal_budget_resac']
    assert np.isclose(np.mean(values[:,1]-values[:,0]), comparison['paired_mean'])
    (b.DATA/'bapr_compensation_confirmation.json').write_text(json.dumps(
        dict(seeds=data['training_seeds'],paired_returns=values.tolist(),comparison=comparison,
             switching_return_means=data['switching_return_means']),indent=2)+'\n')
    fig, axes = plt.subplots(2,1,figsize=(5.75,5.5),gridspec_kw={'height_ratios':[1.45,1]})
    fig.subplots_adjust(left=.24,right=.95,bottom=.12,top=.94,hspace=.55)
    ax=axes[0]
    for i,(resac,bapr) in enumerate(values):
        ax.plot([resac,bapr],[i,i],color=b.BLUE if bapr>resac else b.RED,lw=1.2)
    ax.scatter(values[:,0],range(10),color='white',edgecolor=b.GRAY,s=27,label='RE-SAC',zorder=3)
    ax.scatter(values[:,1],range(10),color=b.BLUE,s=27,label='BAPR',zorder=3)
    ax.set(yticks=range(10),yticklabels=[str(i) for i in range(1,11)],ylim=(9.6,-1.7),
           ylabel='配对训练重复',xlabel='切换回报',title='(a) 相同策略交互预算：9/10个重复改善')
    ax.legend(ncol=2,fontsize=8,loc='upper left',bbox_to_anchor=(0,1.0))
    ax.grid(axis='x')
    ax=axes[1]
    keys=['canonical_no_compensation','canonical_v5_map_compensation','canonical_true_mode_compensation']
    labels=['不做补偿','因果识别后补偿','已知模式补偿']
    v=[data['switching_return_means'][key] for key in keys]
    ax.barh(range(3),v,color=[b.GRAY,b.BLUE,b.GREEN],height=.55)
    for i,value in enumerate(v):ax.text(value+45,i,f'{value:.0f}',va='center',fontsize=8)
    ax.axvline(data['switching_return_means']['resac_b0_8p4m'],color=b.RED,ls='--',lw=.9)
    ax.text(2310,2.65,'RE-SAC均值',ha='center',color=b.RED,fontsize=8)
    ax.set(yticks=range(3),yticklabels=labels,ylim=(2.9,-.6),xlim=(0,4100),
           xlabel='切换回报均值',title='(b) 同一基准策略的动作补偿阶梯')
    ax.grid(axis='x')
    b.save(fig,'fig4_bapr_compensation','已知可逆执行模式下，因果动作补偿优于等策略预算RE-SAC',
           [str(root/'analysis.json'),str(root/'audits')],
           '10个新训练种子，每种子先汇总3个事件流；策略交互均为840万；上图保留全部配对重复，下图为10种子均值；估计器额外监督成本另述；非公交实验')


def csbapr_contrast(b):
    source=b.PROJECTS/'CS-BAPR/paper/source_data/reported_tables.csv'
    with source.open() as handle:
        rows=[r for r in csv.DictReader(handle) if r['table']=='4']
    conditions=list(dict.fromkeys(r['metric'] for r in rows))
    methods=['bapr-pristine','csbapr-no-nau','csbapr']
    values=np.array([[float(next(r['value'] for r in rows if r['method']==m and r['metric']==c))
                      for m in methods] for c in conditions])
    names=['通勤日变化','20倍方波','递增需求']
    fig=plt.figure(figsize=(5.75,4.8))
    gs=fig.add_gridspec(2,2,left=.22,right=.96,bottom=.13,top=.94,hspace=.66,wspace=.32)
    ax=fig.add_subplot(gs[0,:])
    for i,(row,name,color) in enumerate(zip(values,names,[b.BLUE,b.GREEN,b.GOLD])):
        ax.plot(range(3),row,'o-',color=color,label=name,ms=4)
        for x,y in enumerate(row):ax.annotate(f'{y:.0f}',(x,y),xytext=(0,7 if i==2 else -13),
                                             textcoords='offset points',ha='center',fontsize=8,color=color)
    ax.set(xticks=range(3),xticklabels=['前序BAPR','稳定化ReLU','CS-BAPR\n算术输出'],
           ylabel='回报中位数 / 千',ylim=(values.min()-70,values.max()+110),xlim=(-.25,2.28),
           title='(a) 相同反复需求下的控制器比较')
    ax.legend(ncol=3,loc='upper center',fontsize=8)
    ax.grid(axis='y')
    for i,title in enumerate(['(b) 稳定化ReLU − 前序BAPR','(c) 算术输出 − 稳定化ReLU']):
        ax=fig.add_subplot(gs[1,i]);delta=values[:,i+1]-values[:,i]
        ax.barh(range(3),delta,color=[b.BLUE if v>=0 else b.RED for v in delta],height=.52)
        ax.axvline(0,color=b.GRAY,lw=.8)
        for j,v in enumerate(delta):ax.annotate(f'{v:+.0f}',(v,j),xytext=(4 if v>=0 else -4,0),
                                              textcoords='offset points',va='center',ha='left' if v>=0 else 'right',fontsize=8)
        pad=max(abs(delta).max()*.3,5)
        ax.set(yticks=range(3),yticklabels=names if i==0 else [],ylim=(2.6,-.6),
               xlim=(min(delta.min(),0)-pad,max(delta.max(),0)+pad),xlabel='中位数之差 / 千')
        ax.set_title(title,fontsize=8.5,loc='left');ax.grid(axis='x')
    b.save(fig,'fig4_oscillating','CS-BAPR在三类反复需求下提高控制回报，训练稳定化贡献主要收益',
           [str(source)],'沿用各组10种子中位数；差值为组中位数之差，并非配对差的中位数；组合配置比较，不拆解六项修正的因果贡献；未添加未报告的误差线')


if __name__ == '__main__':
    import build_figures as b
    for render in [reward_geometry,bapr_comparison,csbapr_contrast]:
        render(b)
    path=b.OUT/'figure_sources.json'
    existing={r['figure']:r for r in json.loads(path.read_text())}
    existing.update({r['figure']:r for r in b.MANIFEST})
    path.write_text(json.dumps(list(existing.values()),ensure_ascii=False,indent=2)+'\n')
