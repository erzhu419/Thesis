"""Summarize the complete, fixed 135-episode repeated-shock comparison."""

from itertools import product
import json
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import build_figures as b


def render():
    root=b.ROOT/'experiments/csbapr_bus/runs/repeated_v1'
    methods=['bapr-pristine','csbapr-no-nau','csbapr']
    scenes=['normal','single_2x','repeated_2x','single_5x','repeated_5x']
    data={}
    for method,seed in product(methods,range(3)):
        folder=root/f'{method}_seed{seed}'
        assert json.loads((folder/'completed.json').read_text())['evaluation_episodes']==15
        for scene,ev in product(scenes,[17301,17302,17303]):
            row=json.loads((folder/f'{method}_seed{seed}_{scene}_eval{ev}.json').read_text())
            assert (row['method'],row['training_seed'],row['scene'],row['evaluation_seed'])==(method,seed,scene,ev)
            assert row['horizon_seconds']==18000 and len(row['windows'])==60
            assert sum(w['samples'] for w in row['windows'])==18000
            data[method,seed,scene,ev]=row
    def imbalance(row):
        return sum(w['mean_headway_imbalance']*w['decisions'] for w in row['windows'])/sum(w['decisions'] for w in row['windows'])
    records=[]
    for method,scene,seed in product(methods[1:],scenes,range(3)):
        effects=[]
        for ev in [17301,17302,17303]:
            row,ref=data[method,seed,scene,ev],data['bapr-pristine',seed,scene,ev]
            effects.append([(row['reward']-ref['reward'])/1000,
                            100*(row['wait_passenger_seconds']/ref['wait_passenger_seconds']-1),
                            imbalance(row)-imbalance(ref),
                            100*(row['arrived_passengers']/row['generated_passengers']-
                                 ref['arrived_passengers']/ref['generated_passengers'])])
        records.append(dict(method=method,scene=scene,training_seed=seed,
                            **dict(zip(['reward_gain_K','waiting_change_percent','imbalance_change_seconds',
                                        'completion_change_pp'],np.mean(effects,axis=0).tolist()))))
    summary=[]
    for method,scene in product(methods[1:],scenes):
        group=[r for r in records if r['method']==method and r['scene']==scene]
        summary.append(dict(method=method,scene=scene,**{key:float(np.mean([r[key] for r in group]))
                            for key in ['reward_gain_K','waiting_change_percent','imbalance_change_seconds','completion_change_pp']}))
    report=dict(evaluations=len(data),training_seeds=3,evaluation_seeds_per_policy=3,
                aggregation='paired evaluation effects averaged within each training seed first',
                training_seed_effects=records,summary=summary)
    (b.DATA/'csbapr_repeated_summary.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')

    fig=plt.figure(figsize=(5.75,5.65))
    gs=fig.add_gridspec(2,2,left=.17,right=.97,bottom=.21,top=.94,
                          height_ratios=[.85,1.4],hspace=.65,wspace=.42)
    ax=fig.add_subplot(gs[0,:])
    protocol=json.loads((root/'csbapr_seed0/protocol.json').read_text())
    for scene,color,ls,label in [('single_5x',b.GRAY,'--','一次30分钟'),
                                ('repeated_5x',b.GOLD,'-','三次各10分钟')]:
        schedule=protocol['schedules'][scene]
        times=[x[0]/60 for x in schedule]+[300]
        levels=[x[1] for x in schedule]+[1]
        ax.step(times,levels,where='post',color=color,ls=ls,lw=1.2,label=label)
    ax.set(title='(a) 相同峰值、相同高需求总时长',xlabel='运行时间 / 分钟',ylabel='需求倍率',
           xticks=[0,60,120,180,240,300],yticks=[1,5],ylim=(.5,7.5))
    ax.legend(fontsize=8,loc='upper right',ncol=2);ax.grid(axis='y')

    for i,(title,xlabel) in enumerate([
            ('(b) 相对前序BAPR的控制收益','回报增益 / 千'),
            ('(c) 算术输出的附加收益','相对稳定化ReLU的回报差 / 千')]):
        ax=fig.add_subplot(gs[1,i])
        plotted=[('csbapr-no-nau',b.BLUE),('csbapr',b.GREEN)] if i==0 else [('csbapr',b.GREEN)]
        for m,(method,color) in enumerate(plotted):
            for j,scene in enumerate(scenes):
                values=np.array([r['reward_gain_K'] for r in records if r['method']==method and r['scene']==scene])
                if i==1:
                    values-=np.array([r['reward_gain_K'] for r in records if r['method']=='csbapr-no-nau' and r['scene']==scene])
                y=j+(m-.5)*.24 if i==0 else j
                ax.scatter(values,[y-.035,y,y+.035],facecolor='white',edgecolor=color,s=15,zorder=3)
                ax.plot([min(values),max(values)],[y,y],color=color,lw=1)
                ax.scatter([np.mean(values)],[y],color=color,marker='D',s=18,zorder=4)
        ax.axvline(0,color=b.GRAY,lw=.8)
        ax.set(yticks=range(5),yticklabels=['正常','单次2倍','三次2倍','单次5倍','三次5倍'] if i==0 else [],
               ylim=(4.5,-.5),xlabel=xlabel)
        ax.set_title(title,fontsize=9,loc='left');ax.grid(axis='x');ax.margins(x=.12)
        ax.xaxis.set_major_locator(b.matplotlib.ticker.MaxNLocator(3))
    fig.legend(handles=[Line2D([],[],color=b.BLUE,label='稳定化ReLU'),
                        Line2D([],[],color=b.GREEN,label='CS-BAPR算术输出'),
                        Line2D([],[],color=b.GRAY,marker='o',mfc='white',lw=0,label='训练重复'),
                        Line2D([],[],color=b.GRAY,marker='D',lw=0,label='均值')],
               ncol=2,loc='lower center',bbox_to_anchor=(.56,.015),fontsize=8)
    b.save(fig,'fig4_repeated_shocks','CS-BAPR在单次和重复需求冲击下均提高控制回报',
           [str(root)],'135个完整300分钟回合；3个训练种子乘3个评价种子，先按训练种子汇总；(b)(c)空心点为训练种子配对效应，菱形为均值，线段连接最小与最大训练种子效应；保留全部五种场景；未加平滑或显著性标记')
    path=b.OUT/'figure_sources.json'
    existing={r['figure']:r for r in json.loads(path.read_text())}
    existing.update({r['figure']:r for r in b.MANIFEST})
    path.write_text(json.dumps(list(existing.values()),ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(summary,ensure_ascii=False,indent=2))


if __name__=='__main__':
    render()
