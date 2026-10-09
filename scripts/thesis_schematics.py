"""Transport and learning schematics for the dissertation."""

import numpy as np
from matplotlib.patches import Rectangle, Circle, Polygon


def frame(ax, x, y, w, h, title, color):
    ax.add_patch(Rectangle((x, y), w, h, fc="white", ec="#BDC5CB", lw=.75, zorder=0))
    ax.add_patch(Rectangle((x, y+h-.055), w, .055, fc=color, alpha=.10, ec="none"))
    ax.text(x+.014, y+h-.028, title, va="center", fontsize=9, weight="bold", color=color)


def research_route(b):
    fig, ax = b.diagram(6.15)
    ax.text(.5, .99, "从仿真到现实的公交系统驻站控制方法研究", ha="center", va="top", fontsize=10.5, weight="bold")
    ax.text(.5, .925, "共同基础：公交事件状态 · 有界驻站动作 · 乘客服务目标", ha="center", fontsize=9)
    stages = [
        (.700, "第3章  平稳随机环境", "无预先目标域训练数据", "固定随机规律与类别差异", "共享策略与RE-SAC\n正则化和保守价值评价", b.BLUE),
        (.505, "第4章  非平稳环境", "利用当前观测与运行历史", "随机扰动＋未知运行变化", "BAPR / CS-BAPR\n条件识别与策略响应", b.GREEN),
        (.285, "第5章  跨保真迁移", "获得固定目标离线数据", "源目标转移与访问分布差异", "公交H2O+\n目标数据＋源仿真训练", b.RED),
        (.090, "第6章  世界模型修正", "进一步利用目标预测证据", "源预测关系的复用与修正", "MC-WM\n模型选择与可信样本使用", b.PURPLE),
    ]
    for y, title, info, problem, method, color in stages:
        frame(ax, .025, y, .95, .163, title, color)
        ax.text(.047, y+.074, info, fontsize=8.5, va="center", color=color)
        ax.text(.047, y+.033, problem, fontsize=8.1, va="center")
        ax.text(.735, y+.055, method, fontsize=9, ha="center", va="center", linespacing=1.7)
        b.arrow(ax, (.445, y+.06), (.505, y+.06), color)
    b.arrow(ax, (.5, .700), (.5, .675))
    b.arrow(ax, (.5, .505), (.5, .455))
    ax.text(.52, .475, "目标域信息增加", fontsize=8.2, va="center", color=b.GRAY)
    b.arrow(ax, (.5, .285), (.5, .260))
    ax.text(.5, .035, "各章结构：问题 → 模型 → 算法 → 结果分析", ha="center", fontsize=9)
    b.save(fig, "fig1_1_research_route", "目标信息条件与利用方式递进的公交驻站控制研究路线", ["第1章1.4"], "研究问题与假设示意，无实验统计量")


def resac_architecture(b):
    fig, ax = b.diagram(5.25)
    ax.text(.03, .973, "(a) 车辆身份与运行观测", fontsize=10, weight="bold", color=b.BLUE)
    b.node(ax, .035, .83, .43, .10, "车辆、站点、方向、时段\n可学习类别嵌入", b.BLUE, size=9)
    b.node(ax, .56, .83, .405, .10, "连续运行量\n前后时距、服务时间与速度", b.GREEN, size=8.6)
    b.node(ax, .22, .69, .56, .08, "拼接为当前车辆的状态表征", b.BLUE, size=9)
    b.arrow(ax, (.25, .83), (.38, .77), b.BLUE)
    b.arrow(ax, (.76, .83), (.62, .77), b.GREEN)

    ax.text(.03, .63, "(b) 一个共享策略，多个动作价值估计", fontsize=10, weight="bold", color=b.PURPLE)
    b.node(ax, .035, .40, .27, .13, "共享演员\n" + r"$a\sim\pi_\theta(\cdot\mid s)$", b.BLUE, size=10)
    ax.plot([.22, .014, .014], [.73, .73, .465], color=b.BLUE, lw=.9)
    b.arrow(ax, (.014, .465), (.035, .465), b.BLUE)
    b.arrow(ax, (.75, .69), (.935, .555), b.PURPLE)
    ax.text(.665, .565, "在线动作价值网络", ha="center", fontsize=8.5, color=b.PURPLE)
    for x, sub in zip([.44, .63, .82], ["1", "j", "K"]):
        b.node(ax, x, .465, .14, .075, rf"$Q_{{\phi_{sub}}}$", b.PURPLE, size=12)
        b.node(ax, x, .335, .14, .075, rf"$Q_{{\phi'_{sub}}}$", b.GRAY, size=12)
        b.arrow(ax, (x + .07, .465), (x + .07, .41), b.GRAY, dashed=True)
    b.arrow(ax, (.305, .485), (.44, .50), b.PURPLE)
    ax.text(.37, .515, "动作", ha="center", fontsize=8)
    ax.text(.70, .294, "逐成员软更新对应目标网络", ha="center", fontsize=8.5, color=b.GRAY)

    ax.text(.03, .24, "(c) 交互经验与参数学习", fontsize=10, weight="bold", color=b.GREEN)
    b.node(ax, .035, .09, .30, .10, "公交停站执行\n同车转移进入共享回放", b.GREEN, size=8.5)
    b.node(ax, .46, .09, .505, .10, "评论家：拟合各自保守目标\n演员：平均动作价值减去分歧惩罚", b.PURPLE, size=8.5)
    ax.plot([.035, .014, .014], [.42, .42, .14], color=b.GREEN, lw=.9)
    b.arrow(ax, (.014, .14), (.035, .14), b.GREEN)
    b.arrow(ax, (.335, .14), (.46, .14), b.GREEN)
    ax.text(.70, .037, "实线：状态、动作与学习数据　虚线：目标参数更新", ha="center", fontsize=7.5)
    b.save(fig, "fig3_1_resac_architecture", "共享策略保留车辆身份，集成动作价值与逐成员目标支持保守学习",
           ["第3章3.1至3.3"], "方法结构示意；网络下标与正文一致，非实验统计图")


def transitduet_architecture(b):
    fig, ax = b.diagram(5.6)
    ax.text(.04, .978, "(a) 始发端：滚动生成下一班计划", fontsize=10, weight="bold", color=b.BLUE)
    for x, label in [(.055, "运行摘要与车队预算\n近期停站反馈"),
                     (.385, "上层发车策略\n请求间隔修正"),
                     (.715, "最早发车与偏移约束\n生成可执行计划")]:
        b.node(ax, x, .837, .25, .102, label, b.BLUE, size=8.3)
    b.arrow(ax, (.305, .887), (.385, .887), b.BLUE)
    b.arrow(ax, (.635, .887), (.715, .887), b.BLUE)
    b.arrow(ax, (.84, .837), (.84, .773), b.BLUE)
    ax.plot([.14, .91], [.754, .754], color=b.GRAY, lw=.85)
    for x, label in [(.19, r"$t_{j-1}^{\mathrm{act}}$"), (.53, r"$t_j^{\mathrm{plan}}$"),
                     (.86, r"$t_{j+1}^{\mathrm{plan}}$")]:
        ax.plot([x, x], [.742, .765], color=b.BLUE, lw=1.4)
        ax.text(x, .714, label, ha="center", fontsize=11)
    ax.annotate("", (.19, .797), (.53, .797),
                arrowprops=dict(arrowstyle="<->", color=b.BLUE, lw=.8))
    ax.text(.36, .809, r"$g_j=h_j^{\mathrm{plan}}$", ha="center", fontsize=10, color=b.BLUE)

    ax.text(.04, .646, "(b) 在途端：沿班次目标逐站控制", fontsize=10, weight="bold", color=b.GREEN)
    for x, label in [(.055, "本车状态与班次目标\n" + r"$(s_{j,i},g_j)$"),
                     (.385, "下层共享停站策略\n" + r"$\pi_{\mathrm{L}}(\cdot\mid s_{j,i},g_j)$"),
                     (.715, "附加停站\n0至30秒离散动作")]:
        b.node(ax, x, .502, .25, .108, label, b.GREEN, size=8.3)
    b.arrow(ax, (.305, .556), (.385, .556), b.GREEN)
    b.arrow(ax, (.635, .556), (.715, .556), b.GREEN)
    ax.plot([.53, .033, .033], [.696, .696, .556], color=b.BLUE, lw=.9)
    b.arrow(ax, (.033, .556), (.055, .556), b.BLUE)

    ax.plot([.15, .86], [.425, .425], color="#BEC7CC", lw=5, zorder=0)
    for x, label in [(.18, "站点 i"), (.50, "站点 i+1"), (.83, "站点 i+2")]:
        ax.scatter(x, .425, s=24, marker="s", facecolor="white", edgecolor=b.GREEN, zorder=3)
        ax.text(x, .392, label, ha="center", va="top", fontsize=8.4)
    b.arrow(ax, (.23, .446), (.43, .446), b.GREEN)
    b.arrow(ax, (.55, .446), (.77, .446), b.GREEN)
    b.arrow(ax, (.84, .502), (.84, .433), b.GREEN)
    ax.text(.51, .335, r"班次目标 $g_j$ 沿途保持；后续班次使用各自目标", ha="center", fontsize=8.5)

    ax.text(.51, .265, "(c) 服务评价与反馈", ha="center", fontsize=9, weight="bold", color=b.GRAY)
    b.node(ax, .055, .113, .40, .108, "已完成班次的停站统计\n进入后续上层决策", b.BLUE, size=8.6)
    b.node(ax, .565, .113, .40, .108, "候车与车内时间、车头时距\n在途车辆小时与班次执行", b.GRAY, size=8.6)
    b.arrow(ax, (.255, .312), (.255, .225), b.BLUE, dashed=True)
    b.arrow(ax, (.765, .312), (.765, .225), b.GRAY)
    ax.plot([.055, .016, .016], [.167, .167, .887], color=b.BLUE, lw=.9, ls="--")
    b.arrow(ax, (.016, .887), (.055, .887), b.BLUE, dashed=True)
    ax.text(.51, .038, "实线：计划与执行信息　虚线：已发生的运行反馈", ha="center", fontsize=8.4)
    b.save(fig, "fig4_2_transitduet_architecture", "滚动计划生成班次专属目标，停站执行的历史反馈参与后续决策",
           ["第4章4.5.1至4.5.5"], "方法与时空关系示意；位置和时间间距无定量比例，反馈箭头不表示已验证独立增益")


def render(b):
    def box(ax, x, y, w, h, title, color=b.BLUE):
        b.node(ax, x, y, w, h, title, color, size=8.5)

    def bus(ax, x, y, c=b.BLUE, w=.08):
        ax.add_patch(Rectangle((x, y), w, .032, fc=c, ec=c, lw=.6))
        for u in (.15, .40, .65):
            ax.add_patch(Rectangle((x+w*u, y+.013), w*.15, .013, fc="white", ec="none"))
        for u in (.20, .80):
            ax.add_patch(Circle((x+w*u, y), .006, fc="#333333", ec="white", lw=.4))

    def road(ax, y):
        ax.plot([.07,.93], [y,y], color="#B9C1C7", lw=9, zorder=0)
        ax.plot([.07,.93], [y,y], color="white", ls=(0,(5,5)), lw=.7, zorder=1)

    def station(ax, x, y, label):
        ax.plot([x,x], [y-.032,y-.068], color=b.GREEN, lw=1)
        ax.scatter(x,y-.037,marker="s",s=14,color=b.GREEN,zorder=3)
        ax.text(x,y-.087,label,ha="center",va="top",fontsize=8)

    def path(ax, points, color=b.GRAY, dashed=False):
        xs, ys = zip(*points)
        ax.plot(xs[:-1], ys[:-1], color=color, lw=.9, ls="--" if dashed else "-")
        b.arrow(ax, points[-2], points[-1], color, dashed=dashed)

    def finish(fig, name, claim, section):
        b.save(fig, name, claim, [section], "方法结构与物理关系示意；图中轨迹、形状和长度不代表实验测量")

    research_route(b)

    fig, ax = b.diagram(4.95)
    frame(ax,.015,.66,.97,.32,"(a) 车辆、站点与前后间隔",b.BLUE)
    road(ax,.81)
    for x,label,c in [(.14,"后车",b.GRAY),(.46,"控制车辆",b.BLUE),(.80,"前车",b.GRAY)]:
        bus(ax,x,.817,c);ax.text(x+.04,.86,label,ha="center",fontsize=8.5)
    station(ax,.50,.81,"控制站点")
    for a,z,txt in [(.18,.49,r'$h^{\mathrm{b}}$'),(.51,.84,r'$h^{\mathrm{f}}$')]:
        ax.annotate("",(a,.898),(z,.898),arrowprops=dict(arrowstyle="<->",color=b.GRAY,lw=.8))
        ax.text((a+z)/2,.915,txt,ha="center",fontsize=11)
    frame(ax,.015,.37,.97,.25,"(b) 一次到站事件的物理时间",b.GREEN)
    for x,w,c,t in [(.06,.25,b.GRAY,"上下客服务"),(.31,.28,b.BLUE,"附加停站"),(.59,.35,b.GREEN,"驶向下一站")]:
        ax.add_patch(Rectangle((x,.458),w,.060,fc=c,alpha=.17,ec=c,lw=.8))
        ax.text(x+w/2,.488,t,ha="center",va="center",fontsize=9)
    for x,t in [(.06,"到站"),(.31,"决策"),(.59,"离站"),(.94,"下一事件")]:
        ax.plot([x,x],[.446,.529],color=b.GRAY,lw=.7);ax.text(x,.415,t,ha="center",fontsize=8)
    frame(ax,.015,.025,.97,.305,"(c) 事件驱动闭环",b.PURPLE)
    for x,t in [(.055,"观测状态\n类别＋连续量"),(.38,"共享策略\n有界停站动作"),(.705,"公交运行\n乘客与交通推进")]:
        box(ax,x,.155,.24,.09,t,b.PURPLE)
    b.arrow(ax,(.295,.20),(.38,.20));b.arrow(ax,(.62,.20),(.705,.20))
    path(ax,[(.825,.155),(.825,.09),(.175,.09),(.175,.155)])
    ax.text(.50,.048,r'$(s_k,a_k,r_k,\Delta t_k,s_{k+1})$',ha="center",fontsize=11)
    finish(fig,"fig2_1_event_control","车辆间隔、事件时间和策略闭环相互对应","第2章2.2至2.3")

    fig, ax = b.diagram(4.0)
    for y,title,labels,c,dash in [(.62,"本研究：跨保真公交评价",["SUMO离线经验","简易仿真训练","SUMO冻结评价"],b.BLUE,False),
                               (.19,"实际运营中的数据角色",["真实运营离线数据","校准SUMO训练","道路公交执行"],b.GRAY,True)]:
        ax.text(.03,y+.24,title,color=c,fontsize=10,weight="bold")
        for x,label in zip([.04,.37,.70],labels): b.node(ax,x,y,.26,.16,label,c,dashed=dash,size=9)
        for x in [.30,.63]: b.arrow(ax,(x,y+.08),(x+.07,y+.08),c,dashed=dash)
    for x,label in zip([.17,.50,.83],["目标数据","源环境交互","目标环境检验"]):
        b.arrow(ax,(x,.60),(x,.47),b.GRAY,dashed=True)
        ax.text(x,.535,label,ha="center",va="center",fontsize=8.3,color=b.GRAY,backgroundcolor="white",zorder=5)
    ax.text(.5,.055,"由代理验证走向道路运行，仍需现场数据与执行条件的检验",ha="center",fontsize=8.6)
    finish(fig,"fig2_3_fidelity_roles","跨保真实验与现实部署具有对应的数据角色，但不同的目标环境","第2章2.3及2.5")

    resac_architecture(b)

    fig, ax = b.diagram(4.65)
    frame(ax,.015,.685,.97,.29,"(a) 运行条件变化与结构化策略",b.BLUE)
    for x,t in [(.05,"公交事件状态"),(.38,"共享特征"),(.71,"策略输出结构")]:box(ax,x,.765,.235,.095,t)
    b.arrow(ax,(.285,.812),(.38,.812));b.arrow(ax,(.615,.812),(.71,.812))
    ax.text(.83,.714,"ReLU / tanh / NAU-NMU / KAN",ha="center",fontsize=7.3)
    frame(ax,.015,.305,.97,.335,"(b) 训练稳定化",b.GREEN)
    labels=["目标价值聚合","奖励标准差缩放","熵温度下限","行为克隆锚定","变化反馈预热","在策略尾部统计"]
    for i,t in enumerate(labels):
        x=.05+(i%3)*.33;y=.455-(i//3)*.105
        ax.text(x,y+.034,f"{i+1:02d}",color=b.GREEN,fontsize=12,weight="bold",va="center")
        ax.text(x+.065,y+.034,t,fontsize=8.4,va="center")
        ax.plot([x,x+.26],[y,y],color="#CDDAD6",lw=.7)
    b.arrow(ax,(.83,.685),(.83,.64),b.GREEN)
    frame(ax,.015,.035,.97,.225,"(c) 保守响应闭环",b.RED)
    for x,t in [(.05,"惊异度与分歧"),(.38,"保守程度调节"),(.71,"有界停站执行")]:box(ax,x,.09,.235,.09,t,b.RED)
    b.arrow(ax,(.285,.135),(.38,.135));b.arrow(ax,(.615,.135),(.71,.135))
    finish(fig,"fig4_cs_architecture","输出结构与训练稳定化共同决定变化条件下的保守响应","第4章4.3")


    fig, ax = b.diagram(5.2)
    frame(ax,.015,.56,.97,.42,"(a) 目标离线数据与源仿真共同训练",b.BLUE)
    box(ax,.05,.775,.27,.105,"SUMO离线数据\n离线初始化",b.BLUE)
    box(ax,.05,.615,.27,.105,"简易仿真交互\n目标与源经验各半",b.GREEN)
    box(ax,.43,.685,.22,.12,"共享演员\n事件状态编码",b.PURPLE)
    box(ax,.735,.685,.22,.12,"冻结策略\n源训练终点",b.BLUE)
    b.arrow(ax,(.32,.828),(.43,.77));b.arrow(ax,(.32,.668),(.43,.72));b.arrow(ax,(.65,.745),(.735,.745))
    frame(ax,.015,.245,.97,.265,"(b) 决策接口：同时事件与两部分动作",b.GREEN)
    for i in range(5):ax.add_patch(Rectangle((.045+.040*i,.327),.030,.075,fc=b.GREEN,alpha=.25+.1*(i%3),ec=b.GREEN,lw=.5))
    ax.text(.145,.286,"事件槽与已选动作",ha="center",fontsize=7.8)
    box(ax,.335,.32,.265,.095,"当前事件与批内汇总",b.GREEN);b.arrow(ax,(.255,.367),(.335,.367))
    ax.plot([.69,.69],[.322,.411],color=b.PURPLE,lw=2)
    xx=np.linspace(.735,.93,80);yy=.322+.08*np.sin(np.linspace(0,np.pi,80))
    ax.fill_between(xx,.322,yy,color=b.PURPLE,alpha=.20);ax.plot(xx,yy,color=b.PURPLE,lw=1)
    ax.text(.80,.286,"零停站概率＋正停站密度",ha="center",fontsize=7.7);b.arrow(ax,(.60,.367),(.658,.367))
    frame(ax,.015,.025,.97,.17,"(c) 部署执行与服务评价",b.RED)
    for x,t in [(.045,"动作幅度校准"),(.37,"低载尾部完成保障"),(.71,"SUMO闭环评价")]:box(ax,x,.052,.25,.062,t,b.RED)
    b.arrow(ax,(.295,.083),(.37,.083));b.arrow(ax,(.62,.083),(.71,.083))
    finish(fig,"fig5_2_h2o_workflow","SUMO离线数据与简易仿真共同训练，执行规则形成公交控制器","第5章5.2至5.3")

    fig, ax = b.diagram(5.45)
    frame(ax,.015,.58,.97,.40,"(a) 源结构复用与模型选择",b.BLUE)
    box(ax,.05,.79,.245,.105,"目标经验划分\n拟合、选择、校准",b.BLUE)
    box(ax,.38,.805,.245,.09,"源模型＋残差",b.BLUE);box(ax,.69,.805,.245,.09,"直接目标模型",b.GREEN)
    b.arrow(ax,(.295,.845),(.38,.85));path(ax,[(.172,.895),(.172,.919),(.81,.919),(.81,.895)])
    for x in [.50,.81]:b.arrow(ax,(x,.805),(x,.73))
    ax.add_patch(Polygon([[.35,.68],[.655,.75],[.96,.68],[.655,.61]],fc="#ECE7EF",ec=b.PURPLE,lw=.8))
    ax.text(.655,.681,"留出数据预测误差",ha="center",va="center",fontsize=9)
    ax.text(.17,.66,"每次运行固定\n所选模型类别",ha="center",va="center",fontsize=8.6)
    frame(ax,.015,.24,.97,.29,"(b) 数据经验与模型经验",b.GREEN)
    box(ax,.05,.325,.25,.115,"环境采样经验\n状态、动作与后继",b.GREEN)
    box(ax,.40,.325,.24,.115,"所选世界模型\n一步后继生成",b.BLUE);box(ax,.735,.325,.22,.115,"模型经验\n置信权重",b.PURPLE)
    b.arrow(ax,(.30,.382),(.40,.382));b.arrow(ax,(.64,.382),(.735,.382))
    path(ax,[(.655,.61),(.655,.558),(.52,.558),(.52,.44)],b.PURPLE)
    box(ax,.34,.055,.36,.10,"评论家回归与策略更新",b.BLUE)
    path(ax,[(.175,.325),(.175,.105),(.34,.105)],b.GREEN);path(ax,[(.845,.325),(.845,.105),(.70,.105)],b.PURPLE)
    ax.text(.175,.205,"数据经验",ha="center",fontsize=8.5,color=b.GREEN);ax.text(.845,.205,"可信模型经验",ha="center",fontsize=8.5,color=b.PURPLE)
    finish(fig,"fig6_1_mcwm_architecture","留出预测误差决定源结构复用，环境和模型经验共同参与控制更新","第6章6.2至6.3")
