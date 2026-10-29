"""Render conventional thesis figures in Python; keep existing numerical evidence."""
from pathlib import Path
import csv
import json
import os
import re
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parents[3]
OUT = Path(__file__).resolve().parent
ORIGINAL = ROOT / "manuscript/figures"

# Cairo uses this small font directory; matplotlib explicitly registers the same fonts.
FONTDIR = OUT / "fonts"
FONTDIR.mkdir(exist_ok=True)
for name in ("simsun.ttc", "times.ttf", "timesbd.ttf", "timesi.ttf", "timesbi.ttf"):
    link = FONTDIR / name
    if not link.exists():
        link.symlink_to(Path("/mnt/c/Windows/Fonts") / name)
config = OUT / "fonts.conf"
config.write_text(f'<?xml version="1.0"?><!DOCTYPE fontconfig SYSTEM "fonts.dtd"><fontconfig><include>/etc/fonts/fonts.conf</include><dir>{FONTDIR}</dir></fontconfig>')
os.environ["FONTCONFIG_FILE"] = str(config)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.patches import Rectangle, FancyArrowPatch
from matplotlib.ticker import FuncFormatter
import numpy as np
import cairosvg

for name in ("simsun.ttc", "times.ttf"):
    font_manager.fontManager.addfont(str(FONTDIR / name))
plt.rcParams.update({
    "font.family": ["Times New Roman", "SimSun"], "font.size": 10,
    "axes.labelsize": 10, "xtick.labelsize": 9, "ytick.labelsize": 9,
    "axes.linewidth": .8, "axes.spines.right": False, "axes.spines.top": False,
    "axes.unicode_minus": False, "legend.frameon": False,
    "svg.fonttype": "none", "mathtext.fontset": "stix",
    "figure.facecolor": "white", "savefig.facecolor": "white",
})
NS = {"s": "http://www.w3.org/2000/svg"}
ET.register_namespace("", NS["s"])
ET.register_namespace("xlink", "http://www.w3.org/1999/xlink")
K = FuncFormatter(lambda v, pos: f"{v:g}K")
MANIFEST = []
TEXTBOXES = []


def save(fig, name, sources, stats):
    # Detect actual box overflows, which would require enlarging the affected node.
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    for artist, ax, bounds in TEXTBOXES:
        if artist.figure is not fig:
            continue
        box = artist.get_window_extent(renderer)
        x, y, w, h = bounds
        low, high = ax.transData.transform([(x, y), (x+w, y+h)])
        if not (box.x0 >= low[0]-1 and box.x1 <= high[0]+1 and box.y0 >= low[1]-1 and box.y1 <= high[1]+1):
            raise ValueError(f"Text overflows node: {name}: {artist.get_text()}")
    fig.savefig(OUT / f"{name}.svg", bbox_inches="tight", pad_inches=.03)
    fig.savefig(OUT / f"{name}.png", dpi=600, bbox_inches="tight", pad_inches=.03)
    plt.close(fig)
    MANIFEST.append({"name": name, "sources": sources, "statistics": stats,
                     "method": "Python matplotlib", "formats": ["editable SVG", "600 dpi PNG"]})


def canvas(height=6):
    fig, ax = plt.subplots(figsize=(5.75, height))
    fig.subplots_adjust(left=.015, right=.985, top=.99, bottom=.01)
    ax.set(xlim=(0, 1), ylim=(0, 1))
    ax.axis("off")
    return fig, ax


def node(ax, x, y, w, h, label, size=10):
    ax.add_patch(Rectangle((x, y), w, h, facecolor="white", edgecolor="black", lw=.9, zorder=3))
    artist = ax.text(x+w/2, y+h/2, label, ha="center", va="center", fontsize=size,
                     linespacing=1.45, zorder=4)
    TEXTBOXES.append((artist, ax, (x, y, w, h)))


def arrow(ax, start, end):
    ax.add_patch(FancyArrowPatch(start, end, arrowstyle="-|>", mutation_scale=9,
                                lw=.8, color="black", zorder=2))


def fork(ax, top, points, junction):
    ax.plot([top[0], top[0]], [top[1], junction], color="black", lw=.8, zorder=1)
    xs = [p[0] for p in points]
    ax.plot([min(xs), max(xs)], [junction, junction], color="black", lw=.8, zorder=1)
    for x, y in points:
        arrow(ax, (x, junction), (x, y))


def merge(ax, starts, end, junction):
    xs = [p[0] for p in starts]
    for x, y in starts:
        ax.plot([x, x], [y, junction], color="black", lw=.8, zorder=1)
    ax.plot([min(xs), max(xs)], [junction, junction], color="black", lw=.8, zorder=1)
    arrow(ax, (end[0], junction), end)


def research_route():
    fig, ax = canvas(7.5)
    node(ax,.04,.885,.92,.10,"双向按时刻表逐站驻站控制\n问题定义与统一事件决策模型",10.5)
    ax.text(.255,.848,"目标环境数据可得性与源仿真可信度",ha="center",fontsize=9)
    ax.text(.755,.848,"核心方法",ha="center",fontsize=10)
    rows = [
        (.665,"子问题一\n无目标数据\n统计规律平稳","RE-SAC\n车队共享鲁棒学习"),
        (.485,"子问题二\n无目标数据\n统计规律非平稳变化","BAPR与CS-BAPR\n条件识别及软符号适应"),
        (.305,"子问题三\n有固定目标离线数据\n源仿真标定可信","H2O+\n公交跨保真混合训练"),
        (.125,"子问题四\n有固定目标离线数据\n源仿真标定不足","MC-WM\n模型修正世界模型辅助训练"),
    ]
    fork(ax,(.5,.885),[(.255,.81),(.755,.81)],.825)
    for i,(y,problem,method) in enumerate(rows):
        node(ax,.04,y,.43,.145,problem)
        node(ax,.54,y,.42,.145,method)
        arrow(ax,(.47,y+.0725),(.54,y+.0725))
        if i < 3:
            arrow(ax,(.255,y),(.255,y-.035))
            arrow(ax,(.755,y),(.755,y-.035))
    merge(ax,[(.255,.125),(.755,.125)],(.5,.085),.10)
    node(ax,.04,.005,.92,.08,"快速公交仿真单线/多线实验　SUMO高保真目标域跨保真评价\n受控动力学机理验证",9.2)
    save(fig,"fig1_1_research_route",["第1章技术路线图说明"],"研究框架示意，无实验统计量")


def chapter3():
    fig,ax=canvas(6.1)
    node(ax,.04,.88,.92,.10,"平稳随机环境下\n双向按时刻表逐站驻站问题",10.5)
    fork(ax,(.5,.88),[(.18,.80),(.5,.80),(.82,.80)],.84)
    nodes=[(.035,"单车经验失衡","类别特征嵌入\n全车队经验共享表征"),
           (.355,"偶然随机状态扰动\n自举目标污染","目标网络权重范数\n正则化项"),
           (.675,"认知样本稀疏\n未知动作高估","集成分歧\n保守评价项")]
    for x,problem,method in nodes:
        node(ax,x,.66,.29,.14,problem,9.6)
        arrow(ax,(x+.145,.66),(x+.145,.57))
        node(ax,x,.43,.29,.14,method,9.6)
    merge(ax,[(.18,.43),(.5,.43),(.82,.43)],(.5,.33),.38)
    node(ax,.16,.23,.68,.10,"RE-SAC\n车队共享鲁棒驻站控制",10.5)
    arrow(ax,(.5,.23),(.5,.14))
    node(ax,.04,.02,.92,.12,"快速公交仿真双向单线路与长沙多线路场景\n系统验证与消融剖析",10)
    save(fig,"fig3_1_chapter_route",["第3章技术路线图说明"],"研究框架示意，无实验统计量")


def chapter4():
    fig,ax=canvas(6.6)
    node(ax,.04,.89,.92,.09,"非平稳环境下客流与路况统计规律时变",10.5)
    fork(ax,(.5,.89),[(.255,.80),(.745,.80)],.845)
    left=[(.66,.14,"子问题一\n新旧经验混杂下的价值评价失准"),
          (.435,.14,"BAPR近期统计反馈\n运行长度加权动态调节保守系数")]
    right=[(.66,.14,"子问题二\n极端冲击突破训练范围后的策略外推失真"),
           (.475,.12,"六项训练稳定化设计\n可学习基准动作加参考归零神经修正结构"),
           (.275,.13,"SINDy短期代价辨识\n软符号雅可比导数约束（CS-BAPR）")]
    for x,col in [(.035,left),(.525,right)]:
        for i,(y,h,t) in enumerate(col):
            node(ax,x,y,.44,h,t,9.2)
            if i:
                py=col[i-1][0]
                arrow(ax,(x+.22,py),(x+.22,y+h))
    merge(ax,[(.255,.435),(.745,.275)],(.5,.16),.21)
    node(ax,.04,.025,.92,.135,"长沙多线路六类非平稳客流场景\n受控动力学平台\n系统验证",10)
    save(fig,"fig4_1_chapter_route",["第4章技术路线图说明"],"研究框架示意，无实验统计量")


def chapter5():
    fig,ax=canvas(6.8)
    blocks=[(.855,.125,"掌握目标域固定离线数据与经标定快速源仿真\n目标环境不允许大规模在线试错"),
            (.66,.125,"纯离线学习、目标域在线微调与源域在线训练\n三种单一模式的缺陷"),
            (.465,.125,"双层仿真标定检验\n自然运行检验＋多策略驻站响应检验"),
            (.23,.165,"基于H2O+的公交跨保真混合训练方法\n同时到站车组顺序条件状态表征\nIQL离线初始化与双域混合回放续训"),
            (.025,.135,"SUMO高保真目标环境\n多维度对比验证与域分类器诊断")]
    for i,(y,h,t) in enumerate(blocks):
        node(ax,.055,y,.89,h,t,10)
        if i:arrow(ax,(.5,blocks[i-1][0]),(.5,y+h))
    save(fig,"fig5_1_chapter_route",["第5章技术路线图说明"],"研究框架示意，无实验统计量")


def chapter6():
    fig,ax=canvas(7.0)
    node(ax,.04,.885,.92,.095,"源仿真参数标定不足\n持续动力学偏移",10.5)
    arrow(ax,(.5,.885),(.5,.83))
    node(ax,.04,.73,.92,.10,"纯数据混合/加权\n无法修正失准后继预测",10)
    arrow(ax,(.5,.73),(.5,.68))
    node(ax,.20,.60,.60,.08,"目标域固定离线数据",10)
    fork(ax,(.5,.60),[(.255,.535),(.745,.535)],.565)
    node(ax,.04,.425,.43,.11,"源预测残差修正模型",9.5)
    node(ax,.53,.425,.43,.11,"直接目标预测模型",9.5)
    merge(ax,[(.255,.425),(.745,.425)],(.5,.365),.395)
    node(ax,.04,.255,.92,.11,"未参与拟合的验证集误差自动择优\n公交混合事件世界模型",10)
    arrow(ax,(.5,.255),(.5,.215))
    node(ax,.04,.115,.92,.10,"局部预测误差与归一化对数置信度递推\n模型样本权重与一步转移数据辅助价值更新",9.5)
    arrow(ax,(.5,.115),(.5,.075))
    node(ax,.04,.005,.92,.07,"多档源标定偏差下的SUMO多线路场景与受控动力学平台",9.3)
    save(fig,"fig6_1_chapter_route",["第6章技术路线图说明"],"研究框架示意，无实验统计量")


def temperature():
    source=ORIGINAL/"source_data/published_values.json"
    d=json.loads(source.read_text())["temperature"]
    fig,ax=plt.subplots(figsize=(5.75,3.85))
    fig.subplots_adjust(left=.16,right=.975,bottom=.17,top=.91)
    x=np.arange(1,6); values=np.asarray(d["gains"])/1000
    ax.bar(x,values,width=.52,facecolor="white",edgecolor="black",linewidth=.9)
    ax.axhline(0,color="black",lw=.8)
    low,high=np.asarray(d["ci95_t"])/1000; mean=d["mean"]/1000
    ax.errorbar(6.4,mean,yerr=[[mean-low],[high-mean]],fmt="D",mfc="black",mec="black",color="black",capsize=4,ms=5)
    ax.axvline(5.7,color=".5",lw=.7,ls="--")
    for i,v in zip(x,values):ax.text(i,v+(.45 if v>=0 else -.8),f"{v:g}K",ha="center",va="bottom" if v>=0 else "top",fontsize=9)
    ax.set(xticks=[1,2,3,4,5,6.4],xticklabels=["1","2","3","4","5","均值"],
           xlabel="独立训练重复",ylabel="固定温度相对自动调温的配对回报差",ylim=(-2.6,18.6),xlim=(.4,7.1))
    ax.yaxis.set_major_formatter(K)
    ax.text(6.4,17.1,"95%置信区间",ha="center",fontsize=9)
    save(fig,"fig3_3_temperature_paired",[str(source)],"5个独立训练配对差；均值8939，95% Student-t区间[1915,15963]；df=4")


def offline_learning():
    source=ROOT/"data/external/offline-sumo/paper/generated/figure2_learning_curves_data.csv"
    data=list(csv.DictReader(source.open()))
    fig,ax=plt.subplots(figsize=(5.75,4.05))
    fig.subplots_adjust(left=.165,right=.975,bottom=.25,top=.965)
    colors=["#222222","#555555","#888888","#245681","#617e46","#8b5b3e"]
    styles=["-","--","-.",":","-","--"]
    for method,c,ls,marker in zip(dict.fromkeys(r["method"] for r in data),colors,styles,"os^DvX"):
        rr=sorted([r for r in data if r["method"]==method],key=lambda r:float(r["epoch"]))
        x=np.array([float(r["epoch"])/10 for r in rr])
        m=np.array([float(r["return_mean"])/1000 for r in rr])
        sd=np.array([float(r["return_std"])/1000 for r in rr])
        ax.plot(x,m,color=c,ls=ls,marker=marker,ms=3.3,lw=1.1,label=method.replace("OnlineSAC","在线SAC"))
        ax.fill_between(x,m-sd,m+sd,color=c,alpha=.09,lw=0)
    ax.set(xlabel="新增目标控制决策 / 次",ylabel="完整回合回报")
    ax.xaxis.set_major_formatter(K);ax.yaxis.set_major_formatter(K)
    fig.legend(*ax.get_legend_handles_labels(),loc="lower center",bbox_to_anchor=(.5,.01),ncol=3,fontsize=9)
    save(fig,"fig5_offline_learning",[str(source)],"5个训练种子，各种子平均10个完整18000秒SUMO回合；曲线为均值，阴影为跨种子标准差")


def redraw_k_axes():
    skip={"fig3_3_temperature_paired","fig5_offline_learning"}
    for source in sorted(ORIGINAL.glob("*.svg")):
        if source.stem in skip:continue
        tree=ET.parse(source); root=tree.getroot();changes=[]
        for axis in root.findall(".//s:g",NS):
            if not axis.get("id","").startswith("matplotlib.axis_"):continue
            labels=[]
            for t in axis.findall(".//s:text",NS):
                value="".join(t.itertext())
                if re.search(r"/\s*千(?!米)",value):labels.append(t)
            if not labels:continue
            for t in labels:
                old="".join(t.itertext())
                new=re.sub(r"\s*/\s*千([次步]?)",lambda m:" / "+m[1] if m[1] else "",old)
                t.text=new;changes.append([old,new])
            for tick in axis.findall("s:g",NS):
                if not re.match(r"[xy]tick_",tick.get("id","")):continue
                for t in tick.findall(".//s:text",NS):
                    value="".join(t.itertext())
                    if re.fullmatch(r"[−+\-]?\d+(?:\.\d+)?",value):t.text=value+"K"
        if not changes:continue
        if source.stem=="fig4_oscillating":
            # Its direct value labels use the same thousand-scale values as the axes.
            for axes in root.findall(".//s:g",NS):
                if not axes.get("id","").startswith("axes_"):continue
                for group in axes.findall("s:g",NS):
                    if not group.get("id","").startswith("text_"):continue
                    for t in group.findall(".//s:text",NS):
                        value="".join(t.itertext())
                        if re.fullmatch(r"[−+\-]?\d+(?:\.\d+)?",value):t.text=value+"K"
        for t in root.findall(".//s:text",NS):
            style=t.get("style","")
            family="SimSun" if re.search(r"[\u4e00-\u9fff]","".join(t.itertext())) else "Times New Roman"
            style=re.sub(r"font-family:\s*[^;]+", f"font-family: '{family}'",style)
            t.set("style",style)
        target=OUT/source.name;tree.write(target,encoding="utf-8",xml_declaration=True)
        cairosvg.svg2png(url=str(target),write_to=str(target.with_suffix(".png")),dpi=600)
        MANIFEST.append({"name":source.stem,"sources":[str(source)],"method":"Python SVG axis redraw preserving original data paths",
                         "axis_changes":changes,"statistics":"原图数据路径、误差条、曲线与统计定义保持原样",
                         "formats":["editable SVG","600 dpi PNG"]})


def main():
    research_route();chapter3();chapter4();chapter5();chapter6()
    temperature();offline_learning();redraw_k_axes()
    (OUT/"figure_redraw_manifest.json").write_text(json.dumps(MANIFEST,ensure_ascii=False,indent=2)+"\n")
    mapping=json.loads((OUT/"figure_map_source.json").read_text())
    route={"图3-1":"fig3_1_chapter_route","图4-1":"fig4_1_chapter_route","图5-1":"fig5_1_chapter_route","图6-1":"fig6_1_chapter_route"}
    names={r["name"] for r in MANIFEST}
    for f in mapping["figures"]:
        name=route.get(f["figure"],f["source_name"])
        if name in names:
            f["source"]=str(OUT/f"{name}.png")
            f["editable_svg"]=str(OUT/f"{name}.svg")
            f["status"]="redrawn"
    (OUT/"figure_map_final.json").write_text(json.dumps(mapping,ensure_ascii=False,indent=2)+"\n")
    print(f"Wrote {len(MANIFEST)} figures; mapped {len(mapping['figures'])} manuscript figures.")


if __name__=="__main__":main()
