"""Patch the formatted Word baseline, preserving its unrelated XML parts."""

from copy import deepcopy
import json
from pathlib import Path
import shutil
from zipfile import ZipFile, ZIP_DEFLATED

from lxml import etree as E
from PIL import Image

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
BASE = ROOT / "output/排版完成.docx"
BACKUP = HERE / "排版完成_修改前.docx"
STAGE = HERE / "排版完成_结果整合.docx"
NS = {"w":"http://schemas.openxmlformats.org/wordprocessingml/2006/main",
      "m":"http://schemas.openxmlformats.org/officeDocument/2006/math",
      "a":"http://schemas.openxmlformats.org/drawingml/2006/main",
      "wp":"http://schemas.openxmlformats.org/drawingml/2006/wordprocessingDrawing",
      "r":"http://schemas.openxmlformats.org/officeDocument/2006/relationships",
      "pic":"http://schemas.openxmlformats.org/drawingml/2006/picture"}
REL = "http://schemas.openxmlformats.org/package/2006/relationships"


def q(tag):
    prefix, local = tag.split(":")
    return "{"+NS[prefix]+"}"+local


def txt(element):
    return "".join(element.xpath(".//w:t/text() | .//m:t/text()", namespaces=NS))


class Editor:
    def __init__(self):
        if not BACKUP.exists():
            shutil.copy2(BASE, BACKUP)
        with ZipFile(BACKUP) as archive:
            self.parts = {name:archive.read(name) for name in archive.namelist()}
        self.doc = E.fromstring(self.parts["word/document.xml"])
        self.body = self.doc.find("w:body", NS)
        self.old = list(self.body)
        assert len(self.old) == 3099 and txt(self.old[0]) == "从仿真到现实的公交系统驻站控制方法研究"
        self.rels = E.fromstring(self.parts["word/_rels/document.xml.rels"])
        self.bookmark_id = max(int(n.get(q("w:id"))) for n in self.doc.findall(".//w:bookmarkStart", NS))+1
        self.picture_id = max(int(n.get("id")) for n in self.doc.findall(".//wp:docPr", NS))+100
        self.changed = []

    def para(self, text, style="body", bookmark=None):
        templates = {"body":512, "h3":1387, "h4":1280, "figurecaption":2020,
                     "tablecaption":2024}
        source = self.old[templates[style]]
        p = E.Element(q("w:p"))
        p.append(deepcopy(source.find("w:pPr", NS)))
        if bookmark:
            start = E.SubElement(p,q("w:bookmarkStart"))
            start.set(q("w:id"),str(self.bookmark_id)); start.set(q("w:name"),bookmark)
        run = E.SubElement(p,q("w:r"))
        t = E.SubElement(run,q("w:t")); t.text=text
        t.set("{http://www.w3.org/XML/1998/namespace}space","preserve")
        if bookmark:
            end = E.SubElement(p,q("w:bookmarkEnd")); end.set(q("w:id"),str(self.bookmark_id))
            self.bookmark_id += 1
        return p

    def replace(self, index, text):
        p = self.old[index]
        assert p.tag == q("w:p")
        self.changed.append(dict(index=index, before=txt(p), after=text))
        preserved = [n for n in p if n.tag in (q("w:pPr"),q("w:bookmarkStart"),q("w:bookmarkEnd"))]
        for n in list(p):
            if n not in preserved:
                p.remove(n)
        run=E.SubElement(p,q("w:r"))
        t=E.SubElement(run,q("w:t")); t.text=text

    def insert(self, index, nodes):
        for node in nodes:
            self.old[index].addprevious(node)

    def table(self, rows, widths):
        table = deepcopy(self.old[2016])
        for n in list(table):
            if n.tag == q("w:tr"):
                table.remove(n)
        grid = table.find("w:tblGrid",NS)
        for n in list(grid): grid.remove(n)
        total=8306
        sizes=[round(total*w/sum(widths)) for w in widths]
        sizes[-1]=total-sum(sizes[:-1])
        for size in sizes:
            col=E.SubElement(grid,q("w:gridCol")); col.set(q("w:w"),str(size))
        row_templates=self.old[2016].findall("w:tr",NS)
        for i, values in enumerate(rows):
            tr=E.SubElement(table,q("w:tr"))
            trpr=E.SubElement(tr,q("w:trPr")); E.SubElement(trpr,q("w:cantSplit"))
            if i==0: E.SubElement(trpr,q("w:tblHeader"))
            source=row_templates[0 if i==0 else 1].findall("w:tc",NS)
            for j,value in enumerate(values):
                cell=deepcopy(source[0 if j==0 else 1])
                cell.find("w:tcPr/w:tcW",NS).set(q("w:w"),str(sizes[j]))
                p=cell.find("w:p",NS)
                for n in list(p):
                    if n.tag!=q("w:pPr"):p.remove(n)
                ppr=p.find("w:pPr",NS)
                for n in list(ppr):
                    if n.tag==q("w:keepNext"):ppr.remove(n)
                if i<len(rows)-1:E.SubElement(ppr,q("w:keepNext"))
                run=E.SubElement(p,q("w:r")); rp=E.SubElement(run,q("w:rPr"))
                if i==0:E.SubElement(rp,q("w:b"))
                fonts=E.SubElement(rp,q("w:rFonts"))
                fonts.set(q("w:ascii"),"Times New Roman"); fonts.set(q("w:hAnsi"),"Times New Roman")
                fonts.set(q("w:eastAsia"),"宋体")
                sz=E.SubElement(rp,q("w:sz"));sz.set(q("w:val"),"21")
                t=E.SubElement(run,q("w:t"));t.text=str(value)
                tr.append(cell)
        return table

    def figure(self, name, description):
        path=HERE/"figures"/(name+".png")
        self.picture_id+=1
        rid="rId"+str(max(int(n.get("Id")[3:]) for n in self.rels)+1)
        target="media/thesis_results_"+name+".png"
        node=E.SubElement(self.rels,"{"+REL+"}Relationship")
        node.set("Id",rid);node.set("Type",NS["r"]+"/image");node.set("Target",target)
        self.parts["word/"+target]=path.read_bytes()
        p=deepcopy(self.old[2019])
        for key in list(p.attrib):del p.attrib[key]
        p.find(".//a:blip",NS).set(q("r:embed"),rid)
        with Image.open(path) as im:height=round(5255999*im.height/im.width)
        for n in p.xpath(".//wp:extent | .//a:xfrm/a:ext",namespaces=NS):
            n.set("cx","5255999");n.set("cy",str(height))
        for n in p.xpath(".//wp:docPr | .//pic:cNvPr",namespaces=NS):
            n.set("id",str(self.picture_id));n.set("name",name);n.set("descr",description)
        return p

    def save(self):
        self.parts["word/document.xml"]=E.tostring(self.doc,xml_declaration=True,encoding="UTF-8",standalone=True)
        self.parts["word/_rels/document.xml.rels"]=E.tostring(self.rels,xml_declaration=True,encoding="UTF-8",standalone=True)
        with ZipFile(STAGE,"w",ZIP_DEFLATED) as archive:
            for name,value in self.parts.items():archive.writestr(name,value)
        (HERE/"word_changes.json").write_text(json.dumps(self.changed,ensure_ascii=False,indent=2)+"\n")
        print("Saved",STAGE,"; replaced paragraphs",len(self.changed))


def main():
    data=json.loads((HERE/"integration_values.json").read_text())
    ed=Editor()
    p=ed.para
    replacements={
        6:"（1）针对车辆与站点的类别差异及随机运行中的动作价值估计波动，提出结合类别嵌入与连续状态的共享停站策略，并提出鲁棒集成软演员评论家方法（Robust Ensemble Soft Actor-Critic，RE-SAC）。同学习器比较中，学习嵌入相对独热编码在正常运行、客流激增和快速切换下，分别降低5.51%、1.41%和4.91%的人均广义时间，即两倍候车时间与车内时间之和。进一步交叉考察车头时距与乘客时间两种目标、仅驻站与驻站—速度协同两种动作：在双动作下，RE-SAC相对不含指定正则项的集成参照，分别降低11.32%、3.56%和5.93%的计划时距偏差；以乘客时间训练时，对应的人均广义时间降幅为20.09%、5.46%和14.27%。结果表明，保守价值学习的作用不限于单一奖励，但不同运营目标仍存在取舍。",
        7:"（2）针对运行规律变化与局部控制权限有限的问题，提出贝叶斯自适应策略正则化（Bayesian Adaptive Policy Regularization，BAPR），通过环境识别和信念调节形成条件适应；进一步提出因果符号贝叶斯自适应策略正则化（Causal-Symbolic Bayesian Adaptive Policy Regularization，CS-BAPR）。公交主配置以训练稳定化与策略输出结构为核心。在正常需求、单次与重复冲击、持续高需求及短时极端冲击下，稳定化感知机配置相对前序BAPR均降低平均时距奖励惩罚：仅驻站时降幅为58.82%至69.79%，驻站—速度协同时为15.50%至28.56%。普通输出与算术输出的比较表明，结构化输出并非在每种条件下都有额外优势；改用乘客时间目标后，两者服务成本接近，说明时距目标收益不能直接解释为同等幅度的乘客时间收益。",
        8:"（3）针对目标环境中难以持续采集试错数据的问题，在混合离线与在线强化学习（Hybrid Offline-and-Online Reinforcement Learning，H2O）及其扩展H2O+基础上建立公交跨保真训练流程。固定SUMO离线数据与简易仿真交互共同支持策略训练，同时到站车辆编码、零与正停站混合动作及低载尾部保障构成完整控制器。在十个新环境块、三档需求和九种控制器的1350次配对评价中，完成保障相对相同学习策略的直接父控制器，使窗口内人均广义时间减少92.12秒、下降3.71%，完成率提高0.466个百分点。同更新预算比较支持混合训练方案相对纯离线方案的收益。进一步将有界速度调节与驻站联合建模，在相同数据来源下切换时距目标与乘客时间目标，为检验跨保真学习在不同公交控制权限和运营目标下的作用建立共同对照。",
        9:"（4）针对源目标转移关系失配，提出基于模型修正的世界模型方法（Model-Corrected World Model，MC-WM），以留出预测误差选择残差修正或直接目标模型，并以对数置信评价加权一步模型样本。在允许目标交互的三种接触多关节动力学（Multi-Joint dynamics with Contact，MuJoCo）变化任务中，相对仅目标数据软演员评论家方法（Soft Actor-Critic，SAC），平均回报分别提高1925.7、1184.5和133.4；执行器变化下，直接建模相对强制残差的回报增量为1529.9。公交扩展采用固定目标离线数据与混合事件模型，并以归一化置信备份缓解模型作用衰减。仅驻站、乘客时间目标下，在正常标定、联合50%和75%源参数偏差下，相对共同完成保障的Daganzo式规则，人均广义时间分别减少57.67、70.67和76.26秒。驻站—速度协同下，以乘客时间训练时，正常标定与联合75%偏差下相对H2O+分别降低3.48%和4.81%的广义时间，相对Daganzo式规则分别降低1.94%和4.13%；以时距训练时，相对规则分别降低4.87%和3.18%的平均时距惩罚。结果支持利用目标离线数据修正源仿真训练，并显示这一作用可在两种运营目标下获得。",
        15:"This thesis proposes a shared holding policy combining categorical embeddings with continuous observations and develops Robust Ensemble Soft Actor-Critic (RE-SAC) for conservative action-value learning. With a common learner, learned embeddings reduce generalized passenger time relative to one-hot encoding by 5.51%, 1.41%, and 4.91% under routine operations, demand surges, and rapid switching. Generalized time is twice waiting time plus onboard time. A crossed comparison covers two objectives, headway regularity and passenger time, and two action settings, holding alone and joint holding-speed control. Under joint control, RE-SAC reduces planned-headway deviation relative to an ensemble baseline without the specified regularization by 11.32%, 3.56%, and 5.93% in the three scenarios. When trained for passenger time, the corresponding reductions are 20.09%, 5.46%, and 14.27%. Conservative value learning therefore provides benefits beyond a single reward, while operational objectives retain distinct trade-offs.",
        16:"To address changing operating laws and bounded control authority, this thesis proposes Bayesian Adaptive Policy Regularization (BAPR), using environment identification and belief-based adjustment, and subsequently develops Causal-Symbolic Bayesian Adaptive Policy Regularization (CS-BAPR). The main bus configuration combines training stabilization with alternative policy-output structures. Across routine demand, single and repeated shocks, sustained high demand, and a brief extreme shock, the stabilized perceptron configuration reduces the mean headway-reward penalty relative to the preceding BAPR configuration by 58.82% to 69.79% for holding alone and by 15.50% to 28.56% for joint holding-speed control. Arithmetic outputs do not provide an additional benefit in every condition. When training instead targets passenger time, service costs remain close, so headway-reward gains cannot be interpreted as equal reductions in passenger time.",
        17:"A bus-specific cross-fidelity workflow builds on Hybrid Offline-and-Online Reinforcement Learning (H2O) and its extension H2O+, combining fixed SUMO offline data with new simplified-simulator experience. Simultaneous-arrival encoding, mixed zero-and-positive holding actions, and a low-load completion reserve form the complete controller. In 1,350 paired evaluations covering 10 new environment blocks, three demand levels, and nine controllers, the reserve reduces within-window generalized time by 92.12 seconds, or 3.71%, and increases completion by 0.466 percentage points relative to the direct parent with the same learned policy. An update-budget-matched comparison supports hybrid training over the tested offline-only scheme. A further extension combines holding with bounded speed adjustment and switches between headway and passenger-time objectives using the same data-access pattern, enabling comparisons across bus-control objectives and action settings.",
        18:"The Model-Corrected World Model (MC-WM) selects residual correction or direct target modeling using held-out prediction errors and weights one-step model samples through log-confidence evaluation. Across three controlled dynamics shifts in Multi-Joint dynamics with Contact (MuJoCo), with target interaction, mean return improves over target-only Soft Actor-Critic (SAC) by 1,925.7, 1,184.5, and 133.4. Under actuator shift, direct modeling gains 1,529.9 return over forced residual correction. The bus extension uses fixed target offline data and hybrid event models, with a normalized confidence backup limiting attenuation of model updates. For holding alone with a passenger-time objective, mean generalized time decreases relative to a Daganzo-style rule with the same completion reserve by 57.67, 70.67, and 76.26 seconds under calibrated, joint 50%, and joint 75% source-parameter mismatch. With joint holding-speed control and a passenger-time objective, reductions relative to H2O+ are 3.48% and 4.81% under calibrated and joint 75% mismatch conditions, and reductions relative to the Daganzo-style rule are 1.94% and 4.13%. With a headway objective, mean headway penalties decrease relative to that rule by 4.87% and 3.18%. These results support using target offline data to correct source-simulator training under both operational objectives.",
        513:"速度调节与驻站作用于不同物理过程：前者改变离站后的路段行驶，后者改变站点离开时刻。原有多线路离线学习、公交H2O+和MC-WM主实验仅学习驻站，速度倍率固定为1；实际车速仍受道路限速、信号和跟驰条件影响。在此基础上，第3.4.9、第4.4.7、第5.4.4和第6.4.4节补充驻站—速度协同控制。每个算法的比较组具有相同动作范围，不能把增加速度权限本身解释为算法收益。用于标定偏差的道路平均速度属于环境转移条件，与策略输出的速度倍率分别定义。",
        537:"具体输入维度取决于各研究的控制接口。多线路公交离线学习将15个原始观测量与上一驻站动作共同用于输入。H2O+最多组织16个同时事件槽位，原始打包状态为245维，规范化后为229维；有效槽位标记及批内已选驻站量共同描述决策上下文。联合速度控制时，再在规范化状态中加入16个槽位的已选速度输出，得到245维联合动作状态。两个245维表示分别对应规范化前的驻站状态和规范化后的联合状态，变量组成不同。这些维度由实际控制信息形成，并非所有公交任务必须使用同一维数。",
        562:"第3、4章原有公交实验以车头时距的计划偏差和前后对称性构成训练奖励，同时独立统计乘客服务指标；第5、6章原有跨保真实验以广义乘客时间为目标。补充实验在两类方法间交叉切换目标，使章节的比较不依赖于这一初始分工。车头时距目标沿用各方法已有奖励，H2O+与MC-WM采用第3.2.6节的计划对齐响应；乘客时间目标统一采用候车权重2、车内时间权重1。结果按实际优化目标解释，不把训练回报、计划偏差与乘客时间混为一个指标。",
        576:"跨保真比较报告式(2-16)的广义时间与完成率；以时距为奖励的补充比较同时报告平均时距惩罚及相对计划的绝对偏差。奖励惩罚包含前后偏差权重、对称性与大偏差项，计划绝对偏差则直接以秒计，两者排序不必完全相同。",
        1397:"零与正驻站是同一个物理量的两部分，不等于驻站与速度两个动作维度。原有单动作实验保持速度倍率为1；第5.2.7节再增加独立的有界速度输出。两类模型均保留精确零驻站概率，联合模型不会把零驻站改写成难以达到的连续边界。",
        1706:"本章原有公交比较仅输出附加驻站，执行端速度倍率为1。第6.4.4节进一步采用第5.2.7节的驻站—速度联合动作，MC-WM与H2O+共用状态和动作接口。世界模型辅助不赋予任何比较方法额外控制权限。偏差实验改变的道路平均速度、客流OD、路口延误及基本服务时间属于环境转移关系，不等于策略可选速度倍率；同组方法还保持相同驻站上限与完成保障。",
        1328:"输出结构与行为克隆交叉比较中的四组均已采用第4.3.2节的训练稳定化，因而表4-9检验的是稳定化之后的组件增量。第4.4.7节另与未采用整套稳定化的前序BAPR比较，并扩展奖励和动作范围，两种比较分别对应完整配置收益与输出结构的附加作用。",
        1335:"第4.4.1至4.4.6节及第4.4.7节的稳定化主比较没有启用符号发现和不变风险最小化约束。基于稀疏符号参考的补充比较在第4.4.7节末单独分析，其收益不与主配置合并，也不把NAU/NMU输出视为恢复了公交因果结构。附录D给出的策略响应界仍以完整映射的函数与导数条件为前提。",
        1339:"本章将非平稳公交运行写为隐含条件识别与有界控制响应问题，由BAPR的信念调节进一步研究CS-BAPR训练稳定化和策略结构。原有重复冲击比较揭示了控制回报、时距不均衡和候车时间之间的差异；新增两种动作、两种奖励比较进一步表明，相对前序BAPR，稳定化感知机在六种需求条件下均减小平均时距奖励惩罚，仅驻站下降58.82%至69.79%，驻站—速度协同下降15.50%至28.56%。算术输出与符号参考没有在所有条件下形成额外收益，乘客时间目标下的完整配置成本接近。由此，主要实证结论落在非平稳时距控制中的配置收益，而不是所有服务指标共同占优。",
        1671:"控制收益还取决于运营目标和动作权限。车头时距规则主要调节间隔，低载尾部保障为特定状态保留服务机会。原有比较在给定计划下仅选择驻站；第5.4.4节在共同的双动作接口中分别优化时距与乘客时间，说明同一跨保真训练方法可以服务于不同目标，但不能以某项评价指标替代实际奖励。",
        1682:"驻站—速度协同扩展保留目标离线数据与源仿真交互的数据分工，并将车头时距目标和乘客时间目标分开训练。双动作下，正常标定的H2O+取得低于Daganzo式规则的平均时距惩罚，但乘客时间目标下尚未超过该规则；高源偏差也改变其控制结果。相同动作在不同环境中的后果差异仍需处理。下一章研究MC-WM，将目标数据从策略训练进一步用于预测修正，比较残差模型与直接目标模型，并在相同动作、奖励和执行条件下检验服务增量。",
        1921:"本节将模型选择与贝尔曼可信度用于公交事件过程，在固定SUMO离线数据和新增源仿真经验下评价模型辅助。事件预测模型与归一化置信算法已在前文给出。第6.4.3节保留模型类别、损失系数及置信机制的比较，用于解释模型信息如何进入价值更新，并报告仅驻站的重复运行结果；第6.4.4节进一步在驻站—速度协同下，以H2O+和Daganzo式规则为主要参照，交叉考察两种奖励。早期单次接入、预算与偏差探索见附录F，按各自训练条件解释。",
        2053:"受控任务说明模型类别选择可以影响最终策略：执行器变化时，直接模型相对强制残差模型取得正的平均回报增量。公交比较进一步说明，模型如何被使用同样重要。原置信备份在预测误差基本稳定时仍出现后期梯度衰减，归一化机制恢复模型作用。仅驻站的乘客时间结果和双动作的两种目标比较共同支持模型修正的控制价值：在联合75%源偏差下，双动作MC-WM相对共同梯度约束的H2O+降低3.61%的广义时间和0.80%的平均时距惩罚；相对Daganzo式规则，两种相应训练目标的降幅分别为4.13%和3.18%。目标离线数据因此可以支持存在标定偏差的源仿真训练。",
        2058:"不同训练和评价批次分别报告，历史仅驻站结果不与新联合动作结果合并。第6.4.4节按独立重复运行和共同目标情形取平均，相关测试情形已在开发阶段使用，属于已有场景下的对照评价，不作为未见城市的泛化证明。结果表征完整模型辅助配置在对应数据和执行条件下的平均效果。",
        2059:"模型信息、观测损失尺度和置信使用方式是不同设计因素，第6.4.3.3及6.4.3.5节保留相应机制比较。双动作主比较固定共同的数据来源、动作范围和执行保障，时距训练还共同使用梯度范数约束，因而不能把仅对某一算法施加的训练稳定化误写为模型收益。上下文系数调节未形成稳定额外均值改善，固定模型项系数0.25仍作为主要公交配置。",
        2071:"进一步的仅驻站重复运行确认表明，固定模型项系数0.25在正常标定、联合50%和75%源偏差下，相对共同保障的Daganzo式规则分别减少57.67、70.67和76.26秒人均广义时间。双动作扩展中，乘客时间目标下相对H2O+的降幅为3.48%和4.81%，相对Daganzo式规则为1.94%和4.13%；时距目标下相对规则的平均惩罚降幅为4.87%和3.18%。联合75%偏差下，时距训练的MC-WM还比同样施加梯度约束的H2O+降低3.61%的乘客成本。模型辅助可用于两种公交目标，但乘客目标下的策略不必具有最低时距偏差。模型增加训练开销，部署仍仅使用演员网络。",
        2081:"本文由BAPR的条件识别和信念调节进一步提出CS-BAPR，在既定计划下研究训练稳定化与策略输出结构的作用。两种奖励与两种动作的需求扩展比较表明，稳定化感知机相对前序BAPR在六种条件下均改善平均时距奖励惩罚，仅驻站时降幅为58.82%至69.79%，驻站—速度协同时为15.50%至28.56%。持续高需求和重复冲击均包含在评价中，收益并非只由一次短时扰动体现。算术输出未在所有条件下增加收益；改用乘客时间奖励后，服务成本接近，说明方法的时距目标优势需要与乘客指标分别解释。",
        2089:"归一化置信备份保留未来置信传播，并以相应折扣的补因子缩放即时对数置信。仅驻站重复运行确认中，在正常标定、联合50%和75%源参数偏差下，固定模型项系数0.25相对共同保障的Daganzo式规则分别减少57.67、70.67和76.26秒人均广义时间。收益主要来自候车时间下降，平均乘客完成率同时提高；部分间隔指标仍不如规则。",
        2090:"驻站—速度联合控制进一步检验两种运营目标。以乘客时间训练时，MC-WM在正常标定与联合75%源偏差下，相对H2O+分别降低3.48%和4.81%的广义时间，相对Daganzo式规则分别降低1.94%和4.13%；以车头时距训练时，相对规则的平均时距惩罚分别降低4.87%和3.18%。在高偏差条件下，时距训练的MC-WM还比共同梯度约束的H2O+降低3.61%的广义时间。两种目标的结果共同表明，世界模型的公交应用不局限于一种奖励；其作用既取决于预测关系，也取决于预测进入价值更新的强度与事件结算的一致性。",
        2099:"针对源结构可能不适用于目标环境的问题，提出MC-WM，将源预测的保留写为残差与直接模型的显式选择，并以对数置信评价调节一步模型样本的损失权重。公交扩展按类别事件、计数、连续状态和持续时间组织预测，使固定目标离线数据参与事件模型学习；归一化置信备份则依据累积尺度性质维持模型作用。表6-5支持避免不适当源结构复用的价值；表6-13支持仅驻站条件下相对规则的服务收益，表6-16、表6-17及图6-18进一步给出驻站—速度协同、两种奖励下相对H2O+与Daganzo式规则的结果。理论分析、预测使用和控制评价由此形成连接。",
        2105:"世界模型与上下文调节并非越复杂越好。固定模型项系数0.25形成主要公交结果，上下文修复未获得一致的额外收益。模型拟合、置信更新和样本生成增加训练成本，部署演员结构则保持不变。各奖励与动作实验采用相应数据和转移定义，跨章节数值不能脱离需求强度、评价时域及统计分母直接排名。受控任务中相对直接目标模型方法的回报优势仍未分辨。"
    }
    for index,text in replacements.items():ed.replace(index,text)

    ed.insert(560,[p("联合动作仍采用归一化编码，驻站分量按式(2-11)映射到0至60秒，速度分量取−1至1并线性映射到0.8至1.2倍。速度命令在离站后生效，不改变已经完成的必要上下客服务，也不越过道路限速与跟驰约束。第3章原动作编码为0至60，两维输出在该编码下作等价映射；经验与价值网络始终接收同一动作编码，不把执行单位与训练单位混用。")])
    ed.insert(580,[p("转移区间随模型组织方式而异。车辆续接模型以同一车辆的相邻观测间隔结算乘客时间增量，不同车辆区间可能重叠；全局事件模型则按物理时钟顺序结算，同一时刻的中间转移成本为零。因此，前者的各车训练奖励不能直接相加作为全系统成本。所有补充比较均以独立的系统人数时间积分评价广义时间，在同一任务内保持共同分母和时域。")])
    ed.insert(758,[p("两种奖励与两种动作的补充比较不覆盖已有检查点。第3、4章仍从各自车辆续接接口训练，第5、6章联合动作采用包含速度干预的目标离线数据和全局事件接口。跨保真联合动作组均先完成50000次离线初始化，再采集约25000个源事件、进行20000次策略及评论家续训；目标SUMO训练交互为零。既有单动作比较与新增联合动作比较分别汇总，不将不同状态、动作或事件结算产生的数值合并。")])

    nodes=[p("3.4.9 两种运营目标与两种动作权限的交叉比较","h3"),
        p("前述实验主要以车头时距学习驻站策略。本节保持类别表征、集成规模、状态来源及100回合训练预算，在车头时距与广义乘客时间两种目标下，分别比较仅驻站和驻站—速度协同。每组独立重复训练，固定末次策略，在正常运行、客流激增和快速切换下各评价三个共同运行日；先平均运行日，再对重复训练等权平均。参照为相同集成学习器移除RE-SAC指定正则项的配置，除所比较项外不改变网络和动作范围。"),
        p("时距目标用前后相对计划的平均绝对偏差评价，乘客时间目标用两倍候车时间与车内时间之和评价。表3-10分别列示两类目标，不用时距回报代替乘客成本。两种动作的比较组均拥有相同速度权限，算法差异与动作扩展因而分别解释。"),
        p("表3-10 RE-SAC在两种奖励与动作权限下的目标指标","tablecaption","tbl3_10")]
    names={"normal":"正常","demand_surge":"客流激增","fast_switch":"快速切换"}
    rows=[["训练目标","动作权限","运行条件","集成参照/秒","RE-SAC/秒","降幅/%"]]
    for r in data["resac"]:
        rows.append(["计划时距" if r["objective"]=="headway" else "乘客时间",
            "仅驻站" if r["mode"]=="holding" else "驻站与速度",names[r["scene"]],
            f'{r["control"]:.2f}',f'{r["resac"]:.2f}',f'{r["reduction"]:.2f}'])
    nodes += [ed.table(rows,[1.05,1.15,1.1,1.25,1.15,.85]),
        p("时距行以秒计的计划绝对偏差为指标，乘客行以秒/已出发乘客计的广义时间为指标；降幅以同一行集成参照均值为分母。"),
        ed.figure("fig3_13_objectives_actions","RE-SAC两种奖励与两种动作的目标指标降幅"),
        p("图3-13 RE-SAC在两种动作下对相应训练目标的改善","figurecaption","fig3_13"),
        p("（a）以车头时距为奖励时的计划偏差降幅；（b）以乘客时间为奖励时的广义时间降幅。图中采用全部重复运行均值，柱高对应表3-10。"),
        p("仅驻站时，时距目标下的计划偏差在三个场景均下降，乘客目标下的广义时间分别降低14.20%、4.02%和11.29%。增加速度输出后，时距目标的计划偏差降幅分别为11.32%、3.56%和5.93%；乘客目标的时间降幅分别为20.09%、5.46%和14.27%。这表明指定正则化的平均收益并不只出现在原有驻站和时距奖励设置中。"),
        p("不同目标仍产生不同运行选择。正常双动作下，时距训练的RE-SAC计划偏差为37.54秒，乘客训练时为65.71秒；后者广义时间更低，分别为1305.44与1422.97秒。两类策略不是对同一指标的优劣替换：前者重视间隔参照，后者允许部分间隔偏离以减少服务时间。因而，后续章节切换到乘客目标并不否定时距控制，而是对算法在另一运营目标下的检验。")]
    ed.insert(1107,nodes)

    ed.insert(1412,[p("5.2.7 驻站—速度联合动作与可选运营目标","h3"),
        p("联合控制在原有零与正驻站混合策略外增加速度分量，物理范围为0.8至1.2倍，驻站仍为0至60秒。速度分量以状态条件高斯变量经双曲正切变换产生，精确零驻站概率保持不变。批内前车已选驻站和速度均进入后续车辆的决策上下文，收齐全部动作后才推进物理时间。规范化的229维驻站状态追加16个已选速度分量，形成245维联合动作状态，编码器输出维数仍为256。"),
        p("目标离线数据包含联合动作干预，经验始终保存完整两维策略动作。驻站与速度按同一时序在SUMO和简易仿真执行，离站后速度命令不能倒过来改变本次必要服务。两个环境采用一致的全局事件结算，批内零时长转移不重复增加乘客成本，终止前最后一段实际经过时间也计入对应经验。由此，离线初始化与源仿真续训的状态、动作、奖励和后继状态具有相同物理含义。"),
        p("两类奖励共用这一转移接口。乘客目标采用式(2-13)，时距目标按第3.2.6节的计划对齐奖励在后继决策状态计算，包含前后偏差、对称性和大偏差项；终止转移的时距奖励置为零。目标切换不改变数据可获得条件：训练仅使用SUMO离线数据与简易仿真交互，SUMO新增运行只用于评价。"),
        p("Daganzo式规则在联合动作比较中保留其前后时距比例驻站反馈，速度倍率固定为1。该参照表示不依赖训练的传统驻站方案；学习策略额外使用有界速度调节，因而相对规则的结果评价的是联合控制方案，而相对同样双动作的H2O+或MC-WM才比较相同权限下的学习方法。")])

    ed.insert(1329,[p("4.4.7 两种目标与动作下的需求扩展比较","h3"),
        p("为区分完整训练稳定化的收益与输出头的附加作用，本节比较前序BAPR、稳定化感知机CS-BAPR及稳定化算术输出CS-BAPR。前序配置不采用第4.3.2节的六项稳定化，其余任务、表征和100回合末次策略选择相同；两类稳定化配置沿用已完成训练。分别以时距和乘客时间为目标，采用仅驻站或驻站—速度协同动作，每个固定策略在六种需求条件下各运行三个共同评价日，结果按独立重复运行取均值。"),
        p("评价时域为18000秒。单次5倍冲击发生于3600至5400秒；重复5倍冲击分别发生于3600至5400、9000至10800和14400至16200秒。持续5倍与持续10倍从时域起点保持至终点；50倍短时冲击仅持续3600至4200秒，不解释为全时域50倍需求。表4-10以每次决策的负奖励绝对值表示平均时距惩罚，另列计划绝对偏差的变化，以区分奖励设计与直接运行指标。"),
        p("表4-10 稳定化感知机相对前序BAPR的时距目标结果","tablecaption","tbl4_10"),
        ed.table(cs_headway_rows(data),[1.05,1.05,1.12,1.12,1.12,1.3]),
        p("时距惩罚为负奖励的绝对值，降幅为前序BAPR减CS-BAPR后除以前序BAPR；计划偏差变化为CS-BAPR减BAPR的百分比，负值表示偏差减少。两项指标均先对各次运行取平均。"),
        ed.figure("fig4_11_stabilization","六种需求条件下完整稳定化配置的时距惩罚改善"),
        p("图4-11 六种需求条件下CS-BAPR完整配置的时距惩罚降幅","figurecaption","fig4_11"),
        p("各柱均相对同动作、同奖励的前序BAPR。两面板使用相同横轴范围；不同输出头均包含完整训练稳定化。"),
        p("稳定化感知机在六种条件下均减小平均时距惩罚。仅驻站时降幅为58.82%至69.79%，正常条件的计划绝对偏差由162.18秒降至103.00秒，减少36.49%；联合动作时惩罚降幅为15.50%至28.56%。然而，双动作下部分计划绝对偏差并未同步下降。原生奖励同时考虑偏差权重与前后对称性，不能以奖励改善替代每项时距指标改善。整体配置与前序方法的比较说明了稳定化收益；两个稳定化输出头的比较则显示，算术结构并未在每种动作条件下继续提高收益。"),
        p("表4-11 乘客时间目标下稳定化感知机相对前序BAPR的广义时间变化","tablecaption","tbl4_11"),
        ed.table(cs_passenger_rows(data),[1.8,1.5,1.5]),
        p("表中为CS-BAPR减前序BAPR后除以前序BAPR的百分比，负值表示乘客成本较低。"),
        p("乘客目标下，六种需求的时间变化均小于0.3%，两种动作的结果整体接近。以正常联合动作为例，BAPR与稳定化感知机的广义时间分别为11297.46和11291.87秒，差异约0.05%。因此，本组实验的主要优势仍体现为时距目标下的策略学习；乘客目标结果用于检验目标切换后的控制表现，不支持把时距惩罚的大幅下降写成等比例乘客时间收益。"),
        p("另对稀疏符号参考的适配进行独立比较。将固定参考动作改为随投影状态学习的上下文参考，使神经残差在参考状态为零，而完整动作仍可随需求和载客条件变化。相对原固定参考的符号配置，时距惩罚平均减少约0.69%至4.28%；相对同样采用上下文参考但不施加符号约束的配置，正常与短时冲击下方向有利，持续5倍、10倍及50倍短时冲击下却退化。末期熵温度约为0.017，未出现持续放大。可学习参考缓解了固定动作与状态响应的冲突，但符号约束的方向质量仍限制外推收益，故主结果采用稳定化配置，符号适配不与其合并。")])

    ed.insert(1668,[p("5.4.4 双动作下两种运营目标的跨保真结果","h3"),
        p("本节采用第5.2.7节的联合动作接口，固定目标离线数据，完成50000次离线初始化后，以简易仿真采集约25000个事件并新增20000次策略与评论家更新。比较正常标定和联合75%源偏差两类训练条件；后者将道路平均速度设为原值的0.25倍、OD与基本服务时间设为1.75倍、额外路口遭遇密度设为0.75。目标SUMO始终不变，也不参与训练交互。两档目标需求分别采用1.00倍和1.25倍，每档五个共同场景，在24000秒时域内评价固定策略。"),
        p("乘客时间与时距目标分别训练。时距训练共同对演员、评论家和价值函数的梯度范数施加上限10，保留原奖励零点与计划尺度；乘客目标采用原训练设置。各方法都使用相同完成保障，Daganzo式规则按第6.4.3.7节定义且速度倍率为1。表5-13为独立重复训练后的场景均值，两个目标的指标分别计算，不直接比较它们的数值大小。"),
        p("表5-13 双动作H2O+在两种目标下与Daganzo式规则的比较","tablecaption","tbl5_13"),
        ed.table(h2o_rows(data),[1.15,1.35,1.15,1.35,1.0]),
        p("乘客目标的指标为人均广义时间，单位秒/已出发乘客；时距目标为每次决策的平均时距惩罚。降幅为规则减H2O+后除以规则，正值表示H2O+较低。规则使用同一组目标场景，不随源仿真条件重复训练。"),
        p("正常标定下，H2O+时距惩罚比Daganzo式规则低4.47%；联合75%偏差下仍低2.40%。这说明一致的事件编码与联合动作可以支持以车头时距为目标的跨保真训练，并非混合学习只能优化乘客时间。但乘客目标下H2O+的广义时间分别为2727.62和2703.55秒，高于规则的2684.64秒，目标切换与增加速度输出本身不足以保证服务成本优势。第6.4.4节在完全相同的动作和评价权限下加入模型修正，检验目标数据能否进一步改善这两类结果。")])

    ed.insert(1710,[p("联合动作扩展使预测器同时接收驻站和速度两维条件。已选速度与已选驻站一样按批内规则写回状态，不交由模型猜测；模型预测物理边界后的事件变化。乘客奖励由预测的非负时间成本形成，时距奖励由预测后继事件的时距量按同一计划对齐函数计算。第6.4.4节比较的MC-WM与H2O+采用相同联合状态与动作数据；模型仅增加训练期的一步预测信息，不在部署时生成速度命令或接入额外目标数据。")])
    ed.insert(2050,[p("6.4.4 驻站—速度协同下两种奖励的对照结果","h3"),
        p("本节在第5.4.4节的共同预算、动作和评价场景中比较MC-WM、H2O+及Daganzo式规则。MC-WM沿用归一化置信备份，真实混合项与模型项系数分别为0.75和0.25，模型根据已有目标离线数据拟合、更新和选择。两种学习方法复用共同离线初始化，均只在简易仿真新增训练经验；模型拟合不增加SUMO在线训练。乘客目标保持已有配置，时距目标对两种方法共同采用梯度范数上限10，以约束带计划尺度奖励的联合动作更新。"),
        p("结果先对每次训练的十个目标情形取平均，再对独立重复训练等权平均。完成保障在三种方法中相同，固定末次策略，不根据某个测试情形替换检查点。Daganzo式规则仅采用驻站比例反馈，H2O+与MC-WM均采用联合动作；相对规则反映联合控制方案，相对H2O+则比较共同动作权限下的模型修正。"),
        p("表6-16 乘客时间奖励下双动作策略的乘客成本与时距惩罚","tablecaption","tbl6_16"),
        ed.table(dual_rows(data,"passenger"),[1.35,1.05,1.35,1.35]),
        p("正常标定下，MC-WM相对H2O+减少94.94秒，降幅3.48%；联合75%偏差下减少129.91秒，降幅4.81%。相对Daganzo式规则，分别减少51.95和110.99秒，降幅1.94%和4.13%。高源偏差下仍能降低乘客成本，说明目标离线数据学习的模型可为源仿真中的价值更新补充有用信息。"),
        p("表6-17 车头时距奖励下双动作策略的平均控制结果","tablecaption","tbl6_17"),
        ed.table(dual_rows(data,"headway"),[1.2,1.0,1.22,1.3]),
        p("平均时距惩罚为负奖励绝对值，越小越好；广义时间为完整24000秒窗口内的独立人数时间积分。两项指标不互相替代。"),
        ed.figure("fig6_18_dual_objectives","两种奖励下MC-WM和H2O+相对Daganzo式规则的目标指标降幅"),
        p("图6-18 双动作学习在两种奖励下相对Daganzo式规则的平均收益","figurecaption","fig6_18"),
        p("（a）乘客奖励对应的广义时间降幅；（b）时距奖励对应的平均时距惩罚降幅。两面板零线均表示规则水平，正值为改善；采用共同场景和全部重复运行均值。"),
        p("时距目标下，MC-WM在正常标定和联合75%偏差下的平均惩罚为309.11与314.58，比Daganzo式规则低4.87%和3.18%，也分别比共同梯度约束的H2O+低0.42%和0.80%。相应计划绝对偏差分别为148.15和148.50秒，H2O+为148.92和150.06秒；模型组对这一直接运行指标也有小幅改善。"),
        p("高源偏差下，时距训练的MC-WM广义时间为2614.78秒，比H2O+的2712.57秒减少97.80秒、下降3.61%，比规则低2.60%；正常标定下，两种学习方法的时间仅相差1.48秒，MC-WM略高0.06%。以时距训练并不保证每项乘客指标均更优，但本组结果同时给出了平均时距收益和高偏差下的乘客收益。"),
        p("两种奖励产生不同折中。正常标定下，乘客训练的MC-WM广义时间低于时距训练，但其计划绝对偏差较大；这一现象与第3.4.9节一致。论文前后各章评价重心的变化因而来自明确的运营目标选择，而非算法失去可比性。单动作既有乘客结果与本节联合动作结果共同表明，MC-WM的模型修正可以支持驻站控制，也可以支持驻站—速度协同；目标改变时应重新定义并解释奖励与服务指标，而不是只追求一个统一回报数值。")])
    ed.save()


def cs_get(data,mode,objective,scene,arm):
    return next(r for r in data["cs"] if (r["mode"],r["objective"],r["scene"],r["arm"])
                ==(mode,objective,scene,arm))


CS_SCENES={"normal":"正常","pulse5":"单次5倍","repeated5":"重复5倍", "static5":"持续5倍",
           "static10":"持续10倍","burst50":"短时50倍"}


def cs_headway_rows(data):
    rows=[["运行条件","动作权限","BAPR惩罚","CS-BAPR惩罚","惩罚降幅/%","计划偏差变化/%"]]
    for mode in ("holding","dual"):
        for scene,name in CS_SCENES.items():
            b=cs_get(data,mode,"headway",scene,"pristine_mlp")
            c=cs_get(data,mode,"headway",scene,"stabilized_mlp")
            change=100*(c["plan_mae"]-b["plan_mae"])/b["plan_mae"]
            gain=100*(c["recorded_reward_per_decision"]-b["recorded_reward_per_decision"])/abs(b["recorded_reward_per_decision"])
            rows.append([name,"仅驻站" if mode=="holding" else "驻站与速度",
                f'{-b["recorded_reward_per_decision"]:.2f}',f'{-c["recorded_reward_per_decision"]:.2f}',f'{gain:.2f}',f'{change:+.2f}'])
    return rows


def cs_passenger_rows(data):
    rows=[["运行条件","仅驻站变化/%","驻站与速度变化/%"]]
    for scene,name in CS_SCENES.items():
        row=[name]
        for mode in ("holding","dual"):
            b=cs_get(data,mode,"passenger",scene,"pristine_mlp")
            c=cs_get(data,mode,"passenger",scene,"stabilized_mlp")
            row.append(f'{100*(c["cost"]-b["cost"])/b["cost"]:+.3f}')
        rows.append(row)
    return rows


def h2o_rows(data):
    rows=[["训练目标","源仿真条件","H2O+指标","规则指标","降幅/%"]]
    for objective,metric in (("passenger","cost"),("headway","ridge")):
        for r in data["dual"]:
            if r["objective"]==objective and r["arm"]=="h2oplus":
                b=abs(data["daganzo"][metric]);v=abs(r[metric])
                rows.append(["乘客时间" if objective=="passenger" else "车头时距",
                    "正常标定" if r["scene"]=="calibrated" else "联合75%偏差",
                    f'{v:.2f}',f'{b:.2f}',f'{100*(b-v)/b:+.2f}'])
    return rows


def dual_rows(data,objective):
    rows=[["源仿真条件","控制方法","广义时间/秒","平均时距惩罚"]] if objective=="passenger" else [
        ["源仿真条件","控制方法","平均时距惩罚","广义时间/秒"]]
    for scene in ("calibrated","joint75"):
        name="正常标定" if scene=="calibrated" else "联合75%偏差"
        for arm,label in (("h2oplus","H2O+"),("mcwm","MC-WM")):
            r=next(r for r in data["dual"] if (r["scene"],r["objective"],r["arm"])==(scene,objective,arm))
            rows.append([name,label,f'{r["cost"]:.2f}',f'{-r["ridge"]:.2f}'] if objective=="passenger"
                        else [name,label,f'{-r["ridge"]:.2f}',f'{r["cost"]:.2f}'])
        rows.append([name,"Daganzo规则",f'{data["daganzo"]["cost"]:.2f}',f'{-data["daganzo"]["ridge"]:.2f}'] if objective=="passenger"
                    else [name,"Daganzo规则",f'{-data["daganzo"]["ridge"]:.2f}',f'{data["daganzo"]["cost"]:.2f}'])
    return rows


if __name__=="__main__":main()
