from pathlib import Path
from shutil import copy2
from zipfile import ZipFile
import re
import xml.etree.ElementTree as ET

from edit_section_1_2_1 import DOCUMENT, W, M, W14, revise_paragraph


BACKUP = Path(__file__).with_name("校订前_1.2.2.docx")

# Prose segments retain the positions of the existing citation objects.
REVISIONS = {
    280: (
        "强化学习通过运行经验改进决策，可按动作选择方式、是否使用预测模型以及如何评价不利结果区分。基于价值的方法先估计动作的长期收益，再选择动作；直接策略优化改进状态到动作的映射；演员评论家方法由演员产生动作、评论家估计收益，再据此更新演员。",
        "有模型方法学习环境转移，并用预测进行规划或补充训练经验；无模型方法不显式学习这种转移模型。鲁棒学习则关注不确定条件下的不利后果，可以与上述方法结合。",
    ),
    281: (
        "策略表征研究关注如何构造状态到动作的映射",
        "，探索策略研究关注如何在尝试新动作与利用已有经验之间权衡。",
        "深度Q网络（Deep Q-Network，DQN）从观测学习离散动作价值；深度确定性策略梯度（Deep Deterministic Policy Gradient，DDPG）结合演员评论家和经验回放，处理连续动作。",
        "信赖域策略优化（Trust Region Policy Optimization，TRPO）和近端策略优化（Proximal Policy Optimization，PPO）限制策略更新幅度，以改善学习稳定性。",
        "这些方法为离散驻站时长选择和连续驻站输出提供了求解基础，但车辆状态、奖励目标和动作范围仍需依据公交任务定义。",
    ),
    282: (
        "学习型驻站控制利用运行交互拟合动作价值或策略，减少对预先指定的解析控制律的依赖。",
        "Alesiani和Gkiotsalitis研究高频服务中的强化学习驻站；Wang和Sun将动态驻站建模为多智能体强化学习，使车辆根据运行反馈改进决策。",
        "这类方法优化长期回报，也需要表示本车动作与其他车辆运行之间的相互影响。",
    ),
    283: (
        "解析预测与学习可以共同用于驻站决策。He等将多阶段前瞻与近似动态规划、Q学习结合，预测驻站对后续运行的影响，并以神经网络近似价值、积累控制经验。",
        "与仅依据当前车头时距的反馈相比，该方法评价未来运行；与每次重新求解完整预测问题相比，价值近似又能复用已有经验。因此，学习型控制不仅改变控制律的形式，也改变预测和历史经验参与决策的方式。",
    ),
    284: (
        "不同车辆的到站时刻通常不同，公交驻站决策具有异步性。车辆动作又共同影响后续服务，因而需要判断各次动作对整体回报的贡献，这称为信用分配。Wang和Sun采用异步多智能体框架及图注意力评价网络，表示其他车辆事件的影响。",
        "Rodriguez等在联合驻站与跳站任务中扩展状态和奖励，使策略能够观察邻近车辆的作用，并在事件仿真中评价合作学习和执行不确定性的影响。",
    ),
    285: (
        "一般多智能体强化学习（Multi-Agent Reinforcement Learning，MARL）同样需要评价联合动作、区分个体贡献，并处理多个策略同时学习造成的非平稳性。多智能体深度确定性策略梯度（Multi-Agent Deep Deterministic Policy Gradient，MADDPG）在训练评论家时利用其他智能体的信息；反事实多智能体策略梯度（Counterfactual Multi-Agent Policy Gradients，COMA）利用反事实基线评价个体动作对团队回报的贡献。",
        "价值分解网络（Value-Decomposition Networks，VDN）将个体价值相加，单调价值混合网络（QMIX）采用单调混合结构，两者以不同方式连接团队评价和个体决策。",
        "将这些方法用于公交时，还需表示到站时序、班次车辆复用，以及乘客需求对后续站点和车辆的影响。",
    ),
    286: (
        "相关多智能体综述分别从控制与决策方法",
        "、可扩展性与可迁移性",
        "以及博弈和控制的联系",
        "梳理协同学习。公交应用中需要区分两种非平稳性：其他车辆的策略不断更新，使当前学习者面对的交互过程改变；客流或道路规律变化，则使外部环境改变。多智能体非平稳性研究主要通过集中训练、其他智能体策略表征和通信等处理前一问题。",
        "本文采用共享策略，减少多套独立策略同时更新的复杂性；第4章的适应机制则针对外部运行规律变化，二者不能互相替代。",
    ),
    287: (
        "多线路运行还涉及线路间协调及不同决策层次之间的配合，分层强化学习提供了组织这些决策的一种方式。",
        "Wang和Sun针对共享走廊，将多线路信息加入状态和奖励，通过偏好权重生成机制处理多目标评价。",
        "Yu等结合驻站与专用车道加速，由上层选择控制类型、下层确定驻站时长或加速强度。",
        "Li等研究繁忙多线路走廊的站点排队，以图表示车辆之间的关系。",
        "这些研究分别处理车辆协作、线路协作和动作层次协调，控制范围不同，所需状态与评价目标也不同。",
    ),
    288: (
        "本文以前期单智能体公交研究为基础，将类别特征嵌入与连续运行状态共同输入共享策略。车辆、站点、方向和时段等类别信息，使同一策略能够区分不同控制对象，处理双向、按时刻表运行中的经验不均衡。",
        "在给定计划下，车队各车辆的到站决策由同一策略处理，不再为每辆车单独训练完整策略。后续研究在这一控制基础上，逐步处理随机价值估计、运行规律变化和源目标环境差异。",
    ),
    289: (
        "类别特征嵌入用可学习向量保留对象差异。Guo和Berkhahn使类别之间与预测任务有关的关系从数据中形成；多任务学习研究讨论如何兼顾共享表征与任务差异。",
        "公交类别编号表示身份或服务位置，不表示具有大小关系的连续物理量。例如，站点编号增大并不意味着需求按相同比例增加。嵌入使策略在共享控制经验的同时保留这些差异，其具体作用由第3章的表征比较检验。",
    ),
    290: (
        "学习配置与奖励选择也影响公交控制效果。Xu等系统评价用于缓解串车的强化学习配置；Yu等利用大语言模型生成并迭代调整驻站奖励。",
        "本文从计划间隔与前后车关系分析奖励函数，区分间隔均衡和计划对齐，再通过运营结果考察奖励选择是否同时改善乘客时间。",
    ),
    291: (
        "交通信号研究同样重视将交通机理纳入强化学习。",
        "PressLight依据最大压力控制构造状态与奖励，把动作评价与上下游排队联系起来；CoLight通过图注意力汇集相邻路口信息，学习网络协作。",
        "它们分别将交通关系用于学习目标和状态表示。公交驻站也需考虑相邻车辆与上下游服务，但决策由车辆到站触发，驻站还直接增加车内乘客时间。因此，本文采用车辆事件状态和乘客服务评价，而非直接沿用信号相位控制的建模方式。",
    ),
    292: (
        "学习方法还需区分环境随机性与估计不足。客流到达和车辆运行具有随机波动；有限数据又使学习器不能准确掌握动作后果。Depeweg等研究贝叶斯模型中的不确定性分解及风险敏感学习；分布强化学习则学习回报的概率分布，而不只估计平均回报。",
        "在公交应用中，Wang和Sun结合回报分布表示与元学习模块，研究交通扰动、服务中断和需求突增下的鲁棒动态控制。",
    ),
    293: (
        "价值集成用多个评价网络估计同一动作的收益。自助采样深度Q网络（Bootstrapped Deep Q-Network，Bootstrapped DQN）利用多个价值输出分支支持探索；双延迟深度确定性策略梯度（Twin Delayed Deep Deterministic Policy Gradient，TD3）用两个目标评价网络中的较小值等机制缓解价值高估；随机集成双Q学习（Randomized Ensembled Double Q-Learning，REDQ）结合价值集成、随机子集最小值和较高更新数据比学习，其中更新数据比指每新增一条交互转移执行的学习更新次数。",
        "软演员评论家方法（Soft Actor-Critic，SAC）将策略熵纳入目标，在预期回报与随机探索之间权衡。",
        "集成分歧、较小值聚合和熵调节分别影响估计、学习目标与探索行为，作用不同。",
    ),
    294: (
        "Kendall和Gal以及Hüllermeier和Waegeman将随机观测或环境过程带来的不确定性称为偶然不确定性（aleatoric uncertainty），将知识不足带来的不确定性称为认知不确定性（epistemic uncertainty）。",
        "Dropout近似贝叶斯推断和深度集成提供了不同的预测不确定性估计方法。",
        "在序贯决策中，估计对象还可能是环境转移或累计价值；Clements等针对学习得到的Q值区分风险与估计不确定性。",
        "因此，评论家之间的分歧可以用于调节学习，但不能直接视为已经分离出的环境随机方差。",
    ),
    295: (
        "回报分布表示与价值集成可以结合。分位数回归和隐式分位数网络分别学习离散分位点与分位函数，使风险偏好作用于回报分布。",
        "截断分位数评价器方法（Truncated Quantile Critics，TQC）结合分位数表示、截断和集成调节过估计；评价器多样化集成方法（Ensemble-Diversified Actor-Critic，EDAC）在固定数据学习中增强价值网络的差异，以处理数据覆盖之外的动作。",
        "前者描述回报结果的分布，后者利用多个网络的估计差异；本文的集成保守评价主要沿后一思路改进公交动作选择。",
    ),
    296: (
        "鲁棒强化学习在允许的环境模型集合内考察不利结果。鲁棒动态规划用最不利转移评价动作，其矩形性条件允许各状态动作对的转移分布独立选择；正则化马尔可夫决策过程（Markov Decision Process，MDP）则在价值或策略目标中加入正则项。",
        "Derman等建立了指定奖励、转移不确定集合与相应策略、价值正则化之间的联系，为计算鲁棒目标提供了另一种表达。",
        "作者提出的鲁棒集成软演员评论家方法（Robust Ensemble Soft Actor-Critic，RE-SAC）由此引入目标价值偏移与集成保守评价。第3章分析这两类修正对驻站动作的作用，附录A.7说明其与鲁棒目标之间的关系及成立条件。",
    ),
    297: (
        "风险敏感评价进一步规定应多大程度关注较差的运行结果。条件风险价值（Conditional Value-at-Risk，CVaR）描述给定尾部比例内的不利平均成本；将均值与CVaR联合优化，可以兼顾总体表现与尾部后果。CVaR也可表示为在受限概率重加权范围内取最不利期望，与分布鲁棒优化（Distributionally Robust Optimization，DRO）相联系。",
        "这些关系为保守评价提供依据，但不意味着任意网络惩罚都等价于CVaR。本文据此分析给定运行规律下的保守价值学习；运行规律本身改变时，还需要识别当前条件。",
    ),
    298: (
        "变化识别用于判断近期运行记录是否仍符合原有规律。贝叶斯在线变化点检测维护距最近一次变化所经历的步数分布，据此更新对历史信息的判断。",
        "识别到变化后，策略还需对新的状态范围作出合理响应。神经加法单元（Neural Addition Unit，NAU）与神经乘法单元（Neural Multiplication Unit，NMU）通过约束加法和乘法结构研究数值外推，为结构化策略输出提供了启发。",
        "第4章据此区分两个问题：近期证据怎样影响学习，以及策略的函数结构怎样影响变化后的驻站响应。",
    ),
    299: (
        "非平稳强化学习研究转移或奖励规律随时间改变时的决策。开放多智能体研究进一步区分任务、环境和智能体组成变化。",
        "Padakandla梳理动态环境中的适应方法；Khetarpal等从变化范围、变化原因及持续学习评价组织相关研究。",
        "Lecarpentier和Rachelson在转移、奖励随时间有界变化的条件下研究最不利规划。",
        "对应公交任务，大型活动散场、临时管制和持续需求增长具有不同变化幅度与持续时间，应分别考察单次冲击、反复变化和长期偏移。已知的日内高峰可以由时段状态描述，不能仅因需求随时刻改变，就认定条件转移规律未知或发生了非平稳变化。",
    ),
    300: (
        "环境识别方法从已经发生的状态转移推断隐含运行条件。隐参数马尔可夫决策过程（Hidden Parameter Markov Decision Process，HiP-MDP）以潜在参数描述相关任务的动力学差异；通用策略与在线系统辨识（Universal Policy with Online System Identification，UP-OSI）根据近期状态动作历史推断系统参数，再将其输入通用策略。",
        "概率嵌入式演员评论家方法（Probabilistic Embeddings for Actor-Critic Reinforcement Learning，PEARL）用概率上下文区分任务推断与控制；变分贝叶斯自适应深度强化学习（Variational Bayes-Adaptive Deep Reinforcement Learning，VariBAD）将任务不确定性用于动作选择；快速运动适应方法（Rapid Motor Adaptation，RMA）结合基本策略与适应模块。",
        "作者的BAPR前期研究（Bayesian Amnesic Piecewise-Robust）结合环境识别、模式后验更新与保守价值学习。",
        "本文沿此基础研究贝叶斯自适应策略正则化（Bayesian Adaptive Policy Regularization，BAPR），并进一步提出因果符号贝叶斯自适应策略正则化（Causal-Symbolic Bayesian Adaptive Policy Regularization，CS-BAPR）",
        "。公交训练根据近期奖励和价值分歧调节保守程度，并研究结构化驻站输出；受控执行器实验则另外检验模式识别后的动作补偿。",
    ),
    301: (
        "隐含环境推断也将非平稳控制与部分可观测性、元学习联系起来。",
        "若改变转移规律的模式不能直接观测，可以在给定模式演化规律下，将运行状态和模式共同构成部分可观测马尔可夫决策过程（Partially Observable Markov Decision Process，POMDP）。Ghosh等讨论未知环境引起的隐式部分可观测性；VariBAD通过跨任务训练近似任务后验，并将其用于决策。",
        "POMDP描述控制器缺少哪些信息，后验更新表示近期证据如何改变模式判断，元学习则从多个任务中学习适应方法。第4章利用这一联系组织条件识别，但不将公交中的近期统计量调节等同于已经实施跨任务元训练。",
    ),
    302: (
        "因果建模研究关注环境关系对强化学习控制的作用",
        "；符号辨识则以少量可写成公式的关系描述转移。非线性动力系统稀疏辨识（Sparse Identification of Nonlinear Dynamics，SINDy）在候选函数中选择少量有效项，拟合动力学；不变风险最小化（Invariant Risk Minimization，IRM）研究在多个环境中保持一致的预测关系。",
        "这些方法启发本文以符号参考约束驻站策略的状态响应，使训练范围外的动作仍有明确关系可供参照。附录D分析参考拟合误差、策略导数误差与外推偏移之间的关系。符号关系是否准确、是否改善实际控制仍需分别检验，稀疏拟合本身也不等于因果识别。",
    ),
    303: (
        "控制约束与回报目标也需分别定义。",
        "约束策略优化以期望累计成本约束策略更新，提供了同时描述收益与约束的方法。",
        "本文分别规定驻站动作范围、统计运行违例和评价行程完成率，避免仅凭较高回报推断各项运营要求已经满足。",
    ),
    304: (
        "综上，演员评论家能够表达连续驻站动作，并利用已有经验评价后续收益，是本文各学习方法的共同基础。在训练前未获得目标域转移数据的条件下，第3章用类别共享策略与RE-SAC处理给定运行规律下的随机波动；第4章进一步通过BAPR的条件识别与CS-BAPR的结构化策略，处理运行规律变化。获得固定目标离线数据后，问题转为怎样结合目标记录与源仿真交互：第5章将H2O+适配于公交驻站，第6章再用MC-WM研究源仿真标定不足时的预测修正。下面继续梳理与这两类数据利用方式有关的仿真迁移研究。",
    ),
}


def main():
    with ZipFile(DOCUMENT) as source:
        original = source.read("word/document.xml").decode("utf-8")
        body = ET.fromstring(original).find(W + "body")
        assert "".join(body[279].itertext()) == "1.2.2 强化学习控制"
        assert "".join(body[305].itertext()) == "1.2.3 仿真迁移"
        updated = original
        for index, prose in REVISIONS.items():
            paragraph_id = body[index].get(W14 + "paraId")
            pattern = r'<w:p\b[^>]*w14:paraId="' + paragraph_id + r'"[^>]*>.*?</w:p>'
            matches = list(re.finditer(pattern, updated, flags=re.S))
            assert len(matches) == 1, (index, paragraph_id)
            match = matches[0]
            revised = revise_paragraph(match.group(0), body[index], prose)
            mask = lambda raw: re.sub(r"(<w:t(?:\s[^>]*)?>).*?(</w:t>)", r"\1\2", raw, flags=re.S)
            assert mask(match.group(0)) == mask(revised)
            updated = updated[:match.start()] + revised + updated[match.end():]
        new_body = ET.fromstring(updated).find(W + "body")
        assert len(body) == len(new_body)
        changed = [i for i, (a, b) in enumerate(zip(body, new_body)) if ET.tostring(a) != ET.tostring(b)]
        assert changed == list(REVISIONS), changed
        for index in changed:
            for tag in (W + "hyperlink", M + "oMath", W + "pPr"):
                assert [ET.tostring(x) for x in body[index].iter(tag)] == [ET.tostring(x) for x in new_body[index].iter(tag)]
        if not BACKUP.exists():
            copy2(DOCUMENT, BACKUP)
        temporary = DOCUMENT.with_suffix(".section-edit.docx")
        with ZipFile(temporary, "w") as target:
            for item in source.infolist():
                data = updated.encode("utf-8") if item.filename == "word/document.xml" else source.read(item.filename)
                target.writestr(item, data)
    temporary.replace(DOCUMENT)
    with ZipFile(BACKUP) as before, ZipFile(DOCUMENT) as after:
        assert before.namelist() == after.namelist()
        assert [name for name in before.namelist() if before.read(name) != after.read(name)] == ["word/document.xml"]
    print("Changed body blocks:", changed)
    print("Only section 1.2.2 prose changed; abstracts, citations, native math and all formatting preserved.")
    for index in changed:
        print(index, "".join(n.text or "" for n in new_body[index].iter() if n.tag in (W + "t", M + "t")))
    print("OUTPUT", DOCUMENT)


if __name__ == "__main__":
    main()
