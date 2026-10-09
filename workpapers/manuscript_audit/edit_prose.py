"""One-off, source-preserving editorial changes for the 2026-09-29 review."""

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2] / "manuscript"
paths = sorted((ROOT / "chapters").glob("*.md"))
texts = {p.name[:2]: p.read_text() for p in paths}


def change(chapter, old, new):
    assert old in texts[chapter], (chapter, old)
    texts[chapter] = texts[chapter].replace(old, new)


def first(old, new):
    for chapter in texts:
        if old in texts[chapter]:
            texts[chapter] = texts[chapter].replace(old, new, 1)
            return
    raise ValueError(old)


change("01", "异步信用分配和图结构表示等方式", "异步信用分配等方式")
change("01", "分布式价值表示", "回报分布表示")
change("01", "模型生成的数据支持之外对价值进行保守正则", "模型生成的分布外状态动作上对价值进行保守正则")
first("SUMO提供了", "城市交通微观仿真平台（Simulation of Urban MObility，SUMO）提供了")
first("马尔可夫决策过程则", "马尔可夫决策过程（Markov Decision Process，MDP）则")
first("隐参数MDP", "隐参数马尔可夫决策过程（Hidden Parameter Markov Decision Process，HiP-MDP）")
first("UP-OSI将", "通用策略与在线系统辨识（Universal Policy with Online System Identification，UP-OSI）将")
first("PEARL通过", "概率嵌入式演员评论家方法（Probabilistic Embeddings for Actor-Critic Reinforcement Learning，PEARL）通过")
first("VariBAD将", "变分贝叶斯自适应深度强化学习（Variational Bayes-Adaptive Deep Reinforcement Learning，VariBAD）将")
first("RMA则", "快速运动适应方法（Rapid Motor Adaptation，RMA）则")
first("早期BAPR与其因果符号扩展CS-BAPR", "早期贝叶斯自适应策略正则化（Bayesian Adaptive Policy Regularization，BAPR）与其因果符号扩展（Causal-Symbolic Bayesian Adaptive Policy Regularization，CS-BAPR）")
change("01", "CS-BAPR的历史全称为Causal-Symbolic Bayesian Adaptive Policy Regularization，即因果符号贝叶斯自适应策略正则化；公交实现", "CS-BAPR的公交实现")
first("SINDy以", "非线性动力系统稀疏辨识（Sparse Identification of Nonlinear Dynamics，SINDy）以")
first("IRM研究", "不变风险最小化（Invariant Risk Minimization，IRM）研究")
first("均值与条件风险价值的联合优化", "均值与条件风险价值（Conditional Value-at-Risk，CVaR）的联合优化")
for old, new in {
    "BCQ约束": "批量约束Q学习（Batch-Constrained Q-Learning，BCQ）约束",
    "BEAR从": "自举误差累积削减（Bootstrapping Error Accumulation Reduction，BEAR）从",
    "BRAC统一": "行为正则化演员评论家方法（Behavior Regularized Actor Critic，BRAC）统一",
    "AWR通过": "优势加权回归（Advantage-Weighted Regression，AWR）通过",
    "CRR依据": "评论家正则化回归（Critic Regularized Regression，CRR）依据",
    "TD3+BC在": "结合行为克隆（Behavior Cloning，BC）的TD3方法（TD3+BC）在",
    "D4RL将": "深度数据驱动强化学习数据集（Datasets for Deep Data-Driven Reinforcement Learning，D4RL）将",
    "RL Unplugged提供": "强化学习固定数据基准（Reinforcement Learning Unplugged，RL Unplugged）提供",
    "RLPD研究": "利用先验数据的强化学习（Reinforcement Learning with Prior Data，RLPD）研究",
    "WSRL则": "预热强化学习（Warm-Start Reinforcement Learning，WSRL）则",
    "AWAC结合": "优势加权演员评论家方法（Advantage-Weighted Actor-Critic，AWAC）结合",
    "Cal-QL关注": "校准Q学习（Calibrated Q-Learning，Cal-QL）关注",
    "PEX则": "策略扩展（Policy Expansion，PEX）则",
    "客流OD": "客流起讫点（Origin-Destination，OD）",
    "PETS结合": "概率集成轨迹采样方法（Probabilistic Ensembles with Trajectory Sampling，PETS）结合",
    "PlaNet在": "深度规划网络（Deep Planning Network，PlaNet）在",
    "TD-MPC及TD-MPC2": "时序差分模型预测控制（Temporal Difference Model Predictive Control，TD-MPC）及其第二版TD-MPC2",
    "MVE通过": "模型价值扩展（Model-Based Value Expansion，MVE）通过",
    "STEVE根据": "随机集成价值扩展（Stochastic Ensemble Value Expansion，STEVE）根据",
    "MOPO以": "基于模型的离线策略优化（Model-Based Offline Policy Optimization，MOPO）以",
    "MOReL构造": "基于模型的离线强化学习（Model-Based Offline Reinforcement Learning，MOReL）构造",
    "COMBO则": "保守离线模型型策略优化（Conservative Offline Model-Based Policy Optimization，COMBO）则",
    "公交GPS与": "公交全球定位系统（Global Positioning System，GPS）记录与",
    "CARL通过": "情境化强化学习基准（Contextualize and Assess Reinforcement Learning，CARL）通过",
    "初始 Embedding-SAC": "初始类别嵌入软演员评论家方法（Embedding Soft Actor-Critic，Embedding-SAC）",
    "训练期TD标签": "训练期时序差分（Temporal Difference，TD）标签",
    "HIRO则": "异策略校正层级强化学习（Hierarchical Reinforcement Learning with Off-Policy Correction，HIRO）则",
    "CPU小时": "中央处理器（Central Processing Unit，CPU）小时",
    "训练AUC均值": "训练的接收者操作特征曲线下面积（Area Under the Receiver Operating Characteristic Curve，AUC）均值",
    "200个SiLU单元": "200个Sigmoid线性单元（Sigmoid Linear Unit，SiLU）",
    "两层ReLU函数": "两层整流线性单元（Rectified Linear Unit，ReLU）函数",
    "AI Feynman利用": "人工智能（Artificial Intelligence，AI）符号回归方法AI Feynman利用",
    "SINDy-RL则": "结合SINDy的强化学习方法（SINDy Reinforcement Learning，SINDy-RL）则",
    "RCPO以": "奖励约束策略优化（Reward Constrained Policy Optimization，RCPO）以",
    "PID拉格朗日方法": "比例积分微分（Proportional-Integral-Derivative，PID）拉格朗日方法",
    "源模型状态RMSE": "源模型状态均方根误差（Root Mean Squared Error，RMSE）",
    "其ETA观测": "其预计到达时间（Estimated Time of Arrival，ETA）观测",
}.items():
    first(old, new)

# These names have already been introduced in the literature review.
change("06", "本章提出基于模型修正的世界模型方法（Model-Corrected World Model，MC-WM）。", "本章提出MC-WM。")
change("05", "混合离线与在线强化学习（Hybrid Offline-and-Online Reinforcement Learning，H2O）及其扩展H2O+由Niu等提出", "H2O及其扩展H2O+由Niu等提出")

change("02", "本文按原研究分别报告固定数据、训练交互、验证访问及最终评价。", "表2-4分别列出固定数据、训练交互与评价对象；验证访问和最终评价在各章实验协议中说明。\n\n表2-4 各类研究的数据来源与环境访问条件")

introductions = {
    "图2-7": "图2-7进一步按需求水平分解七类停站干预的响应差，比较平均结果相近时局部服务后果是否仍然一致。",
    "图3-11": "图3-11汇总两类分歧项的平均作用，显示其影响随运行条件和更新对象而变化。",
    "表3-7": "表3-7分别列出行动者与评论家分歧项开关组合的控制回报和运营指标，用于区分两项的作用。",
    "表4-5": "表4-5给出反复变化需求下的回报中位数，用于比较连续适应过程中的控制效果。",
    "表4-10": "表4-10把计划班次、实际车辆投入和累计停站负担放在同一统计范围内，补充峰值车辆指标的解释。",
    "图4-14": "图4-14比较早期BAPR与RE-SAC在共同评价种子下的奖励差和乘客时间差，显示运行目标改善与乘客时间收益之间的区别。",
    "图4-17": "图4-17比较等累计强度下单次与重复冲击的回报差，使冲击频率与总需求负荷的作用得到区分。",
    "表4-14": "表4-14列出输出结构与行为克隆的交叉结果，检验完整配置的收益能否进一步归因于单个组件。",
    "表6-1": "表6-1给出各任务改变的动力学因素及违例定义，明确回报与违例统计对应的环境条件。",
    "表6-3": "表6-3单独列出重力变化任务的回报与违例结果，用于比较源知识保留和目标建模的作用。",
    "图6-9": "图6-9比较共同离线初始化之后的续训方式，同时展示乘客时间与完成率的变化。",
    "图6-12": "图6-12在相同执行规则下比较不同数据来源与运营规则，区分学习方式和服务保障的影响。",
    "表6-9": "表6-9同时报告多训练种子的服务结果与训练成本，使控制改善及其计算代价能够共同评价。",
}
for label, paragraph in introductions.items():
    chapter = "0" + label[1]
    text = texts[chapter]
    caption = re.search(r"(?m)^" + re.escape(label) + r" .+$", text)
    assert caption, label
    start = caption.start()
    if label[0] == "图":
        start = text.rfind("![]", 0, start)
        assert start >= 0, label
    texts[chapter] = text[:start] + paragraph + "\n\n" + text[start:]

# Link results to the precise mathematical object established in the appendix.
change("03", "附录A给出相关性质及惩罚随输入改变时的反例。", "固定惩罚下的单调性与收缩性由定理A.1给出，熵正则化形式对应引理A.2和定理A.2，精确迭代的固定点结论见推论A.1。命题A.1进一步构造输入依赖惩罚的反例，说明上述证明不能直接替代联合网络训练的收敛分析。")
change("04", "固定信念条件下的贝尔曼性质见附录C。", "命题C.1将固定信念下的模式混合写成合法转移核，定理C.1据此给出价值更新的收缩性；定理C.2则说明信念若随被比较的价值函数变化，反馈可能破坏收缩。这一区别限定了本节使用贝叶斯权重的理论解释。")
change("04", "其路径等价和模式误判分析见附录B。", "附录B中，定理B.1先给出有界命令的精确补偿条件，定理B.2再证明极性变换保持动作边界；随后通过逐路径比较和性能差恒等式分析模式误判。第4.7.3节用模式可识别与难以识别的公交条件检验这一机制。")
change("04", "附录D给出相应证明，并说明导数跳跃对这一条件的影响。", "附录D的命题D.10给出标量复合导数界，第D.5.3节将其扩展至向量映射；命题D.7和命题D.8说明导数跳跃为何不满足该全局条件。因此，输出头的平滑性不能单独保证完整策略的响应界。")
change("06", "附录E给出完整定义及推论。", "引理E.2给出总变差与期望差的联系，定理E.2据此证明相同奖励下的价值误差界，第E.3.3节加入奖励误差得到本节的完整形式。附录E第E.4节的数据覆盖反例说明，为何局部预测准确不能推出全局价值准确。")
change("06", "### 6.4.3 预测误差、置信排序与控制收益\n", "### 6.4.3 预测误差、置信排序与控制收益\n\n定理E.4说明残差拟合误差一致时，置信信号如何与源目标差异形成条件排序关系；定理E.5进一步给出拟合误差波动下保持排序所需的间隔。这些结论解释置信加权的适用条件，实际控制收益则由第6.6节模型类别干预和第6.7节公交比较判断。\n")
change("04", "### 4.3.3 分布外响应的条件误差界\n", "### 4.3.3 分布外响应的条件误差界\n\n本节应用推论D.6的完整映射误差界，将基准点误差、导数误差和输入偏移的作用分别展开。\n")

# Cite borrowed methods at their use, not solely in the introduction.
change("05", "### 5.4.4 离线初始化与源仿真续训\n", "### 5.4.4 离线初始化与源仿真续训\n\n离线阶段采用IQL的期望分位回归与优势加权策略提取，续训阶段采用SAC式熵正则化更新，并按公交H2O+的设置组合两类目标。[@kostrikov2022iql; @haarnoja2018soft; @niu2025h2oplus]\n")
change("05", "### 5.4.5 转移比率诊断\n", "### 5.4.5 转移比率诊断\n\n联合与边缘分类器之差用于消去状态动作访问分布比，从而诊断条件转移差异；这一思路来自DARC及H2O的跨动力学修正。[@eysenbach2021off; @niu2022h2o]\n")
change("03", "### 3.3.5 算子稳定性\n", "### 3.3.5 算子稳定性\n\n本节在正则化MDP的框架下分析固定惩罚的更新映射。[@Geist2019TheoryRegularized] 这里研究的是给定转移核上的价值算子，具体惩罚项则来自本章RE-SAC定义。\n")

# Referring back to an equation makes its later use explicit.
equation_links = {
    "02": {"eq2_001": "[式](#eq2_001)定义的两类时距分别对应前后服务间隔。", "eq2_003": "[式](#eq2_003)把间隔传播分为可控停站差与外生运行差。", "eq2_008": "[式](#eq2_008)明确区分归一化策略输出与以秒计的执行动作。", "eq2_014": "[式](#eq2_014)建立了事件成本与个体时间累计之间的等价关系。", "eq2_015": "后续公交跨保真比较统一采用[式](#eq2_015)中的广义时间与完成率。", "eq2_018": "[式](#eq2_018)保留事件持续时间和终止信息，使固定日志能够用于相同的事件决策任务。"},
    "03": {"eq3_007": "[式](#eq3_007)是本章计划间隔奖励的比较基准。", "eq3_011": "[式](#eq3_011)只给出局部时距均衡动作，不包含乘客车内代价。", "eq3_019": "[式](#eq3_019)进一步说明车内乘客增多时最优局部停站缩短的原因。", "eq3_021": "[式](#eq3_021)给出价值均值，下式刻画同一组评价器的分歧。", "eq3_024": "[式](#eq3_024)中的惩罚改变目标值，其作用不同于下一节的策略悲观评价。", "eq3_027": "[式](#eq3_027)把熵项、平均价值与分歧项共同纳入动作选择。", "eq3_029": "[式](#eq3_029)所需的固定条件将在附录A中逐项给出。", "eq3_031": "[式](#eq3_031)区分初值误差的衰减与逐次逼近误差的累积。"},
    "04": {"eq4_002": "[式](#eq4_002)只更新模式信念，控制响应还需由策略和执行映射确定。", "eq4_008": "[式](#eq4_008)的右端同时包含特征网络与输出头的常数。", "eq4_011": "[式](#eq4_011)对应推论D.6，三个误差项分别随偏移呈常数、一次和二次变化。", "eq4_018": "[式](#eq4_018)将计划调整限制在时间窗口与最早可执行时刻的共同范围内。", "eq4_022": "[式](#eq4_022)更新的是约束乘子，而不是客流到达率。", "eq4_029": "[式](#eq4_029)表明实际间隔偏差取决于相邻班次延误之差。", "eq4_036": "[式](#eq4_036)对应附录B的动作补偿条件，模式误判时不再保证执行等价。"},
    "05": {"eq5_001": "[式](#eq5_001)区分后续各比较允许使用的训练信息。", "eq5_009": "[式](#eq5_009)把精确零停站作为概率质量点，与正停站的连续密度共同组成策略。", "eq5_014": "[式](#eq5_014)给出本实验的混合目标，系数在比较中保持固定。", "eq5_015": "[式](#eq5_015)是否具有实际修正价值，取决于留出轨迹上的分类与权重诊断。", "eq5_020": "[式](#eq5_020)给出完整控制器的执行动作，后续实验分别比较学习策略与这一执行变换。", "eq5_025": "[式](#eq5_025)用于将广义时间收益拆分为候车改善和车内代价。"},
    "06": {"eq6_004": "[式](#eq6_004)用于比较模型预测，并非控制回报的替代指标。", "eq6_008": "[式](#eq6_008)中的线性项与神经余项共享已选特征支持。", "eq6_012": "[式](#eq6_012)将局部预测可信程度沿后续决策传播。", "eq6_016": "[式](#eq6_016)使模型权重直接作用于价值拟合损失。", "eq6_019": "[式](#eq6_019)是定理E.1在当前残差核上的应用。", "eq6_020": "[式](#eq6_020)分别给出奖励误差与转移误差对价值的影响。", "eq6_027": "[式](#eq6_027)在公交混合更新之外加入模型项，原有混合数据项保持不变。"},
}
for chapter, additions in equation_links.items():
    for label, sentence in additions.items():
        pattern = r"(<!-- eq:" + label + r" -->\n\n\$\$\n.*?\n\$\$)"
        texts[chapter], count = re.subn(pattern, lambda m: m[0]+"\n\n"+sentence, texts[chapter], flags=re.S)
        assert count == 1, label

for path in paths:
    path.write_text(texts[path.name[:2]])
