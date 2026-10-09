# 研究计划与学术简历草稿

> 说明：本文档基于当前提供的 12 篇论文 PDF 整理。由于尚未确认每篇论文的投稿、在投或录用状态，以下统一表述为“代表性论文/工作论文”，正式使用时可按真实状态改为“已发表”“在投”“预印本”或“准备投稿”。

## 一、个人研究定位

我的研究方向是面向真实交通系统与复杂决策环境的可信强化学习，重点关注非平稳性、随机扰动、分布外泛化、离线到在线迁移、多目标权衡以及形式化可验证的算法稳定性。围绕公交运行控制这一高不确定、高约束、强时空耦合的典型场景，我从单车/车队控制、离线强化学习、非平稳鲁棒控制、跨城市迁移、多目标决策、实时扰动路径选择等多个层面展开研究，并逐步形成了一条较完整的研究主线：

**在开放、动态、数据受限的真实系统中，如何构建既能学习、又能解释、可迁移、可验证的智能决策方法。**

具体而言，我的研究尝试回答四类问题：

1. **不确定性如何被分解和控制。** 在公交控制中，交通速度、乘客到达和信号扰动带来不可消除的 aleatoric uncertainty；而稀疏样本、长尾状态和仿真-现实差距带来 epistemic uncertainty。我的工作将这两类风险显式分解，并通过鲁棒正则、Q-ensemble、LCB pessimism 和 Bayesian change detection 等机制分别处理。

2. **非平稳系统中策略如何稳定适应。** 城市交通环境常呈现 piecewise-stationary 特征：稳定一段时间后发生突发拥堵、客流激增或服务中断。我的 BAPR 系列工作将 Bayesian Online Change Detection 与鲁棒 ensemble RL 结合，用 belief-weighted Bellman operator 描述模式切换下的自适应保守性，并用 Lean 4 对关键收缩性结论进行形式化验证。

3. **策略如何跨城市、跨强度、跨模型迁移。** 针对公交控制中的 sim-to-real、cross-city transfer 和 OOD demand scaling，我提出机制分解的 residual world model、Neural Arithmetic Unit/KAN actor head、causal-factored transfer adapter 等方法，探索从“单一城市仿真最优”走向“跨城市可复用决策系统”的路径。

4. **智能系统如何自我验证。** 除交通 RL 外，我还研究 self-hypothesizing/self-validating agents 和 hypothesis coherence metric，关注智能体提出假设、验证假设、修正知识库的能力。这一方向与交通决策中的模型选择、机制发现、仿真校正和自动研究流程形成方法论呼应。

## 二、代表性论文与研究成果

### 1. 公交控制基础建模与单智能体强化学习

**Single Agent Robust Deep Reinforcement Learning for Bus Fleet Control**

该工作将传统上常用多智能体建模的公交车队控制问题转化为单智能体强化学习问题。通过显式编码 vehicle ID、station ID、time period 等类别变量，并与 headway、occupancy、segment velocity 等连续状态共同构成高维状态表示，模型能够在双向、时刻表驱动、车队规模随时段变化的更真实公交场景下学习统一策略。该工作建立了后续 RE-SAC、BAPR、offline SUMO benchmark 等研究的基础实验场景和问题设定。

### 2. 鲁棒 ensemble SAC 与不确定性分解

**RE-SAC: Disentangling Aleatoric and Epistemic Risks in Bus Fleet Control**

该工作提出 Robust-Ensemble Soft Actor-Critic，将公交控制中的 aleatoric risk 与 epistemic risk 分开处理。方法上，使用 IPM-based weight regularization 抑制高随机环境中的 Q-value oscillation，用 diversified Q-ensemble 和 lower-confidence-bound 机制处理长尾状态中的数据不足。理论上，工作指出冻结参数的 Bellman backup 是保证 γ-contraction 的关键结构条件，并给出 Q-dependent penalty 破坏收缩性的反例，相关核心命题通过 Lean 4 形式化验证。

### 3. 非平稳环境中的 Bayesian amnesic robust RL

**BAPR: Bayesian Amnesic Piecewise-Robust Reinforcement Learning for Non-stationary Continuous Control**

BAPR 将 Bayesian Online Change Detection 引入鲁棒 ensemble SAC，用 run-length posterior 和 mode belief 调节策略的保守程度，使模型在稳定阶段不过度保守，在突发模式切换后提升风险规避能力。该工作提出 frozen-belief mode-mixture Bellman operator，并证明其在固定 belief 和 penalty 时保持 γ-contraction；同时构造 Q-dependent belief 破坏 contraction 的形式化反例。该研究是后续 BAMOR、CS-BAPR 等工作的理论核心。

### 4. 分布外需求强度泛化与架构边界

**CS-BAPR: Formal OOD Extrapolation Bounds for Reinforcement Learning via Neural Arithmetic Units**

该工作研究公交控制策略在训练需求强度之外的 magnitude OOD 泛化问题。文章指出 ReLU actor head 的导数不连续性会限制可证明外推稳定性，并引入 NAU/NMU、KAN 和 smooth MLP-with-tanh 等 actor head 作为可选架构。核心理论结果是 OOD Extrapolation Stability Bound：若训练域边界误差、导数误差和导数 Lipschitz 常数受控，则 OOD 策略误差以距离的二次多项式形式增长。该工作将训练稳定化技巧与结构化 actor head 结合，为公交 RL 的极端需求场景提供了可解释的架构选择依据。

### 5. 大规模 SUMO 离线强化学习基准

**Uncertainty-regularized Offline Reinforcement Learning Matches Offline-to-online Fine-tuning for Bus Holding Control: A SUMO Benchmark**

该工作构建了 Changsha 12 条公交线路、389 辆公交、353 个站点、310 万离线 transitions 的 SUMO 微观仿真基准，用于系统比较 offline RL 与 offline-to-online RL。研究发现，经过不确定性正则化的纯离线 RE-SAC offline 能够与 architecture-matched WSRL-E10 等 offline-to-online 方法竞争，并在最终 checkpoint 稳定性上更强。该工作的重要贡献不只是算法，而是提出了更严格的 post-hoc checkpoint evaluation protocol，揭示常见训练中 best metric 可能高估弱 baseline 的真实性能。

### 6. 双层公交目标 headway 规划与实时 holding 控制

**TransitDuet: Bi-level Reinforcement Learning for Joint Bus Target-headway Planning and Holding Control**

TransitDuet 将上层 target-headway planning 与下层 real-time holding control 建模为异步双层强化学习问题。上层在 dispatch event 输出目标 headway shift，下层在每个站点到达事件执行 holding action，两层具有不同时间尺度、不同状态空间和延迟反馈。该工作采用 goal-conditioned cross-level coupling、θ-OGD 自适应 fleet overshoot penalty 和下层 Lagrangian relaxation，实现规划与控制的联合优化。该方向将我的研究从“固定时刻表下的控制”推进到“计划-控制一体化”。

### 7. 跨城市机制分解模型迁移

**Causal-Factored Cross-City Model Transfer for Bus Holding Control with Open Transit Data**

CFCMT 面向公交 holding policy 的跨城市迁移问题，提出机制分解的 residual world-model adapter，将模拟器误差分解为 headway、demand/dwell、speed 和 reward 等可解释机制，并使用未标注目标城市统计量进行 source selection。相比 dense H2O+-style residual baseline，该方法在 Singapore、Austin/CapMetro、Halifax Transit 和 MBTA 四个开放公交城市数据包上表现更稳定。该工作强调边界条件：机制分解迁移对低标定预算下的 simulator residual adaptation 有用，但正式运营部署仍需要 vehicle-level validation 或 field validation。

### 8. 多目标非平稳鲁棒强化学习

**BAMOR: Bayesian Amnesic Multi-Objective Robust Reinforcement Learning with Target Cone Tracking under Substrate Constraints**

BAMOR 将 BAPR 从单目标扩展到多目标强化学习，并提出 Target Cone 机制处理不同交通模式下可达目标权衡空间收缩的问题。传统 MORL 假设偏好权重可在完整 simplex 上自由变化，但在严重拥堵等物理约束下，速度等目标可能事实上不可达。BAMOR 用 mode-specific Target Cone 限制偏好采样，并给出局部化 tracking bound，说明在 collapsed objectives 被抑制时，局部 drift 显著小于 global drift。理论链条通过 Lean 4 进行了形式化验证。

### 9. 扰动下的随机公交 hyperpath 重排序

**Re-Ranking Stochastic Transit Hyperpaths Under Disruption: A Wasserstein-Robust Scoring Approach**

该工作研究公交/轨道实时扰动下的 passenger routing。核心观点是：扰动发生时，预计算 stochastic hyperpath 往往仍包含有用备选路径，真正脆弱的是候选路径的排序。因此，方法不是重算整个 hyperpath，而是保留候选结构并基于实时 posterior delay、cancellation belief、feasibility 和 on-time probability 进行在线 re-ranking。四项 posterior pessimism score 在 affine arrival cost 下等价于 Wasserstein-1 ambiguity set 上的 worst-case expected arrival，该数学核心通过 Lean 4 验证。该工作把鲁棒决策从车队控制扩展到乘客路径选择。

### 10. LLM 辅助 residual world model 与 sim-to-real RL

**Model-Corrected World Models: LLM-Guided Residual Dynamics and Safety-Aware Rollouts for Sim-to-Real Reinforcement Learning**

MC-WM 研究物理仿真参数与真实环境不一致时的 model-based RL sim-to-real transfer。方法由 LLM-guided symbolic residual、Bellman QΔ rollout trust critic 和 Pareto-gated safety orchestrator 组成，用残差世界模型修正 simulator dynamics，并在 imagined rollout 中抑制不可信区域。该工作强调 residual model 的 policy-conditional 性质，证明 policy-universal convergence 不成立，并给出 residual simulation lemma、QΔ 与 ICRL discriminator 的 rank equivalence 等理论结果。该方向为后续交通仿真校正与自动机制发现提供通用方法。

### 11. 自假设、自验证研究代理

**Toward Self-Hypothesizing, Self-Validating Agents**

该工作从 AI agent 方法论角度讨论一个能够提出假设、运行验证、发现失败、提出子假设并递归修正的 self-validating architecture。论文首先审计一个最简 same-family A/B gate 自改进循环，发现表面通过的候选在 cross-family re-judgment、sample extension 和 selection-bias modeling 后无法稳健成立。贡献不在于宣称 self-improvement 已经成功，而是给出一套更完整的架构路线图：methodological priors、cheap world model、scheduler、experience feedback、formal alignment layer 和 new-prior generator。该方向与我的交通 RL 研究形成互补：前者关注研究流程的可信自动化，后者关注真实系统中的可信决策。

### 12. 假设一致性与最小嵌入距离

**Minimum Embedding Distance as a Hypothesis Coherence Metric**

该工作提出一种用于假设排序的 composite score：由 posterior KL drift 和 structural cost 两部分构成，用于衡量候选假设与现有 belief network 的一致性。它针对 pure KL 容易错误偏好 orphan-node hypotheses 的问题，引入 BIC-style structural penalty，使与目标变量无因果路径的“孤立解释”被正确降权。该工作虽不宣称信息论形式本身全新，但提供了一个可接入 PyMC/probabilistic program 的 prescriptive formulation，可用于智能体假设评估、模型选择和解释一致性判断。

## 三、总体研究贡献

从这些工作可以归纳出四方面核心贡献。

第一，**构建了面向真实公交控制的强化学习问题族**。从单走廊到 12 线路 SUMO 网络，从单目标 holding 到 target-headway planning + holding，从车队控制到乘客路径重排序，研究对象逐渐接近真实公共交通系统的多层决策结构。

第二，**提出了不确定性分解驱动的鲁棒 RL 方法体系**。RE-SAC 分解 aleatoric/epistemic risk；BAPR 处理 piecewise stationarity；CS-BAPR 处理 magnitude OOD extrapolation；BAMOR 处理多目标可达空间收缩；offline SUMO work 处理离线日志中的分布偏差。这些工作共同指向一个统一问题：在分布不断变化且数据不完备的系统中，如何让 RL 策略稳定学习。

第三，**将形式化验证引入深度强化学习理论核心**。多篇工作使用 Lean 4 验证 contraction、counterexample、tracking bound、Wasserstein DRO identity 等数学结果。形式化验证的重点不是验证所有代码，而是验证算法稳定性声明中最容易被非形式化推导掩盖的理论核心。

第四，**探索了 LLM/agent 与科学建模的结合方式**。MC-WM 使用 LLM 提出 symbolic residual features；self-validating agent 研究假设生成和验证循环；MEIS 研究假设与既有 belief network 的一致性。它们共同服务于一个长期目标：让智能系统不只是执行策略，还能发现模型误差、提出解释、验证假设并更新知识。

## 四、未来研究计划

### 研究总目标

未来 2-3 年，我计划围绕“面向城市交通系统的可验证、可迁移、可自校正强化学习”展开系统研究。目标是将已有算法与基准整合为一个从离线日志、仿真校正、鲁棒策略学习、实时扰动响应到形式化理论验证的完整研究框架。

### 方向一：统一的可信公交强化学习基准与算法栈

已有工作中，单智能体 bus control、RE-SAC、BAPR、offline SUMO 和 TransitDuet 分别解决了建模、鲁棒性、非平稳性、离线学习和双层控制问题。下一步需要将这些模块整合为统一 benchmark 与算法栈：

- 建立标准化数据接口，统一 SUMO logs、GTFS/open transit data、offline transitions、real-time feed 和 simulation residuals。
- 将 RE-SAC offline、BAPR、CS-BAPR 和 TransitDuet 纳入同一评估协议，比较它们在稳定日、拥堵日、客流突增、线路扰动和需求 OOD 下的表现。
- 发展统一的 checkpoint evaluation 与 uncertainty reporting 标准，避免只报告训练过程中的最优 checkpoint。

预期成果是一个可复现实验平台和一篇 benchmark/system paper，为公共交通 RL 提供比单一 toy corridor 更接近真实运营环境的评价基准。

### 方向二：从鲁棒控制到跨城市迁移

真实部署中，单个城市的高质量标定成本很高，因此跨城市迁移是关键问题。基于 CFCMT 和 MC-WM，我计划进一步研究：

- 如何用 mechanism-factored residual world model 表示不同城市之间的局部机制差异。
- 如何用少量目标城市观测或完全未标注统计量选择可信源城市。
- 如何将 LLM-guided symbolic residual discovery 与交通机制约束结合，减少纯神经 residual 的不可解释外推。
- 如何定义“何时不该迁移”的诊断指标，避免在 observation-level 或 actuator-level mismatch 下错误使用 residual correction。

这一方向的目标是形成一套低标定预算下的 cross-city transit control adaptation 方法，并明确其适用边界。

### 方向三：非平稳、多目标与计划-控制协同

BAPR、BAMOR 和 TransitDuet 指向同一个更高层问题：交通系统不是固定单目标控制任务，而是多目标、非平稳、跨时间尺度的运营系统。未来我计划：

- 将 BAPR 的 regime belief 与 BAMOR 的 Target Cone 结合，使策略在不同运行模式下自动调整目标权重空间。
- 将 TransitDuet 的上层 target-headway planning 与 BAMOR 的多目标 preference cone 连接，让上层根据拥堵、客流、准点率和车队约束动态选择目标。
- 研究 headway regularity、waiting time、energy/fuel、on-time arrival、fleet overshoot 等目标之间的可达边界，而不是假设完整 Pareto simplex 始终可用。

预期贡献是一套 mode-aware multi-objective transit RL 框架，能够在不同运行模式下给出可解释的目标权衡。

### 方向四：扰动响应与乘客路径层面的鲁棒决策

现有公交控制研究多关注运营者侧的 holding 和 dispatch，而乘客侧 routing 在扰动中同样关键。基于 hyperpath re-ranking 工作，我计划：

- 将实时 vehicle delay/cancellation posterior 与 operator-side control state 连接，使 route recommendation 能感知控制策略变化。
- 研究“控制-路径推荐”闭环：holding 策略影响车辆到达，路径推荐影响乘客选择和站点需求，二者共同改变系统状态。
- 在 SBB/GTFS-RT 或国内城市公交实时数据上验证 keep-and-rerank 相比 recompute hyperpath 的适用条件。

这一方向可以把研究从“控制车辆”扩展到“同时服务运营者和乘客的实时决策系统”。

### 方向五：可自验证的交通研究代理

长期来看，我希望把 self-validating agent 方向与交通 RL 结合，构建面向交通仿真、策略评估和模型诊断的自动研究助手。该系统应能：

- 自动提出 simulator residual hypotheses，例如某条线路的 dwell model、speed model 或 demand model 失配。
- 用结构化实验验证假设，而不是只依赖同源 LLM judge。
- 用 hypothesis coherence metric 过滤与既有交通机制不一致的解释。
- 生成可复现的 ablation、selection-bias audit 和 cross-seed replication protocol。

这一路线的目标不是让 LLM 直接替代研究者，而是让它成为可审计的 hypothesis generator 和 experiment scheduler。

## 五、简历式研究简介

**研究方向：** 强化学习、鲁棒决策、公共交通智能控制、离线/离线到在线 RL、非平稳 RL、多目标 RL、sim-to-real transfer、形式化验证、LLM-assisted scientific agents。

**研究关键词：** bus holding control, robust ensemble SAC, uncertainty decomposition, Bayesian online change detection, offline reinforcement learning, SUMO simulation, cross-city transfer, residual world model, multi-objective RL, Wasserstein DRO, Lean 4 formal verification, self-validating agents。

**代表性能力：**

- 强化学习算法：SAC、offline actor-critic、ensemble Q-learning、LCB pessimism、bi-level RL、model-based RL、offline-to-online RL。
- 交通仿真与数据：SUMO 微观仿真、公交 holding control、GTFS/open transit data、real-time delay/cancellation feed、hyperpath routing。
- 鲁棒与迁移方法：BOCD、robust Bellman operator、residual world model、causal-factored transfer、OOD extrapolation bound、Target Cone MORL。
- 理论与验证：contraction proof、counterexample construction、tracking bound、Wasserstein DRO identity、Lean 4 formalization。
- 智能体与科学推理：LLM-guided symbolic residual discovery、self-hypothesizing agent architecture、hypothesis coherence metric、probabilistic programming/PyMC。

**一句话简介：**

我主要研究真实交通系统中的可信强化学习，关注公交调度、非平稳鲁棒控制、离线到在线迁移、跨城市泛化和形式化可验证算法设计，目标是在动态、数据受限且存在分布变化的城市交通环境中构建可学习、可解释、可迁移的智能决策系统。

## 六、可用于申请材料的短版研究计划

我的研究计划聚焦于“面向城市交通系统的可验证、可迁移强化学习”。现实公交系统同时具有随机扰动、非平稳模式切换、离线日志偏差、跨城市分布差异和多目标运营约束，传统强化学习方法往往在单一仿真场景中表现良好，却难以稳定迁移到更真实的运行环境。为解决这一问题，我计划从三个层面推进研究。

第一，构建统一的公交强化学习基准，将 SUMO 微观仿真、离线运行日志、开放 GTFS 数据和实时扰动信息纳入同一评估框架，系统比较 online RL、offline RL、offline-to-online RL、robust RL 和 bi-level RL 在不同扰动模式下的稳定性。

第二，发展机制分解的鲁棒强化学习方法。通过 ensemble uncertainty、Bayesian change detection、Target Cone multi-objective sampling 和 causal-factored residual world model，将随机噪声、数据不足、模式切换和跨城市机制差异分别建模，避免用单一不确定性指标处理所有风险。

第三，引入形式化验证与自验证智能体。对于算法稳定性核心，如 Bellman contraction、OOD extrapolation bound、Wasserstein robust scoring 和 tracking bound，我将继续使用 Lean 4 验证关键数学命题；同时探索 LLM-assisted hypothesis generation 和 audit protocol，使模型误差诊断、ablation 设计和实验复现部分自动化。

通过上述研究，我希望形成一套从离线数据学习、仿真-现实校正、鲁棒策略训练、实时扰动响应到理论验证的完整方法体系，为城市公共交通中的智能调度和可信决策提供可复现、可解释、可迁移的技术基础。

## 七、可用于简历的精简版个人陈述

我目前的研究围绕可信强化学习与公共交通智能控制展开，重点解决真实公交系统中的非平稳性、随机扰动、离线数据偏差、跨城市迁移和分布外泛化问题。我的代表性工作包括：提出 RE-SAC 以分解公交控制中的 aleatoric 与 epistemic uncertainty；提出 BAPR 将 Bayesian Online Change Detection 与鲁棒 ensemble RL 结合处理 piecewise-stationary 环境；构建 Changsha 12 线路 SUMO 离线强化学习基准，比较 pure offline 与 offline-to-online 方法；提出 TransitDuet 进行 target-headway planning 与 holding control 的异步双层优化；提出 CFCMT 和 MC-WM 探索跨城市/仿真到现实的 residual world-model adaptation；并使用 Lean 4 对多个核心理论结论进行形式化验证。长期目标是构建可学习、可解释、可迁移、可验证的智能交通决策系统。

