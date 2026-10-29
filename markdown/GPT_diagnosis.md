我按 **`5b8556e5` 固定提交**核对了汇总报告、派生指标和关键训练代码。我的判断是：

**应当保留已经有效的持续策略学习，把下一轮重点放在“局部机制校正如何变成更准确的动作价值”上。当前最值得优先处理的，是局部到全局的模型接口，以及 context 在预测、trust、Q 学习中的时间和条件一致性。**

这次还有一个比“模型信号比较弱”更具体的代码发现：**局部读出训练使用样本发出时保存的 context，而 trust 会用当前 context 预测累计旧样本，再与旧工况产生的结果比较。** 这个条件配对不一致可以直接定位；它是否解释了实际退化，需要用同批诊断验证。

我建议调整原定计划：**先暂缓那 8 个“去 waiting / 去 cost”的完整控制日，将删头放入影子更新实验；完整控制日优先补“冻结世界模型参数、照常训练策略”的对照。**

下面按“已成立的结论—失效机制—具体修法—最小实验”展开。

---

## 一、这次结果应该怎样解释

你的澄清应当成为后续实验分析的前提。

| 层次 | 当前证据 | 可以得出的判断 |
|---|---|---|
| 持续策略学习整体 | A1 相对冻结初始化行为，stable / shock 分别降低 **62.68 / 73.85 秒/乘客**，五个种子均改善 | 已有学习流程有实际收益，应作为可靠基线保留 |
| fresh outcome 的独立贡献 | 尚缺同样 456 次更新、不给 fresh 转移监督的对照 | 收益还不能全部归因于新标签，也包含继续优化的贡献 |
| 世界模型的额外贡献 | 共享 F 后，五种子 model−real075：shock 全日 **−4.203**，停更后窗 **+0.693 秒/乘客** | 有局部收益，但稳定的增量价值尚未建立 |
| 局部物理预测 | 三线 MAE 从约 **20–22 秒**降至 **7.5–9.8 秒** | 稀疏反馈确实能够追踪局部时长变化 |
| 当前局部读出的控制收益 | 原两种子 shock，local−model 全日均值 **+11.476 秒/乘客** | 当前读出在这两个开发种子上造成损失；不能扩大为五种子结论 |

这些数值来自这次的[汇总报告](https://github.com/erzhu419/Thesis/blob/5b8556e537bf19dd8d6671b041d9e9066d4147f1/experiments/mcwm_closed_loop_v4_review_20261020/问题与进展汇总_供GPT6分析.md)、[冻结行为对照](https://github.com/erzhu419/Thesis/blob/5b8556e537bf19dd8d6671b041d9e9066d4147f1/experiments/mcwm_closed_loop_v4_review_20261020/frozen_behavior_comparison.json)和[派生成本指标](https://github.com/erzhu419/Thesis/blob/5b8556e537bf19dd8d6671b041d9e9066d4147f1/experiments/mcwm_closed_loop_v4_review_20261020/review_metrics.json)。

因此，现在应追问：

> **局部预测已经更准了，为什么由它生成的训练目标没有让策略更准确地选择 holding？**

这比继续确认 fresh 是否进入 actor、模型是否更新，更接近当前尚未解决的问题。

---

## 二、第一优先级：局部预测改善，尚未转化成正确的动作后果

### 1. 物理区间和 global 下一事件之间，缺少一个经过验证的转换

你的两类标签定义都可以成立：

- **物理区间**：同一车辆从一个站的 decision-ready，到下一站的 decision-ready。
- **global 转移**：全系统从当前控制事件，到任意下一控制事件。

global 从 A 车转移到 B 车，本身没有问题。代码也确实分别维护全局 pending 和按车辆、班次匹配的物理 pending；当前同时决策 batch 中已执行的其他 holding，还通过 `prior_policy_action` 保留下来了。因此，不应把它诊断成“车辆配错了”或者“同批其他动作全丢了”。[事件流实现](https://github.com/erzhu419/Thesis/blob/5b8556e537bf19dd8d6671b041d9e9066d4147f1/experiments/mcwm_closed_loop_v1/stream.py#L170-L250)

真正的问题是：

**“某辆车到下一站晚了若干秒”，应该怎样改变全系统下一事件、之后的车辆次序、乘客上下车和累计成本？**

例如，一辆受限速车辆晚到，并不意味着紧接着的 global 事件也晚同样的时间——另一辆车可能先到站。受影响车辆的延迟要保留在系统状态里，经过若干其他车辆的事件后，才表现为服务变化和乘客成本。

当前 canonical state 主要描述**本次到达决策点的车辆**，没有显式保存全车队未到站车辆的剩余运行时间、剩余 holding、未来事件进度。[状态契约](https://github.com/erzhu419/Thesis/blob/5b8556e537bf19dd8d6671b041d9e9066d4147f1/experiments/mcwm_bus/runtime/H2Oplus/bus_h2o/mdp_contract_v7.py#L68-L104)

这构成了一个有依据的结构性风险：

> 局部机制确实变了，但现有观察和模型接口未必保留了把这个变化传播到未来决策所需的信息。

它仍是需要验证的解释，不能仅凭状态字段就断言已经找到全部根因。

### 2. 当前物理模型主要学到了区间时长的水平变化

`IntervalDiagnostic` 的拟合目标是：

\[
y=\text{完整同车区间时长}-\text{实际 holding}.
\]

输入包含线路、headway、queue、load、dwell，**不包含 action**；预测时再把 holding 加回去。因此，在固定当前特征下，其形式是：

\[
\widehat T(h)=h+\max\{0,x^\top\theta+c_\ell\}.
\]

也就是说，它预先规定了：

\[
\frac{\partial\widehat T}{\partial h}=1.
\]

这个模型可以很好地发现“该区间整体变慢了”，但没有从这些物理标签中学习 holding 如何进一步改变信号相位、后续排队或下一站服务。[物理区间模型](https://github.com/erzhu419/Thesis/blob/5b8556e537bf19dd8d6671b041d9e9066d4147f1/experiments/mcwm_closed_loop_diagnostic_v2_20261020/interval.py#L101-L168)

这个加性假设未必不合理。问题在于：**目前 MAE 的改善验证了水平校准能力，尚未验证动作响应。**

### 3. 三维偏移到 9×4 读出，承担了过多尚未识别的关系

实际路径是：

1. 在各线路的固定 source 平均输入上计算预测偏移；
2. 得到三个线路级 \(\delta_\ell\)；
3. 展开成系统偏移、当前线路偏移、线路偏移×holding；
4. 用 ridge 映射到四个标准化 global 输出。

其新增动作结构可以精确写出来。设 \(R(s,a,z)\) 是四头校正，\(z_\ell\) 是归一化线路偏移，则固定 \(s,z\) 时：

\[
R(s,a_1,z)-R(s,a_0,z)
=
\mathbf 1_{\ell\in\text{受影响线路}}
z_\ell\frac{a_1-a_0}{2}B_\ell.
\]

因此：

- 当前车辆不在三条线路上时，校正在**标准化四头空间**中的新增动作差为零；
- 在三条线路上时，新增斜率取决于线路、该线路偏移和固定系数，没有站点、客流、载客、其他车辆进度等状态交互。[读出特征与应用位置](https://github.com/erzhu419/Thesis/blob/5b8556e537bf19dd8d6671b041d9e9066d4147f1/experiments/mcwm_closed_loop_action_local_v3_20261020/local_world.py#L74-L153)

这里需要保留一个限定：**不能据此说最终成本差或 Q 差为零。** log-cost 的常数偏移会缩放解码后的成本差，人数偏移也可以通过非线性价值网络改变 bootstrap。准确的结论是：**新增动作关系的表达能力非常受限，最终是否改善了动作排序仍须测量。**

更关键的是，local readout 只改四个 global 头，**没有同步改变下一事件的数量、线路、站点和局部状态生成结构**。四头也没有通过同一条乘客演化轨迹共同导出。[解码实现](https://github.com/erzhu419/Thesis/blob/5b8556e537bf19dd8d6671b041d9e9066d4147f1/experiments/mcwm_bus/control_model.py#L70-L102)

目前的数据恰好呈现了这一接口的问题：

- 同车区间 MAE 显著改善；
- 同流 global 四头 MSE 从 **0.831645 上升到 0.847704**；
- 原两种子的 shock 控制成本上升。

这与“物理变化学对了，但转换为全局动作后果的关系没学对”相符。不过，**waiting 头占预测 MSE 净增量约 74%，仍不能推出它造成了 74% 的控制损失**；成本头还有 bias、RMSE 改善与 MAE 恶化并存的情况。[四头派生指标](https://github.com/erzhu419/Thesis/blob/5b8556e537bf19dd8d6671b041d9e9066d4147f1/experiments/mcwm_closed_loop_v4_review_20261020/review_metrics.json)

**我会把这条“物理预测→事件与动作后果”的接口，列为当前最重要的结构性检查对象。**

---

## 三、第二优先级：context 的时间配对存在一个具体问题

这一点可以比上述结构假说说得更确定。

### 1. 训练 readout 时，用的是样本发出时的 context

`local_world.py` 会在 issue 时保存 `delta_seconds` 和 `features`。标签到达后，拟合 readout 使用保存的旧特征。这符合事前可用信息的约束。[保存与拟合代码](https://github.com/erzhu419/Thesis/blob/5b8556e537bf19dd8d6671b041d9e9066d4147f1/experiments/mcwm_closed_loop_action_local_v3_20261020/local_world.py#L156-L229)

### 2. trust 评分时，却默认使用当前 context

`update_trust()` 取累计 `recent` 中的旧记录，调用：

```python
error = losses(self.predict(y["s"], y["a"]), y, self.norm).mean(1)
```

而 `predict()` 内的 `local_features()` 默认读取**当前**的 `physical.context_delta_seconds()`。[trust 代码](https://github.com/erzhu419/Thesis/blob/5b8556e537bf19dd8d6671b041d9e9066d4147f1/experiments/mcwm_closed_loop_action_local_v3_20261020/learning.py#L16-L29)、[context 默认值](https://github.com/erzhu419/Thesis/blob/5b8556e537bf19dd8d6671b041d9e9066d4147f1/experiments/mcwm_closed_loop_action_local_v3_20261020/local_world.py#L140-L153)

所以，限速之后可能出现：

\[
\underbrace{\widehat P_{\theta_t}(s_i,a_i,z_t)}_{\text{当前工况下的预测}}
\quad\text{对照}\quad
\underbrace{(r_i,s'_i)}_{\text{旧工况产生的结果}}.
\]

如果模型开始正确表达“现在运行更慢”，它未必还能复现限速前的结果。此时误差变大，可能反映**工况不同**，而不只是模型不可靠。

**能够确认的是：readout 拟合和 trust 评分使用的条件不一致。尚不能确认的是：这项不一致对最终成本造成了多大影响。**

### 3. 最小修正应先把两种用途显式分开

建议让模型接口明确接收 context，区分：

- **历史样本拟合/评价**：使用该样本当时可用的 context；
- **当前工况预测**：使用当前 context；
- **当前工况作用于旧状态的反事实生成**：单独标识，不能直接拿旧 outcome 当正确答案。

第一轮甚至不必立即改 trust 权重，只需在同一包、同一模型参数下记录：

\[
e_{\text{saved}}
=
L\!\left(\widehat P_{\theta_t}(s_i,a_i,z_i),y_i\right),
\]

\[
e_{\text{current}}
=
L\!\left(\widehat P_{\theta_t}(s_i,a_i,z_t),y_i\right).
\]

按样本产生时间、线路、冲击阶段分组，再检查哪一种评分更能预测**下一批尚未训练标签**的实际误差。

这能回答：当前 trust 是否在惩罚正确的工况变化。

另外，现有 trust 使用 refit 后的训练误差；已有的事前预测记录可以用来检验这种评分的泛化能力。这里不宜简单把所有历史事前误差混在一起，因为它们也来自不同模型版本、不同工况。

---

## 四、第三优先级：world 在适应当前工况，Q 却可能仍在拟合工况混合

当前 world 使用在线 context，actor/critic 仍只接收原 canonical state；模型生成起点还来自累计 recent 中的旧状态。[模型生成与策略输入](https://github.com/erzhu419/Thesis/blob/5b8556e537bf19dd8d6671b041d9e9066d4147f1/experiments/mcwm_closed_loop_action_local_v3_20261020/learning.py#L31-L95)

这允许出现一种冲突：

- 旧真实转移要求 \(Q(s,a)\) 拟合旧工况后果；
- 当前 world 为相近的 \(s,a\) 生成当前工况后果；
- 同一个 Q 没有显式知道两者对应不同的 \(z\)。

**这还不是已证明的状态混叠。** 时钟、headway、人数等可能已经部分表达工况。需要验证的是：在可见状态相近时，不同 context 是否对应不同的控制相关后果。

### 非平稳数据的迟滞，已经有明确的量级依据

不能再用“32k 旧样本淹没 285 新样本”解释，因为 fresh 已经占观察 minibatch 的 25%。

但 fresh 池本身也包含大量冲击前记录。按报告原两种子每种子约 150 条 shock 后记录计算，到末期：

\[
\text{world 重拟合中的新工况占比}
\approx
\frac12\times\frac{150}{285}
=26.3\%,
\]

\[
\text{观察 TD 样本中的新工况占比}
\approx
\frac14\times\frac{150}{285}
=13.2\%.
\]

此外，family selection 的 128 条窗口，按每 300 秒 5 条，约覆盖：

\[
128/5\times300=7680\text{ 秒}.
\]

这与整个 9000–18000 秒适应窗口几乎同量级。它混合了不同工况和不同历史模型版本的事前误差，确实可能产生适应迟滞。[回放、选模与 refit 实现](https://github.com/erzhu419/Thesis/blob/5b8556e537bf19dd8d6671b041d9e9066d4147f1/experiments/mcwm_closed_loop_v1/learning.py#L110-L230)

这些事实支持做“工况条件化、时间窗口、回放相容性”的诊断，但还不足以决定直接删除历史数据。

### 如果诊断支持 context 瓶颈，最小架构改法是什么

把当时可用的历史摘要 \(z_t\) 作为条件，统一进入：

\[
Q(s_t,z_t,a_t),\qquad V(s_t,z_t),\qquad
\pi(a_t\mid s_t,z_t).
\]

实现时有四个关键点：

1. **每条 replay 保存自己的 context。** 包括 issue 时的 \(z_t\)、下一状态当时可用的 context，以及必要的版本和可用时间。
2. **历史缺失就明确标记缺失。** 能按时间顺序重建就重建；不能给所有旧记录补上当前 \(z_t\)。
3. **用零 context 的同维网络作对照。** 新输入权重可零初始化，使改造起点保持原策略输出，减少“换网络”的混淆。
4. **当前 F 的实现固定解析 229 维状态。** 扩维后的网络输入要与公交原状态分开，F 仍从原状态字段读取 headway 等信息，不能直接把扩维向量传入其固定 reshape。[F 实现](https://github.com/erzhu419/Thesis/blob/5b8556e537bf19dd8d6671b041d9e9066d4147f1/experiments/mcwm_closed_loop_action_local_v3_20261020/action_contract.py#L26-L57)

这是一项可以单独验证的变化。不要同时改窗口、模型权重、entropy、网络宽度，否则即使有效，也难以知道原因。

---

## 五、trust 和优化链应该怎样检查

### 1. 0.173 的平均权重，不能解释成“即时可信度只有 17.3%”

代码中：

\[
c=\frac{1}{1+\mathrm{error}/\tau},
\]

\[
Q_\Delta\leftarrow \log c+0.9Q_\Delta',
\qquad
w=\exp(Q_\Delta).
\]

在近似平稳、忽略函数逼近误差的说明性情形下：

\[
w\approx c^{1/(1-0.9)}=c^{10}.
\]

例如即时 \(c=0.8\)，累计权重就约为 \(0.107\)。因此，小权重可能来自累计定义；它不是对单步正确率的直接估计。[QDelta 实现](https://github.com/erzhu419/Thesis/blob/5b8556e537bf19dd8d6671b041d9e9066d4147f1/experiments/mcwm_bus/runtime/MC-WM/mc_wm/policy/qdelta_bellman.py#L122-L157)

需要把两个问题分开：

- trust 是否把**更有害的合成样本排在更低位置**；
- model TD 的**整体强度**是否合适。

一个便宜的诊断是：保持同一模型和相同总权重质量，比较原逐样本权重与常数均值权重。这样先检验权重排序有没有价值，避免直接设成 1 同时改变排序和总强度。

### 2. 六组预测误差，并不直接对应控制风险

当前 global 四头先取均值，再与另外五组损失取均值。单个标准化成本平方误差的显式系数为 \(1/24\)，但这并不意味着它对实际梯度或控制效果的贡献也恰好是 \(1/24\)。[损失定义](https://github.com/erzhu419/Thesis/blob/5b8556e537bf19dd8d6671b041d9e9066d4147f1/experiments/mcwm_bus/model.py#L93-L107)

后续应增加一个**控制相关的事前评分**：

> 同一真实状态、同一执行动作、同一套冻结价值网络下，模型产生的 Bellman target 与真实转移产生的 target 差多少？

进一步再测候选动作之间的差，而不只测一个动作的绝对预测值。

这与 *Value-Aware Model Learning* 和 *Value Equivalence* 的核心观点一致：模型应按它对价值更新的影响来评价。两篇论文可以作为这里的理论依据，但不能直接把其标准折扣 MDP 的结论当成本项目的性能保证。[proceedings.mlr.press](https://proceedings.mlr.press/v54/farahmand17a.html?utm_source=chatgpt.com)

### 3. 4.14% 的原始梯度范数比，也不等于实际策略只受到 4.14% 的影响

现有探针没有执行影子优化。Adam 的历史动量、逐参数缩放，以及之后的 actor 更新，都会改变传导结果。[梯度诊断代码](https://github.com/erzhu419/Thesis/blob/5b8556e537bf19dd8d6671b041d9e9066d4147f1/experiments/mcwm_closed_loop_diagnostic_v2_20261020/td_diagnostics.py#L26-L41)

这里尤其要避免三个误判：

- **actor-before-critic 只造成一次更新滞后。** 一包有 8 次更新，前面 critic 的变化已经有机会影响后面的 actor。
- **固定策略下“后继动作差很小”不是完整学习效应。** 成本头不改 next state 时，该探针为零也很正常。
- **模型生成使用 `no_grad` 不会切断 Dyna 路径。** 模型通过合成 TD 改 critic，再通过 critic 改 actor，是现有链路的正常含义。

同样，F 已经统一，不能再把旧动作错配当当前根因。剩余问题应通过 F 的实际平坦区比例、有效导数和可达执行动作范围测量。proposal 均值 30 秒、执行均值 8.7 秒，本身还不能证明梯度被吞掉。

---

## 六、我建议的最小下一轮实验

### 阶段 A：先利用已有训练状态做诊断，不新增完整控制日

如果服务器保存了完整训练状态，选择少量预先固定的反馈包，覆盖冲击前、初期、中期、晚期。

#### A1. context 时间配对

计算上一节的 `saved_z` 与 `current_z` 两种误差，检查：

- 差异是否主要出现在冲击前旧样本；
- 差异是否随当前校正幅度增加；
- 哪一种更能预测下一包真正的事前误差。

同时检查 9 维设计矩阵的有效秩、条件数，以及 `线路偏移` 与 `线路偏移×holding` 的共线性。若某线路 holding 几乎不变，ridge 可以稳定输出系数，但未必能区分“水平变化”和“动作斜率”。

#### A2. 一整包 8 次更新的影子实验

复制完整训练状态，包括网络、target、V、Adam 状态、温度、RNG 和 replay。采用相同采样索引与随机数，分别执行：

| 分支 | 世界模型输出 |
|---|---|
| Base | 当前基础 world |
| Local | 当前四头局部校正 |
| Local−W | 去掉 waiting 校正 |
| Local−C | 去掉 cost 校正 |

第一轮固定共同 trust 数值，隔离输出校正本身。记录：

- 真实优化后的 critic 参数变化；
- 相同候选执行动作之间的 \(Q(s,a_1)-Q(s,a_0)\)；
- 8 次更新后的 proposal 和实际 holding；
- 下一包未用于拟合的数据上的 Bellman 误差。

**这一步正好承接原定删头实验的问题，但先检查真实的 critic→actor 传导。**

如果只有目前的小汇总，没有完整 optimizer/RNG/训练包状态，这些量不能事后补算。应在下面新增运行中保存少量诊断快照。

### 阶段 B：少量同状态仿真分支，验证动作后果

这是区分“模型映射错误”和“优化没利用好”的关键。

从预先选定的完整 target 状态出发，选择两个或少数几个**经过 F 后确实不同、且可执行的 holding**，其余同批动作保持一致。

完整快照需要包含 SUMO、Python bridge、乘客账本、pending 状态和随机状态；先确认同动作重放可复现，再做动作分支。

测两个不同层次的终点：

**第一层：一步 Bellman contrast。**

固定同一套 Q、V、策略和随机数，用实际 target 转移与模型转移分别计算：

\[
\Delta y_m
=
\widehat y_m(s,a_1)-\widehat y_m(s,a_0),
\]

\[
\Delta y_{\mathrm{target}}
=
y_{\mathrm{target}}(s,a_1)-y_{\mathrm{target}}(s,a_0).
\]

这里调用当前真实 H2O+ target 公式。它衡量：**在现有价值评价器下，模型有没有更准确地描述两个动作的区别。**

**第二层：固定物理时间窗的真实成本 contrast。**

例如预先固定 600 秒，之后都由同一个冻结策略续跑，比较真实累计乘客成本差。

这个终点补充了一步探针的限制：当前车 holding 的后果，可能要经过很多其他车辆事件才出现。各分支应使用同样长的物理时间窗，不能一边运行到该车下站就停止，另一边运行更久。

这些 target 分支是额外诊断查询，应单列预算，不进入“285 条在线监督”的训练臂。600 秒也只是诊断窗，不能替代全日成本。

### 阶段 C：原定 8 日预算，优先改成两个基础对照

沿用原两个开发种子 × stable/shock，每个新臂 4 日。

| 优先级 | 新臂 | 与谁比较 | 回答什么 |
|---|---|---|---|
| **先做：4 日** | **冻结世界模型参数**：direct/residual 参数保持初始化值，继续相同的 real075、model TD、F、选模规则和 trust 规则 | 当前 `action_model` | 在线参数 refit 是否比原世界模型更有用？ |
| **其次：4 日** | **等更新、无 fresh 转移监督**：同样 456 次策略更新，使用历史有效转移补足相应监督批次 | 当前共享 F 的 `real075` | fresh 转移监督相对继续训练旧数据，增加了多少价值？ |

有几个定义必须写清楚：

- 第一臂是**冻结模型参数**，仍允许相同规则的 family selection；它隔离参数 refit。若连 family selection 一并冻结，就应改称“冻结整个 world adaptation 程序”，解释也要相应扩大。
- trust 使用相同算法即可，其数值自然会随模型不同而变化；这是模型变化的下游效应。固定数值 trust 的纯诊断由阶段 A 完成。
- 无 fresh 监督臂若要尽量隔离 outcome，应仍允许 actor 使用已经观察到的当前状态，拆开 actor 状态采样与 critic 有标签转移采样。**绝不能把历史 reward 接到 fresh state 上。**
- 如果实现上把 fresh 状态也全部移除，那么估计的是“新转移及状态覆盖”的总贡献，应如实命名。
- 各闭环臂轨迹会不同。能保证相同初始化、外部种子、反馈规则和预算，不能声称它们实际收到逐条完全相同的 fresh 数据。
- 如果还要求 world 拟合计算量匹配，可以在冻结参数臂做不用于生成的影子 refit；策略更新次数仍严格一致。

**为什么先做冻结 world？** 因为你当前最直接的问题就是 world 的边际价值。现有 `model−real075` 同时包含“加入模型训练项”和“在线更新模型”，尚未拆开这两件事。

---

## 七、根据诊断结果，具体应该改哪里

我不建议把所有可能的修复一次性合并。下面是有明确触发条件的修改顺序。

| 诊断结果 | 下一项修改 |
|---|---|
| 当前 context 对旧 outcome 的误差明显偏高，且不能预测后续失误 | 修正 context 时间配对；重新验证 trust 评分 |
| 加入因果可用的历史 context 后，留出 Bellman/动作差预测改善 | 将 context 一致地接入 Q、V、actor 与 replay |
| 局部 MAE 改善，但真实动作 contrast 不改善 | 停止继续调大当前 9×4 读出的权重，修物理到事件接口 |
| 模型动作 contrast 改善，影子 actor 却没有朝更好的执行动作变化 | 检查 trust 排序、实际 Adam 更新、F 的控制自由度及状态分布 |
| online world 不优于冻结参数 world | 暂停当前 refit 配方的扩种子，先处理监督目标和跨工况混合 |
| 无 fresh 监督对照同样有效 | 调整论文归因：已有收益主要来自继续优化，fresh 的独立价值仍待建立 |

### 如果需要替换 9×4 接口，我建议的最小方向

保留现有 actor–critic 主干，优先复用 source 中可验证的事件推进逻辑，使局部校正进入一个可执行的事件过程：

1. **保留车辆事件记忆。** 对相关在途车辆记录下一站、已执行 holding、释放时间和估计剩余区间时间。
2. **让 holding 修改对应车辆的未来事件。** 其他车辆先到时，被控车的新预计事件仍保留，不能随着 global 决策切换到别车而丢失。
3. **由同一条事件轨迹产生后继状态和成本。** 登乘、下车、乘客到达共同决定 \(W'\)、\(O'\)，并累计
   \[
   C=\int_t^{t+\Delta t}\bigl(2W(\tau)+O(\tau)\bigr)\,d\tau.
   \]
4. **再投影回控制器所需的观察。** 先验证这个接口能否改善动作 contrast，再决定是否扩充策略输入。

这里有一个重要边界：现有物理标签是“holding＋运行/信号＋下一站服务”的复合时长，不能假装已经分别识别了纯运行时间、服务时间和客流机制。可以先保留 source 的其他机制，只校正可观测的组合量；需要更细标签时，再单列信息预算。

这条改法直接对应你最初的目标：**用少量目标反馈修正变化机制，再通过机制的执行结果改善控制。**

---

## 八、对非平稳性和论文贡献的最后判断

非平稳性在这里值得处理，但目前最有证据的不是“必须增加一个复杂漂移算法”，而是三个具体问题：

- 累计 replay、128 条选模窗口跨越了很长的物理时间；
- 当前 context 与旧 outcome 的配对不一致；
- 当前工况信息没有明确成为价值和策略的条件。

优先解决这些问题，比同时修改 Adam、normalizer、discount、entropy 更容易得到可解释结果。特别是，有限日累计乘客成本使用 `discount=1` 可以成立；直接改成 \(\gamma^{\Delta t}\) 会改变优化目标，不能当作常规修复。

论文表述也需要保留当前证据的边界：学习器确实没有接收真实的 `0.65` 速度因子，但受影响的三条线路及上下游区间已经预先指定。当前验证的是**已知候选位置的在线变化校准**，还没有验证自动发现变化位置或自动因果结构。[场景与区间定义](https://github.com/erzhu419/Thesis/blob/5b8556e537bf19dd8d6671b041d9e9066d4147f1/experiments/mcwm_closed_loop_formal_20261020/scenarios.py#L15-L19)

**我对下一步的明确建议是：保留共享 F 的 real075 作为控制基线；先完成 context 时间配对和真实影子优化诊断；用少量同状态动作分支验证模型是否学对 holding 后果；随后先跑冻结 world 参数的 4 日对照。** 只有模型的动作关系得到支持，才值得提高其训练权重或扩展闭环种子。

上述结论来自固定提交中的代码和结果审查，尚不是修复后的实验结论。另外，仓库入口允许加载服务器外部的 H2Oplus 路径；仓库副本与报告相互吻合，但实际运行时文件是否与副本逐字节一致，仍应随下一轮实验记录下来。


