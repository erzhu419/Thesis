**v65 为什么仍有3/10退化，以及还没有用过的改进**

2026-10-26。本文只回答 v65 合并五种子十组中的三个负例，以及尚未尝试的学习改法。依据当前训练链、冻结结果、上一轮提取的小日志汇总及本轮本地 issue_audit 复算；未训练、改旧代码或新增仿真。

**结论：真实数据没有变假，但从真实观测到动作改善之间，仍有未经验证的价值外推与策略优化。v65 改了 fresh 的 bootstrap，在线 actor 仍直接利用 Q 对替代动作的估计。普通 dripping 系列尚未认真检验限制这一步策略改变的方案，这是有针对性的剩余改进空间。**

**三个坏格实际发生了什么**

差值均为 fresh−no_fresh，单位为秒/全天出发乘客，ΔJ=2ΔW+ΔR。

| 负例 | 全天ΔW | 全天ΔR | 全天ΔJ | 9000–12000的ΔJ | 18300–24000的ΔJ |
|---|---:|---:|---:|---:|---:|
| 6203／稳定 | +0.254 | +0.956 | +1.463 | +5.208 | −3.713 |
| 6203／冲击 | +1.869 | +6.716 | +10.453 | +7.508 | +2.395 |
| 6205／冲击 | −0.361 | +1.790 | +1.069 | +3.482 | −2.317 |

共同点是乘车成本增加，且9000–12000发生明显损失。6203稳定、6205冲击后面追回了一部分，但没有完全抵消；6203冲击尾窗还在继续损失。其中一格没有冲击，不能将三个负例统一归因为冲击响应太慢。

三个坏格没有独有的数值异常。好格6204的actor单步参数变化和值loss峰值反而更大；6205稳定好格和冲击坏格的loss、α与actor步幅相近。不能用大loss、梯度爆炸或温度异常解释这三个负例。

实际F命令总量也不支持“一律减小holding”。三个坏格的全天持车命令秒数差分别为−338.2、+701.8、−491.7。6203冲击中，proposal/8的被动映射总量下降670.3秒，实际F命令总量却增加701.8秒。F代码没有改变，但两条闭环轨迹的状态不同，reserve/floor实际作用也不同。命令总量是各自轨迹上的描述，不能当作同状态策略差，更不等于乘客额外乘车时间。

数据：[v65原汇总](/home/erzhu419/mine_code/Thesis/experiments/mcwm_closed_loop_fresh_value_confirmation_v65_20261025/results/confirmation.json)、[本轮三个坏格及三个新好格复算](/home/erzhu419/mine_code/Thesis/experiments/mcwm_closed_loop_full_project_review_20261026/v65_loss_case_details.json)、[已有更新诊断](/home/erzhu419/mine_code/Thesis/experiments/mcwm_closed_loop_full_project_review_20261026/resident_log_summary.json)。

**为什么多了一条真实信息，反而可能选错动作**

下面只是示意，不是实验数值。原模型估计不holding成本100、holding成本101，所以选择不holding。交通变化后，两动作真实成本变成110、112。新反馈真实地告诉模型“不holding成本110”，但并没有告诉它holding现在成本112。如果后者仍被估成101，策略会改选holding，实际成本从110变成112。

新观测完全正确，错误发生在未观测动作的价值还没有同步校准，却被优化器当成了更好的选择。神经网络共享参数会把新信息推广到其他动作和状态，推广有时正确、有时错误；原策略也不会因收到一条新数据就自动被保留为可回退的可靠参照。这个例子证明“额外真数据必定改善已部署策略”并不成立，不证明它就是三个负例的唯一机制。

v65具备这条风险通路：critic只在事实动作上拟合TD；actor却对零动作和新抽取的正动作评价Q并更新。fresh目标虽改为r+V，actor仍为SAC。V又是从target-Q估出来的未来值，而非观测到的长期乘客收益。因此这次改动没有消除策略优化中的未观测动作估值。[事实回放](/home/erzhu419/mine_code/Thesis/experiments/mcwm_closed_loop_fresh_attribution_v25_20261023/learning.py:77)、[在线actor目标](/home/erzhu419/mine_code/Thesis/experiments/mcwm_closed_loop_action_local_v3_20261020/learning.py:77)。原IQL方法同时使用事实动作内的价值学习与优势加权行为拟合；仅把一部分续值换成V，并不等同于这种完整策略提取方式。[IQL原论文](https://arxiv.org/abs/2110.06169)

此外，当前每5条新反馈触发8次actor、V、critic和温度更新，actor在本次critic更新之前执行；没有相对上一部署策略的KL或行为约束。历史world trust只调模型TD权重，50k IQL的行为拟合只在离线阶段，都不等于当前在线actor已有保守约束。[更新循环](/home/erzhu419/mine_code/Thesis/experiments/mcwm_closed_loop_fresh_value_v64_20261025/control.py:149)、[优化顺序](/home/erzhu419/mine_code/Thesis/experiments/sumo_eval/runtime/H2Oplus/SimpleSAC/transit_v3/trainer.py:1351)。

部署还使用p_hold≥0.5的硬选择，而训练优化的是两分支的概率混合。接近阈值时，概率很小的变化可以变成明显的确定性持车变化。两个Q头的min也不是已校准的独立置信区间。[确定性动作选择](/home/erzhu419/mine_code/Thesis/experiments/sumo_eval/runtime/H2Oplus/SimpleSAC/transit_v4/networks.py:198)。

**优先改法：给在线actor加完整分布的保守约束**

为辨认v65的机制，第一轮应保留v65现有数据采样、fresh-V目标、F、285标签和456次更新，只改变actor更新约束。每包开始保存当时的已部署actor作为本包参考；八步更新期间，给SAC actor目标增加相对参考策略的偏移惩罚。Hurdle策略应同时约束是否holding的概率、正动作均值及标准差，而非只限制平均holding。

可用完整分布KL：KL(Bern(p_new)||Bern(p_ref))＋p_new×KL(N_new||N_ref)。共同的正分支可逆变换不改变该KL。参考策略在包内固定，不随每个minibatch立即追上新策略；这一变动不增加反馈，也不需要第二个正在训练的no_fresh learner。

这针对的是“少量反馈产生的Q误差，立即转化成多次策略移动”。策略更新幅度约束有成熟的原始方法依据，但移植到此处只是候选设计。[TRPO原论文](https://arxiv.org/abs/1502.05477)

最关键的观测应是同一批当前状态上，包前后经过相同F的确定性holding差、gate跨界，以及实际成本。p从.499变成.501时KL仍很小，确定性动作却可能切换；所以只报告KL变小或参数变化小不算成功。如果KL已受控、实际动作仍有明显跳变，就不能继续仅靠调大惩罚系数宣称问题得到修复。

新方案的fresh和no_fresh必须采用同一actor约束，并同时对照旧no_fresh的绝对成本。观察目标是减少早窗和最坏格损失，同时保留全天收益；如果只是把fresh作用全部压到零，或把新的no_fresh做差而获得相对优势，不算完成改进。约束强度须在查看新结果前固定，不同时叠加后面的改法。

**第二个未覆盖方向：按反馈包更新actor，先让critic吸收信息**

当前五条新信息导致八次策略移动。一个明确候选是每包先做八次critic/V更新，最后只做一次actor/α更新。这样保留456次critic更新，但actor变成57次；必须如实说明它改变了actor预算，并给匹配no_fresh同样的日程。基础代码有顺序和间隔设置，但普通dripping系列尚未作这一对照。

它的预期是减少critic尚未吸收新信息时的策略追逐，而不是把“更新顺序改了”直接称为critic可靠。延迟策略更新针对价值估计误差的思路来自TD3原论文；本项目已有双Q并不意味着已经采用了延迟actor。[TD3原论文](https://arxiv.org/abs/1802.09477)

**第三个后备方向：直接抑制未观测动作的无支持高估**

对critic加入保守Q正则，约束同状态未观测动作相对事实动作的虚高，直接针对示意例子中的问题。当前online_cql_alpha=0，普通dripping未采用这条路径。[CQL原论文](https://arxiv.org/abs/2006.04779)

不能直接打开现有CQL参数：基础实现的候选动作尚未经过当前执行映射F，需要先保证正则作用于实际执行动作的Q，避免重引入已经修正的提案／执行不一致。这比actor约束改动更大，优先级低于前两项。仅把全部Q压低而不改善动作选择和实际成本，也不能算成功。

**限制**

以上方向在已查的v1–v65普通dripping实现和协议中尚未找到对应实验。曾经的冻结采集策略、shadow evaluator和固定读出干预，不能替代这些在线actor对照。三种候选都尚未在这里验证，不能保证将30%退化消为零；逐包KL也允许全天累计漂移。现有日志缺少同状态、逐包前后策略与动作优势快照，尚不能确定三个负例各自有多少损失来自未观测动作高估、gate跳变、陈旧样本或早期状态延续。本次没有据此启动新训练。
