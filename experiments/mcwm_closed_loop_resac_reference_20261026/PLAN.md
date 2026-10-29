# RE-SAC向公交MC-WM移植：本地核对与下一步

本轮只读核对RE-SAC论文、实际公交代码及MC-WM当前接线；未训练、未开模拟、未下载服务器数据。参考的是实际实现及修订稿，不将旧稿的理论负惩罚当成执行代码。

## 可以借用的机制

RE-SAC公交实现的分歧渠道同时作用于critic和actor：

`L_Q = MSE(Q_k,y_k) + 0.01 E[sd_sample,k Q_k(s,a)]`

`L_actor = E[alpha log pi − (mean_k z_k − 2 sd_sample,k z_k) + 0.001 ||a_pi−a_replay||²]`

其中`z_k=Q_k+b_k`。critic项压缩回放事实动作的头间分歧；actor LCB直接改变对高分歧动作的偏好。BC是另一项，不能悄悄连同两项分歧机制一起加入。

来源：[实际loss](../../../RE-SAC/jax_experiments/common/bus_full_components.py:18)、[修订公式](../../../RE-SAC/paper/paper_jmlr.tex:218)。

原文的另一条“aleatoric／权重范数”渠道，实际公交代码是把冻结target-head的L1范数**正向加到续值**，并非给在线critic加可导L1权重惩罚。修订稿的匹配full／no-norm bundle三种子对照中，带norm臂全部更差；不把这个bonus作为已证实有效的稳健正则照搬。RE-SAC的late指标是训练第425／450／475／500日，不是本项目18300–24000秒的episode尾窗。[执行目标](../../../RE-SAC/paper/paper_jmlr.tex:193)、[实验口径与对照](../../../RE-SAC/paper/paper_jmlr.tex:293)。

## 当前MC-WM的关键差别

当前真实初始化只有2个Q头，bootstrap_probability=1，actor为min-all；v72运行汇总中的CQL逐头记录也确认为2头。两个头共享encoder。跨头分歧可作代理，但不是已校准epistemic variance；已有TD标准差混合head与batch，不能据它确认尾窗退化病因。

对2头，`min(Q1,Q2)=mean−sd_population=mean−sd_sample/sqrt(2)`。因此现有actor已经隐含悲观项，不能把“新增LCB”描述为从无到有，也不能无意在min上再扣一份完整LCB。

RE-SAC的reward按batch标准化并乘10；当前MC-WM使用1e−4 reward_scale和Huber TD。系数0.01不具备直接可比的损失尺度。修订RE-SAC最强匹配证据是mean-target相对min-target的改善，而非“任意更强悲观正则都有效”。

来源：[真实初始化配置](../mcwm_closed_loop_formal_20261020/inputs/prepared.json:96)、[MC-WM实际actor/F接口](../mcwm_closed_loop_action_local_v3_20261020/learning.py:84)、[RE-SAC奖励及匹配证据](../../../RE-SAC/paper/paper_jmlr.tex:298)。

## 建议的下一项固定实验

只借用分歧渠道的两个接入点：actor用同状态、F后实际动作的全头LCB替换原min读出；critic在事实动作Q上加入显式spread正则，可复用现有CQL事实Q计算。保持2头初始化、反馈选样、混合backup、CQL、285槽456更新、目标时长、F和trust；暂不把norm bonus、BC、更多heads或mean-target一起加入。系数及基准采样配方须在成本结果之前写定，不使用尾窗结果调参。

先做关闭新增项的原8步参数／Adam／RNG保真及非零梯度接线检查，技术通过后直接做固定全天fresh／no_fresh对照。新两臂采用相同正则，同时保留旧CQL no_fresh及v70／v72 fresh作为成本参照；原全日与稳定尾窗仍为推进依据。包内和实际执行时顺带记录同状态头间spread、动作差spread及F后holding，不另开原生诊断批次。

这是一项待验证的移植方案，不是RE-SAC完整复现，也没有确认MC-WM尾窗损失已由epistemic uncertainty解释。当前不启动新实验。
