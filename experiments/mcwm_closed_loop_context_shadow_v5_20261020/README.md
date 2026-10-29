**公交 MC-WM v5：context 时间配对与整包影子优化。**

第一阶段重放6201/6202×稳定/限速的4个action_local完整日。沿用v4的50,000步初始化、24,000秒目标时长、285条全局标签、57包、456次实际策略更新、source与物理区间预算。只增加被动诊断，实际成本与预算须精确复现v4；旧冻结实验及全部已有结果只读。

在deadline8400/10200/13800/17100的世界模型refit后，比较相同模型参数、family、历史状态/动作上的base、issue时保存context、当前context三路误差，并记录9维设计矩阵的可识别性。参数、family及价值评价器冻结后，再对下一包尚未拟合的5条标签评分；未来current固定为anchor时context，saved使用各次issue时可用context，不使用标签释放后的新信息冒充事前预测。

同4个anchor在实际8步更新前复制完整训练状态。Base、Local、Local-W、Local-C四个分支真正执行各8步，包括critic/actor/V/target/温度与Adam状态；共同随机数、相同采样索引和每步实际trust向量隔离输出校正。Local须与实际8步逐值复现。影子分支不接管环境，不把生成样本加入实际训练，额外128次更新/日单列计算预算；不新增target标签。参数、动作价值差、proposal/执行holding及下一包公共固定target下残差分别报告。

原生预检推进约4200秒，50标签、10包、80次实际更新，仍用完整8步影子；只降低初始化/重拟合计算量。预检先验证复制、未来标签隔离及Local精确复现，再提交4个完整日。

先分析第一阶段完整小汇总，再决定是否运行第二阶段：原2种子×两场景的4日冻结world参数对照。该对照只冻结初始direct/residual参数，family/trust与策略仍按原规则更新，用于隔离参数refit的闭环效果。第二阶段须有第一阶段结果及明确的decision.json；本版不依据附件文字自动启动后续实验。

所有详细日志留服务器，训练复制在服务器内存完成，不拉checkpoint。仅收集服务器生成的小型JSON汇总。整包影子诊断测量学习传导，不能代替真实分支运行的乘客收益，也不将saved context视为真实环境工况。

本批串联入口为 `python3 -u continue_stage_a.py`：已有预检通过后提交4项诊断和服务器汇总，只收集 `context_shadow_summary.json`；状态写 `continuation.log`，PID写 `continuation.pid`。它在第一阶段完成后停止，冻结参数对照另凭分析决定启动。

首个原生预检t204483在首次context anchor处遇到device字符串类型错误，失败输出保留于`preflight/`。修正后统一将world设备表示为`torch.device`，预检复测输出和receipt使用`preflight_device_fix`，旧失败结果不覆盖，初始化、时长及数据/更新预算不变。

第一阶段已完成：复测t204576、完整日t204577–t204580及汇总t204581全部成功，4日成本、预算及初始化精确复现v4。原602088字节汇总未覆盖，以gzip仅拉取138748字节。`review_stage_a.py`从该汇总生成`stage_a_review_metrics.json`。

冲击后历史锚点current比saved四头MSE高约3.91%，主要涉及旧工况标签；下一包30条未训练标签的base/saved/current四头MSE为0.614567/0.574281/0.577430，Bellman目标MAE为0.379967/0.376502/0.376548。历史条件不一致确实抬高误差，但不能据此认定换saved即可恢复控制。saved设计9列均满秩；current的rank7属于常量context列的预期结构。

整包8步后Base与Local在冲击后30状态上的执行holding平均差仅0.00002113秒，相当于正常8步动作变化的0.0233%；下一包TD MSE改善0.131%。稳定场景40条后续标签的Local TD MSE仅恶化0.0157%，并非真实控制收益测量。waiting/cost消融效果很小，尚不支持新增对应完整控制日。

`decision.json`决定推进冻结world参数的4日对照，主比较为既有v4 action_model。`FrozenWorld`固定同样初始化拟合后的source/direct/residual神经参数，保持family选择、trust/QDelta、replay及策略学习，并关闭local读出。先运行独立`preflight_frozen_world`短预检，再提交`frozen_runs`四日；每个完整日保持24000秒、285条全局标签、57包、456次策略更新。完整日与18300–24000窗口分别比较稳定/冲击成本，未通过初始化、预算或冻结检查的结果不进入判断。

本轮仅原两个开发种子；历史锚点重复使用累计标签，影子分支固定实际trust且没有接管环境。下一包saved和anchor-current的context信息时点不同。上述诊断不构成独立确认，也没有隔离fresh标签与继续优化各自的收益。

冻结批次串联入口`python3 -u continue_frozen.py`，短预检t204599通过后自动提交4日与服务器汇总，仅收集`frozen_world_summary.json`。状态与PID另存`frozen_continuation.log/.pid`，不改第一阶段的串联记录。
