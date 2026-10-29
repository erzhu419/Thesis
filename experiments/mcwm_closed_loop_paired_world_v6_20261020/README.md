公交MC-WM v6：在同一条action_model实际流上比较初始化冻结world与在线world，定位预测、trust和动作价值之间的差异。

沿用6201/6202×稳定/局部限速4个开发日、50000步初始化、24000秒目标时长、285条选中全局标签、57包与456次实际策略更新。已有4096条source行和102条source物理区间只读复用，物理反馈与局部读出保持原action_model的被动行为。实际成本、标签/更新预算与初始化必须精确复现v4 action_model。

发出动作时保存两world各family的预测，在同一选中标签训练前评分，同时报告各自family与共同online family。四个固定锚点8400/10200/13800/17100在refit后冻结模型、family及公共Bellman评价器，下一包5条未训练标签给出六头、四个global输出及Bellman目标误差。初始化冻结孪生的trust持续接受相同replay序列，拆解误差、tau、即时置信、bootstrap和同一actor动作权重；记录为独立诊断计算。

每个锚点从相同learner/Adam/replay/RNG出发，Online/Frozen真正各执行8步策略更新。两支使用共同online family、逐步相同的实际trust权重和采样随机数，只改变神经world参数；Online须精确复现实际更新。记录参数、动作偏好、执行holding变化及下一包公共目标下的真实TD残差，影子分支不接管环境。每个完整日额外64次影子更新，不增加target标签。

先做原生4200秒短预检：world初始化2次、refit1次，50标签、10包、80次策略更新，诊断仍执行完整8步。通过后运行4个完整被动诊断日并在服务器汇总，仅收集小型JSON；所有轨迹、逐步日志和模型留服务器。旧实验及已有结果不覆盖。

下一步据同流未训练标签与动作价值诊断决定：若online神经更新损害公共Bellman目标预测，修订refit；若预测改善而动作价值变差，优先修订决策目标或模型接口；若trust目标与估计权重分离，先定位其误差/tau/bootstrap来源。先做小规模稳定/冲击实际控制验证，再决定正式扩展。

限制：原两个开发种子，重复轨迹诊断不构成独立确认。额外trust和影子计算单列，影子TD与动作差不等于乘客收益；选中标签与trust更新止于约18000，停更后窗口仅比较实际乘客成本。

本地14项测试通过，涵盖原生字符串设备、两种family评分、未来标签可用性、冻结评价器、trust持续更新、实际learner/优化器/RNG精确一致及缺失/失败结果门控。启动入口为`launch.py stage`、`launch.py preflight`，随后`continue_batch.py`自动串联4日和服务器汇总；它在收集`paired_world_summary.json`后停止。

原生预检已提交为t204634；新批次的状态和PID保存在`continuation.log/.pid`，原生预检通过记录另存`native_preflight_gate.json`。
