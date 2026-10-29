# v66：只约束在线actor，固定v65其他部分

按Astra第一项推进：每包冻结当时的部署actor，8次原SAC更新统一加入固定系数1.0的 `KL(new||packet_reference)`，包含Bernoulli gate与新holding概率加权的正动作Normal KL。fresh/no_fresh采用相同约束；这是软惩罚，不保证KL或实际动作跳变的硬上限。

采样、fresh-V目标、初始化、285槽／57包／456次actor、critic、V和温度更新、24000秒、F及trust均不变。没有叠加延迟actor、CQL、trace、特征或执行规则改动。

开发固定6201／6203、稳定／冲击、fresh／no_fresh共8臂，全部high。6201保留已有收益，6203包含两个已知负例，明确属于开发而非独立确认。先提交6201稳定双臂，首包5槽8步原生预检通过后提交余6臂；首对继续完整日，预检不重复计费。

每包只用其5条已付费当前状态，比较包前后完整分布、gate跨界和同F实际holding秒数，无额外抽样、分支或标签。服务器保留57行小型包日志，最终仅拉按线路／训练时窗汇总与成本端点。

全天、9000–12000和18300–24000秒同时报告新fresh、新no_fresh、原fresh及原no_fresh的W/R和成本。继续条件同时要求优于匹配no_fresh与原no_fresh、保住原fresh全天收益、减少早窗损失且不加重最坏格和稳定尾窗。不能靠做差基线、把fresh压零或KL变小判成功；系数不随新结果调整。

唯一小汇总为 `results/development.json`。常驻入口 `continue_control.py --watch --interval 30`，记录轮次开始／结束与耗时，完成后退出，不自动扩种子。旧代码、原冻结实验和已有结果只读；不拉CSV、轨迹或权重。
