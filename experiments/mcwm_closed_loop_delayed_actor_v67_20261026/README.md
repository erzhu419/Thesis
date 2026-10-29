# v67：每包先8次critic/V，再1次actor/温度更新

按Astra第二项，从原v65无KL版本出发，直接使用原生调度：`online_actor_update_interval=8`、`online_actor_batch_mode=latest`、`online_update_order=critic-before-actor`。先完整加载原50000步初始化及Adam，再设置这三项。

每臂285槽／57包、456次critic/V更新、57次actor/温度更新、24000秒。policy_updates明确表示57次actor更新；原total_updates内部计数仍为456个critic迭代，不混称456次策略更新。原采样、fresh-V目标、F和trust不变。

开发固定6201／6203、稳定／冲击、fresh／no_fresh共8臂，全部high。6201稳定首对先作5槽、8次critic/V、1次actor/温度的原生预检，再放行余6臂；首对继续完整日，不重复运行。参数、Adam及实际optimizer事件核验每包顺序为 `[V,critic]×8 → actor → 温度`。

每包仍只对5条已付费状态做包前后同F诊断。actor快照和KL只用于观测，不进入优化目标。首步保留原TD日志，第8步记录实际actor更新日志。目标仍是全天乘客成本，同时检查早窗、最坏格与稳定尾窗。

保留匹配的新no_fresh，以及原v64/v65的no_fresh/fresh绝对参照；成本门沿用v66，不能靠做坏基线、将fresh作用压零或诊断数值变化过门。固定日程不根据新结果调整，开发通过也不自动扩种子。

常驻入口 `continue_control.py --watch --interval 30`。最终完整12线路分为 `results/development.json` 和 `results/policy_drift.json`，各不超过48KB。汇总终态失败会停止待复核，不无限读取或重复提交。

旧代码、原冻结实验及已有结果只读；仅收必要小JSON／日志，不拉CSV、轨迹或权重。减少7次actor抽样属于本次日程变化，不宣称全流RNG与旧456次actor版本一致。
