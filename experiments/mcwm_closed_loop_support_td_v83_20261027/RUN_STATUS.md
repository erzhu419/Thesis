# v83运行状态

更新：2026-10-27T20:01:34+08:00。8个全天任务全部high，四对双臂均在running，接续正常，当前无失败。

仅在事实TD中，对F单动作支持记录detach容量θ；前向Q、基础Q学习、actor/CQL/target路径和F保持。28项本地测试通过。

首对6208稳定各77项正式预检通过，首8步计入456预算，45条共同核心/上下文前缀精确一致。首包带上下文的单支持样本数为0，实际阻断剂量也是0；这不冒称剂量验证。正式全日汇总另报所有触发量。

| 种子 | 场景 | 臂 | 任务 |
| --- | --- | --- | --- |
| seed6208 | stable | fresh | t211954 |
| seed6208 | stable | no_fresh | t211955 |
| seed6207 | stable | fresh | t211956 |
| seed6207 | stable | no_fresh | t211957 |
| seed6207 | local_speed_drop | fresh | t211958 |
| seed6207 | local_speed_drop | no_fresh | t211959 |
| seed6208 | local_speed_drop | fresh | t211960 |
| seed6208 | local_speed_drop | no_fresh | t211961 |

每臂同50000步初始化、285协变量、fresh另285完整结果、各类456更新和24000秒，F/trust不变。后台接续一次服务器汇总后停止；实时状态见continue_control_state.json。

同时保留v75/v81参照，核对no_fresh对v81的完整轨迹、记录更新和最终模型/Adam/RNG/成本。主读出为全天成本、稳定尾窗和最坏格；尚无全天收益结论。

旧代码、结果及旧门未动。只拉必要小JSON；模型、轨迹、上下文与RNG留服务器。
