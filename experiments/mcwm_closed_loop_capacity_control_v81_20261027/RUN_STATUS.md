# v81运行状态

更新：2026-10-27T17:06:06+08:00。8个完整日已提交，全部high；四对双臂均已进入running，后台接续正常，当前无失败。尚无全天成本结论。

6208稳定首对各75项技术检查通过；首8步复用正式预算，Adam连续，45条核心与上下文共同前缀精确一致。其余6日已按预案提交，不等待首对全天。

| 种子 | 场景 | 臂 | 任务 |
| --- | --- | --- | --- |
| seed6208 | stable | fresh | t211927 |
| seed6208 | stable | no_fresh | t211928 |
| seed6207 | stable | fresh | t211929 |
| seed6207 | stable | no_fresh | t211930 |
| seed6207 | local_speed_drop | fresh | t211931 |
| seed6207 | local_speed_drop | no_fresh | t211932 |
| seed6208 | local_speed_drop | fresh | t211933 |
| seed6208 | local_speed_drop | no_fresh | t211934 |

后台进程自动核对本批完成情况并提交一次服务器汇总，结束后停止。实时接续以continue_control_state.json为准。每臂285协变量，fresh另285完整结果，各类456更新、24000秒、同初始化与F/trust；本批无新增初始化训练或原生分支。

结果同时保留匹配no_fresh与v75绝对参照；读出全天、早窗、稳定尾窗、等待/乘车、赢输与最坏格。旧门、旧代码和结果保持。只拉必要小JSON，原始轨迹、上下文、RNG和冻结模型留服务器。
