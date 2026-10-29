# v104：历史完整控制器同环境比较已完成

30 个历史策略评估日全部完成，零训练；复用 v75 的 12 个实际控制日。成本为 (2×等待＋乘车)/15079，包含未完成旅程的已累计时间，单位为加权秒／乘客。

| 系统 | 全天平均成本 |
| --- | ---: |
| 历史最终 MC-WM：五个 actor 等权 | 2215.134 |
| v75 匹配 CQL no_fresh | 2213.881 |
| v75 fresh | 2207.935 |

fresh 对历史五 actor 均值平均改善 7.199（0.325%），六个环境格全部改善；no_fresh 也比该历史均值低 1.253。因此，本面板不支持“重做版整体先损失了原 MC-WM 的能力”。

| 环境种子／场景 | fresh−匹配 no_fresh | fresh−历史五 actor 均值 |
| --- | ---: | ---: |
| 106206／稳定 | -6.692 | -4.243 |
| 106206／冲击 | -21.528 | -8.489 |
| 106207／稳定 | -11.642 | -13.976 |
| 106207／冲击 | -1.005 | -10.776 |
| 106208／稳定 | +4.656 | -3.531 |
| 106208／冲击 | +0.533 | -2.181 |

额外反馈的匹配结果仍为平均改善 5.946、4/6 改善、2/6 退化。6208 稳定尾窗对匹配基线的 +3.273 也保留。历史均值对照没有消除这些负结果。fresh 相对历史均值的全天等待减少 7.169、乘车增加 7.139；等待按两倍权重计入后形成净收益。

代码确认两者方法不同：历史版为 50k＋20k 更新、0.75 真实 TD＋0.25 模型 TD；v75 为新 50k 起点＋456 日内更新、real-only TD。保留的 0.1V＋0.9SAC 只是 bootstrap 值混合。当前研究没有检验“完整最终 MC-WM 只加 dripping”。

下一步：保留 v73/v75，不追加选样调参或扩种子。严格的“完整 MC-WM＋dripping”需要先单列完整学习态重建，再设计只改反馈通道的匹配比较：历史代码仅导出 actor，seed6201 实查没有 Q/V、优化器及最终 model/trust 的完整 checkpoint。重新初始化这些模块不能冒充原样续训。恢复 WM 尚无已隔离的性能收益证据，本轮不启动昂贵重建或新候选训练。

限制：这是固定六环境的描述性系统比较，训练种子与预算不同，没有同初始化因果归因或显著性结论。六格胜历史“五 actor 均值”不等于胜过每个历史 actor：历史 6202、6204 的六日均值为 2200.511、2207.278，仍低于 fresh。

执行：720000 原生秒、峰值 29 日并行、零训练、无有效 F 分歧。首个导入错误消耗 0 原生秒，失败记录保留。仅拉取 161683 字节正式摘要，末回合结束约 95 秒后汇总完成；旧代码和结果未改。

数据：[完整分层汇总](/home/erzhu419/mine_code/Thesis/experiments/mcwm_closed_loop_historical_comparison_v104_20261028/summary.json)、[实现与历史身份依据](/home/erzhu419/mine_code/Thesis/experiments/mcwm_closed_loop_historical_comparison_v104_20261028/attribution_inputs.json)、[执行收口](/home/erzhu419/mine_code/Thesis/experiments/mcwm_closed_loop_historical_comparison_v104_20261028/execution_summary.json)、[固定协议](/home/erzhu419/mine_code/Thesis/experiments/mcwm_closed_loop_historical_comparison_v104_20261028/protocol.json)。
