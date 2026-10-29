# v125：新增接收车观测通过原生验证

**两任务均exit0；12个固定状态全部完整可读，动作前待下客乘客集合与原日志随后下客集合逐人一致。** 4个非零结果如下，其余8个均为完整读取后的0，没有未知或读取错误。

| 场景／动作时刻 | 动作前待下客 | 原日志随后下客 |
|---|---:|---:|
| 稳定／11910秒 | 2 | 2 |
| 冲击／17005秒 | 2 | 2 |
| 冲击／17604秒 | 2 | 2 |
| 冲击／17908秒 | 5 | 5 |

这证明原先遗漏的服务状态能在动作前取得。预计就绪已过并不等于下客已结束；与此同时，另8个零人数点仍有后续上客和晚离站，单独计数也不能代替完整服务响应。

**下一步只设计一个事实风险目标：实际离站是否晚于`ceil(施加时刻＋已执行hold)`，输入限于待下客人数和hold，保留原时钟基线。** 不把人数直接换算后叠加保持时间。这12个点按异常结果选出，不能用于拟合或估计误报；后续完整样本合同须覆盖原v121每场景285个固定根，并保持原包、时窗和9000秒切点。[设计范围](/home/erzhu419/mine_code/Thesis/experiments/mcwm_receiver_native_v125_20261029/NEXT_STEP.json)已固定，暂不新增采集或拟合。

复现逐条通过24081条issue记录及账本、24710条实际命令；最后两个目标均在apply前停止。新增原生34613秒精确计入，实际并行约371.9秒，总原生墙钟约405.0秒（6.75分钟）。训练、拟合、反馈入训练均0。摘要及进度回传合计17622字节（约17.2KiB），完整记录留服务器，旧代码和结果、原F及双成本门保持。

**限制：** 这些是已暴露的同一种子诊断状态；逐人一致证明此字段在这些状态中的含义和可读性，不是独立预测准确率，更不是优于MC-WM的控制收益。

证据：[原生结果](/home/erzhu419/mine_code/Thesis/experiments/mcwm_receiver_native_v125_20261029/summary.json)、[分析](/home/erzhu419/mine_code/Thesis/experiments/mcwm_receiver_native_v125_20261029/analysis.json)、[调度收口](/home/erzhu419/mine_code/Thesis/experiments/mcwm_receiver_native_v125_20261029/scheduler_final.json)、[回传量](/home/erzhu419/mine_code/Thesis/experiments/mcwm_receiver_native_v125_20261029/readout_receipt.json)。
