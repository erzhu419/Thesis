# v124：接收车当前腿观测接口完成

**已在隔离目录实现，4项合成测试及独立复核通过。** 新接口读取动作前接收车车载乘客的当前stage0，返回目的站为本站的人数、读取完整性和原生字段依据；预计就绪时间明确命名为`estimated_service_ready_*`。

完整读取后确无匹配者才返回0；成员或任一当前stage未知时返回None。测试覆盖当前腿与未来／永久目的区分、完整0与缺失、只读及动作前时序，以及“预计就绪已过但仍有人待下车”。旧采集器、历史记录与控制器未改动。

**下一步是12个原状态的被动验证。** 沿用原no_fresh动作与F，稳定前缀到16705秒、冲击到17908秒，各在最后观测点动作执行前停止。拟新增原生预算合计34613秒，零训练／拟合；完整记录留服务器，只回传计数和核对摘要。[固定方案](/home/erzhu419/mine_code/Thesis/experiments/mcwm_receiver_service_v124_20261029/NEXT_STEP.json)尚未提交或启动。

人数不等于剩余服务秒数，零人数也不保证服务已经结束；合成测试不能替代原生有效性验证或控制收益。原决策时机、服务参数、F及双成本门保持。本轮模拟、训练、采集、拟合和服务器下载均为0。

文件：[接口](/home/erzhu419/mine_code/Thesis/experiments/mcwm_receiver_service_v124_20261029/observe.py)、[测试](/home/erzhu419/mine_code/Thesis/experiments/mcwm_receiver_service_v124_20261029/test_observe.py)、[核验结果](/home/erzhu419/mine_code/Thesis/experiments/mcwm_receiver_service_v124_20261029/implementation_checks.json)、[阶段摘要](/home/erzhu419/mine_code/Thesis/experiments/mcwm_receiver_service_v124_20261029/summary.json)。
