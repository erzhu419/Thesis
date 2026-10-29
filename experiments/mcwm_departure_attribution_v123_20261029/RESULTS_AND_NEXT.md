# v123：发现基础服务时序与观测语义缺口

**12个固定点已完成日志归因，控制下发和离站标签核对正确。实质问题是：预计服务就绪被用作决策时点，却不保证原生上下客已完成，当前输入也缺少接收车的剩余下客信息。**

| 11条成熟记录的事件分类 | 根数 | 离站预测误差合计 |
|---|---:|---:|
| 动作后仍有下客 | 4 | 16秒 |
| 动作后仅见上客、未见下客 | 7 | 18秒 |

稳定18秒、冲击16秒全部对上。12个访问均只有一次成功控制命令；额外时段车辆确实在站且速度为0。三个“无新增上客”的冲击点都仍在下客：最坏点17908秒发动作，17920秒仍下客，17923秒离站，比动作时钟晚9秒。

**代码层原因已核实。** 项目以`到站时刻＋max(1.5×预计下客人数,2.5×预计上客人数)＋4`估计服务完成，并据此发出决策；服务器相关源码与本地逐段一致。SUMO1.25原生下客使用车辆`boardingDuration×乘客boardingFactor`，上下客共享计时器并可继续延长停站，与这个并行取max的估计不同。[原生下客实现](https://github.com/eclipse-sumo/sumo/blob/v1_25_0/src/microsim/devices/MSDevice_Transportable.cpp#L174-L208)、[原生上客计时](https://github.com/eclipse-sumo/sumo/blob/v1_25_0/src/microsim/transportables/MSTransportableControl.cpp#L298-L332)。

v120只记录接收车载客数、容量、空位和预计就绪年龄；当前腿目的站只对来车读取。因此模型无法从现有字段直接区分“车上仍有人要在本站下车”和“基础服务确已完成”。继续拟合统一离站偏移不能补齐这个状态缺口。

**下一步：在隔离观察模块中增加动作前接收车当前腿目的站为本站的人数及读取完整性，并明确ready字段是估计值。** 先做当前腿／未来腿区分、缺失读取与时序接口测试；原决策时机、服务参数、F及双成本门保持。[固定下一步](/home/erzhu419/mine_code/Thesis/experiments/mcwm_departure_attribution_v123_20261029/NEXT_STEP.json)仅含接口与测试，尚未启动新回放或拟合。旧记录缺失的特征不能用未来下客倒填。

**限制。** 上表是按实际事件分类，不是将34秒逐秒确认为纯上下客耗时；内部服务计时器未被原日志保存。唯一删失点13707秒发出60秒hold、13769秒实际离站，晚于观察窗2秒，仍保留原删失标签。这些结果定位当前服务建模缺口，尚不能解释全部历史fresh控制退化。

新增模拟、采集、优化器更新和模型拟合均0；服务器小汇总合计10168字节（约9.9KiB），完整证据留服务器，旧代码和结果未动。证据：[12点时序](/home/erzhu419/mine_code/Thesis/experiments/mcwm_departure_attribution_v123_20261029/summary.json)、[分析与源码](/home/erzhu419/mine_code/Thesis/experiments/mcwm_departure_attribution_v123_20261029/analysis.json)、[服务器源码核对](/home/erzhu419/mine_code/Thesis/experiments/mcwm_departure_attribution_v123_20261029/source_binding.json)、[回传量](/home/erzhu419/mine_code/Thesis/experiments/mcwm_departure_attribution_v123_20261029/readout_receipt.json)。
