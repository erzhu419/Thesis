# v115：换乘风险在动作时是否可见

本笔记仅读本地源码及 v114 已有小汇总；新增原生仿真、训练、服务器下载均为 0。

**结论：现 actor 没有显式换乘意图、临近跨线车辆或接驳时限。原生桥接对象有部分当前信息，可原样读取而不修改 F；现有日志不足以复原完整的根时 ETA。冲击根的近端换乘与稳定根的远端网络后果不能合并成一个已识别的局部机制。**

## 实际字段及边界

| 信息 | 当前 actor/raw | 当时存在的字段/API与含义 |
|---|---|---|
| 本车与同线前后车 | 有 | 线路、站序、前后车距、计划车距、两存在标志；后车距是距离/当前路段速度估计，前车依据已发生服务记录，不是准确到站预报。 |
| 排队与容量 | 有 | `eligible_stranded_passengers` 只数当前已在站、可搭本线路的人；`onboard_passengers`、`bus_capacity` 为当前人数/容量。canonical 保留容量比例，排队比例上限2、载客比例上限1；没有乘客ID或换乘分类。 |
| 到达需求与服务时长 | 有 | `eligible_arrival_rate_per_second` 来自当前小时的预加载OD到达率，并按可用线路计划班次分配；不是“下一辆车将送来多少换乘者”。`base_dwell_seconds` 是到站时按上下客估计的基础服务时长，不含未来额外上客。 |
| 同时事件与全网人数 | 有 | 最多16个**当前决策就绪事件**的局部槽位、同批已选动作，以及全网等待/在车人数；不是全网活动车辆状态表。两固定根都只有1个事件。 |
| 临近跨线车 | 桥接层有，当前actor无 | `DecisionEvent.co_line_forward_headway/co_line_backward_headway` 已由共享站其他线路的活动车距离与路段速度计算；取最近距离、截断、无车时用默认班距。无车辆身份、车载换乘人数或确定ETA。 |
| 当前车辆与乘客身份 | 原生有，actor无 | `bridge.active_bus_ids`、`bus_obj_dic`；`traci.vehicle.getPersonIDList/getSpeed/getDistance/getLaneID/isAtBusStop/getStops`、`traci.busstop.getPersonIDs`。对象有 `belong_line_id_s/distance_n/next_stop_id_s`；读取不推进仿真。即时身份应以原生getter为准，legacy乘客状态只在站界同步。 |
| 乘客意图与线路连接 | 预加载仿真元数据，actor无 | `passenger_obj_dic[id].transfer_stop_id_s/end_stop_id_s`、`line_obj_dic[line].stop_id_l`、`stop_obj_dic[stop].service_line_l` 实际存在。对已出现/当前车载的人，可在声明“意图已知”先验后识别潜在换乘；不能读取未出现乘客的精确未来到达，亦不能把计划路线当实际后续行程。 |

**当前leg应读取原生stage0。** 项目v79已实际验证 `person.getStage(id, 0)` 可读，返回 `type=3`、当前leg的 `line` 许可线路集合及 `destStop`；无需用legacy静态字段猜当前目的地（[v79结果:16](/home/erzhu419/mine_code/Thesis/experiments/mcwm_closed_loop_native_context_v79_20261027/RESULTS_AND_NEXT.md:16)、[实际调用:115](/home/erzhu419/mine_code/Thesis/experiments/mcwm_closed_loop_native_context_v79_20261027/observer.py:115)）。预知下一腿的许可线路/换乘意图仍需额外先验，不能将永久保留的 `transfer_stop_id_s` 直接当作第二腿乘客当前leg的终点；v79可读性也不等于本轮已记录了两个根的stage值。

字段来源：[v4事件字段](/home/erzhu419/mine_code/Thesis/experiments/mcwm_bus/runtime/H2Oplus/bus_h2o/mdp_contract_v4.py:17)、[v7 canonical及归一化](/home/erzhu419/mine_code/Thesis/experiments/mcwm_bus/runtime/H2Oplus/bus_h2o/mdp_contract_v7.py:68)、[原生队列/容量/跨线估计](/home/erzhu419/mine_code/Thesis/experiments/sumo_eval/runtime/H2Oplus/bus_h2o/sumo_env/rl_bridge.py:1347)、[OD率分配](/home/erzhu419/mine_code/Thesis/experiments/sumo_eval/runtime/H2Oplus/bus_h2o/sumo_env/rl_bridge.py:1193)、[乘客静态/动态字段](/home/erzhu419/mine_code/Thesis/experiments/sumo_eval/assets/SUMO_ruiguang/online_control/sim_obj/passenger.py:6)、[逐秒车辆位置更新](/home/erzhu419/mine_code/Thesis/experiments/sumo_eval/assets/SUMO_ruiguang/online_control/sim_obj/bus.py:316)。

跨线信息丢失的具体位置已确认：目标环境固定 `observation_schema="v5"`（[stream.py:302](/home/erzhu419/mine_code/Thesis/experiments/mcwm_closed_loop_v1/stream.py:302)）；[sumo_gym_env.py:231](/home/erzhu419/mine_code/Thesis/experiments/sumo_eval/runtime/H2Oplus/bus_h2o/envs/sumo_gym_env.py:231)仅返回16维v5字段。`co_line_*` 只在另一旧观测分支输出（同文件295–302），不能因 `DecisionEvent` 有字段就称当前actor已使用。v7仅重新打包/归一化这16维事件。

F同样不识别具体换乘：它读取当前事件的同线车距/计划车距，以及时间、载客容量比、需求率、前后车存在性。[v16 reserve](/home/erzhu419/mine_code/Thesis/experiments/mcwm_bus/runtime/H2Oplus/SimpleSAC/evaluate_transit_v16_headway_reserve.py:73)、[v19 floor](/home/erzhu419/mine_code/Thesis/experiments/mcwm_bus/runtime/H2Oplus/SimpleSAC/evaluate_transit_v19_completion_safety_reserve.py:99)。v114两根高、低剂量均已通过同一原F；这不证明F曾预测该次换乘。

## 两个根能识别到哪一步

**冲击 `311X_19`，issue/apply=14547。** 记录的事件时间为14546，同批1车、可上本线队列0、车载42/50。`transfer_15055.0` 已在13915上 `406S_22`；已有前缀exchange可核其14547前未下车。结合其既有意图和活动车辆当前下一站/位置，原则上能表示“已有乘客可能在本站换乘311X”。但actor输入没有该信息；现有ticks也没有当前位置/下一站/跨线车距，不能由本轮小汇总证明在14547已能准确预测5秒后下车。14552下406S、14553上高臂311X，低臂改等至15346，是[事后实现证据](/home/erzhu419/mine_code/Thesis/experiments/mcwm_same_state_holding_v114_20261029/targeted_events.json:195)，不是可直接回填的动作时特征。

**稳定 `705X_17`，issue/apply=17235。** 事件时间17234，同批1车、可上本线队列1、车载50/50。主要受损者 `transfer_8937.0` 此时在另一站 `311S07` 等待，17634才上 `311S_27`；18968/18973的远端换乘差5秒，涉及 `122S_33`。这不是根车上或根站即将到来的已证换乘需求。全网当前等待身份和静态意图可读，但从根动作到此远端时刻差的传递路径尚未识别，根处局部接驳字段不足以直接解释或预测它。[v114已核结果](/home/erzhu419/mine_code/Thesis/experiments/mcwm_same_state_holding_v114_20261029/RESULTS_AND_NEXT.md:17)。

## 本轮可回答与不能回答

可用**截至根时的**已存exchange/visit/tick记录，确认活动车辆、已发生的乘客归属/等待、最近已服务站和静态线路连接；不能使用完成后的visit行所含未来到站/离站时间作为当时ETA。Observer ticks实际只写速度、停站状态、站ID/visit编号、车载人数及阶段，不写lane/distance/next_stop（[observer.py:296](/home/erzhu419/mine_code/Thesis/experiments/mcwm_behavior_replay_v113_20261029/observer.py:296)）。其 `[t,t+1]` 记录使用t时成员；根前最后一个tick不能冒充根时的新身份观测。

静态意图本来就在仿真中预加载（[f_8_create_obj.py:142](/home/erzhu419/mine_code/Thesis/experiments/sumo_eval/runtime/H2Oplus/bus_h2o/sumo_env/case/f_8_create_obj.py:142)），不等于真实部署已有可用的出行意图传感器；尤其不能无条件遍历全体预加载乘客。当前证据支持“先区分原actor缺信息与已有原生信息未接入”，尚不支持宣布增加某个字段、设某个保持阈值即可使全天双成本门通过。本笔记不定义新规则或启动候选。
