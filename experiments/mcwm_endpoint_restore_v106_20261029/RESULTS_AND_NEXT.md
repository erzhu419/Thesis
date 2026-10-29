# v105 复现与 v106 恢复核验

历史 seed6201 actor 精确复现；v105 的唯一身份门失败是 `float` 场景键与 JSON 字符串键直接比较。历史与本次均为 24999 个 source 事件、20000 次续训，场景次数为 1／1／0。旧任务 t212168 的失败记录保留，在新版本中更正判定，无需重训。

v106 任务 t212170 已完成：原快照 actor、全部保存字段恢复、桥接快照恢复、直接续一步与恢复后续一步、最终70000步恢复，五项比较均为零差异。运行85.18秒，仅2次诊断更新；新增环境交互、标签、source训练和模型重拟合均为0。约145MB快照留在服务器。

下一步已转入独立 v107：固定6201完整端点、106206稳定／冲击各一对 fresh/no_fresh，保持原0.75真实TD＋0.25模型TD、trust、Adam、学习率和F，只改变完整新转移是否进入真实数据批次。先通过首包技术核验，再完成四日；同时比较同actor的历史冻结成本，不按中途成本调整配方。

限制：原 v105 进程退出前的未加载续步参考没有执行；v106证明的是从已保存端点恢复后的保存／加载续步等价。历史隐态真值仍不可直接核对，本轮也没有 dripping 控制收益结论。

[详细摘要](/home/erzhu419/mine_code/Thesis/experiments/mcwm_endpoint_restore_v106_20261029/summary.json)；[下一步协议](/home/erzhu419/mine_code/Thesis/experiments/mcwm_full_mcwm_dripping_pilot_v107_20261029/protocol.json)。
