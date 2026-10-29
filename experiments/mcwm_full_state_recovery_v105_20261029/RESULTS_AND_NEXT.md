# v105 完整学习态重建结果

固定6201：未通过固定恢复门，保留本次结果并停止扩展。

决定：`HOLD_IDENTITY_MISMATCH_NO_EXPANSION`。
预算：{"reused_offline_updates": 50000, "new_offline_updates": 0, "source_continuation_updates": 20000, "source_events": 24999, "diagnostic_updates": 0, "diagnostic_attempts": 0, "target_native_seconds": 0, "target_feedback_labels": 0}。

完整checkpoint留在服务器：`/home/zhengliang01/scheduleurm_work/Thesis/experiments/mcwm_full_state_recovery_v105_20261029/run/seed6201/full_state.pt`。
本轮没有target日控制或fresh收益结果，也未启动其他种子。

原历史Q/world内存状态已经丢失；本轮核对的是按原流程重建后的actor身份，以及新快照自身的保存/恢复与续步等价。

[详细小摘要](/home/erzhu419/mine_code/Thesis/experiments/mcwm_full_state_recovery_v105_20261029/recovery_summary.json)。
