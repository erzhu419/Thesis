# v48：真实事件链的物理覆盖与信用衰减

结果和下一步见 [decision.md](decision.md)，小汇总为 `summary.json`。

本版只读复用 v43 真实事件/账本、v44 原生目标及 v46 的16块 M8 评分；不是对 v47 新模型重新评分。每场景仍是三个物理起点、六条45事件链。原始285槽/456更新、初始化、24000秒、F/trust保持；本轮训练、反馈、原生分支、目标抽样及权重加载均为0。

- `protocol.json`：在计算前固定数据范围、六个事件前缀、候选和预算。
- `metrics.py`：逐条分开0.9路径与执行校正；真实前缀保留原生bootstrap，分解奖励、bootstrap−匹配Q及舍入。
- `run.py`：服务器核对真实状态/记忆连续性、源预算、M8缓存目标与实际c，计算全部诊断；完整结果留新目录的 `detail/`。
- `repack.py`：仅从已落盘结果封装小汇总。首次超过60KB后通过它恢复，无重算目标。
- `infra.py`：只上传本版文件，下载前检查小汇总大小。`readonly_jobs.json`保留high任务定义；提交锁超时见 `submission_failure.json`。本次计算在node005直接完成，见 `execution.json`、`packaging_execution.json`。

四项聚焦测试覆盖λ0保真、零时长链接、末端扰动、分解闭合和缺失支持；从项目根目录可运行 `python3 -m unittest experiments.mcwm_closed_loop_credit_reach_v48_20261025.test_metrics`。无需重跑旧实验。
