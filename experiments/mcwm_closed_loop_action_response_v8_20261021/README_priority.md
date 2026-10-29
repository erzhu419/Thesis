本实验自 v8 起，所有新提交任务使用 `priority="high"`，包括预检、诊断和服务器端汇总。`infra.task(...)` 已固定此项；旧版本任务与结果不变。

调度器的 `submit-jsonl --file jobs.json --trusted --json` 接收每条 spec 的顶层 `"priority": "high"`。实际实现位于 `scheduleurm/skill/scheduler_commands/bulk_submit.py` 的 `build_bulk_submit_task`；未指定时默认 `normal`，不能只在 experiment protocol 中写优先级。单任务 `submit` 的对应参数为 `--priority high`。

新脚本统一通过 `infra.task(protocol, label, argv, runner="run.py")` 生成 spec，再由 `infra.submit(rows, batch_name)` 提交。`infra.status(ids)` 保留 `priority` 字段，可用小型状态查询确认提交结果。输入、旧版本与大型中间产物留在服务器；仅上传明确列出的 v8 文件，结果汇总默认最多拉取 150 KB。

执行顺序为 `python3 launch.py stage`、`python3 launch.py preflight`，再以本地 `nohup python3 -u continue_batch.py > continuation.log 2>&1 &` 接续。驱动仅在 `preflight/gate.json` 的 `status=complete` 且 `integrity.all_pass=true` 后提交 12 个 prefix-only 任务，再提交服务器汇总；本阶段复用 v7 后缀真值，不新增 300 秒分支。预检 gate 最多拉取 12 KB，完整预检 summary 留在服务器，最终 `response_summary.json` 最多 150 KB。
