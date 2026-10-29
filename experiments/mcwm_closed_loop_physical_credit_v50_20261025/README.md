# v50：物理时间混合残差信用预检

唯一目标及条件推导见 [TARGET.md](TARGET.md)，结果和下一步见 [decision.md](decision.md)。本版没有训练、采集或原生分支入口。

`protocol.json`在评分前固定τ=300、λ=0/1、16个既有M8块、完整45事件截止，以及晚段奖励/末端bootstrap/同步Q偏移探针。`credit.py`复用原float32递推，只改变trace传播系数；`run.py`核对真实源链、原c、λ0及旧目标，再生成冻结分解。有效系数并非新的RN，单步backup仍含0.1V和0.9SAC。六项测试覆盖保真、零时长、支持中断、真实末端、探针不改输入及残差增益。

服务器复用v43真实反馈、v44/v46缓存目标；原始285槽456更新来源保持。v49长流仅有时序，不能补成TD或动作收益真值。完整384组单臂块诊断留本版 `detail/`，本地只收 `summary.json`（44,129B）。所有扰动都是固定数据数值敏感度，不是新标签或环境结果。

`readonly_jobs.json`保留high定义，提交锁超时及直接只读执行见 `submission_failure.json` / `execution.json`。未改全局调度器、旧代码或结果。六项测试入口：`python3 -m unittest experiments.mcwm_closed_loop_physical_credit_v50_20261025.test_credit`。
