# 持续目标数据驱动的公交 MC-WM，v1

新增独立实现，按 roadmap 的闭环接入真实 SUMO 目标交互：历史目标数据与冻结源预测先验 → 当前世界模型；新完成且已到达的目标记录 → 事前误差结算 → 模型/上下文更新 → 策略更新 → 后续目标决策。[第六章方法方案](第六章方法方案.md) 包含整体结构图、公式和下一阶段实验安排。

## 实现入口

- `stream.py`：原生 source/SUMO 事件流、v7 动作执行、延迟/抽稀释放；另存同车同班次的相邻站点复合过程。
- `learning.py`：历史目标、近期目标、源数据分开采样；direct/residual 候选世界模型、QDeltaBellman、一步模型样本及 actor–critic 更新。
- `run.py`：`frozen`、`replay`、`online`、`context` 四个分支，独立输出目录，原论文和 `F-FINAL-225` 不变。

`online` 恢复普通在线 MC-WM；`context` 在重拟合之间使用四维全局头残差 EMA，重拟合后清零；`replay` 只将新目标数据加入策略回放；`frozen` 冻结同初始化策略。后者是本版的冻结对照，不等同于原最终控制器的重现。

## 运行

在 Thesis 根目录执行；需要 Python ≥3.10、已有 PyTorch/numpy/h5py/pygame/pandas/openpyxl/gymnasium 和 SUMO。读取本机完整 H2O+ 运行时与已有小型 NPZ，未下载服务器数据。`THESIS_H2O_ROOT` 可指定完整运行时；源数据、SUMO 网络和场景路径也可通过命令行参数指定。

```bash
.venv/bin/python -m pytest -q experiments/mcwm_closed_loop_v1/test_stream.py experiments/mcwm_closed_loop_v1/test_learning.py

PYTHONPATH=/usr/share/sumo/tools .venv/bin/python experiments/mcwm_closed_loop_v1/run.py \
  --target sumo --arm online --smoke \
  --delay-seconds 30 --record-every 4 \
  --max-target-batches 40 --max-target-seconds 1200 \
  --out experiments/mcwm_closed_loop_v1/runs/my_online_smoke
```

换用 `--arm context/replay/frozen` 时分别写新的输出目录。已有输出不会覆盖。`--target source-probe` 仅用源仿真验证接口；这类运行不属于 SUMO 目标证据。`.venv` 使用的 SUMO Python 模块来自 `/usr/share/sumo/tools`；本机实际版本记录在每次 `summary.json` 中。

默认历史输入为已有 pilot 的小型 `target_fit.npz`，源先验为 `source_9101.npz`；`--history` 和 `--source-history` 可换用包含 `s,a,ns,r,dt`、可选 `terminal` 的 NPZ。`--history-limit` 限制读取条数，`--recent-window` 限制近期目标回放。策略初始化可用 `--initializer /path/to/offline_initializer.pt` 只读加载现有 v7 IQL 初始化器；未提供时使用明确记录步数的小规模 IQL 初始化。

默认 `refit-every=16`，每轮重拟合后做四步策略更新；`--smoke` 改成每两条可用目标记录触发一次、模型和策略各一步。源模型初始化后冻结；源交互仍可继续进入策略回放。真实 TD 批次中源与目标各半，目标批次再由历史和近期各半组成。生成样本按需生成一步转移；critic 使用 `0.75 * real_TD + 0.25 * mean(exp(Q_delta) * model_TD)`，保留权重的绝对大小。模型相关随机数与主学习器分开。

## 已完成的小规模验证

11 项测试通过，覆盖延迟权限、同批零时长事件、跨车与同车语义、实际执行动作、终止成本、事前预测评分、冻结源模型、重拟合后的延迟上下文和模型/策略实际更新。

本机 SUMO 1.27.1，四个分支均运行 40 个事件批次至仿真时刻 1148 秒；反馈延迟 30 秒，每四条保留一条：

| 分支 | 进入训练的新目标记录 | 模型重拟合 | 新策略更新 | 更新后目标决策 |
|---|---:|---:|---:|---:|
| frozen | 0 | 0 | 0 | 0 |
| replay | 9 | 0 | 4 | 33 |
| online | 9 | 4 | 4 | 33 |
| context | 9 | 4 | 4 | 33 |

原始记录保留在 `runs/sumo_{frozen,replay,online,context}_v1/`，延迟上下文修订后的复测见 `runs/sumo_context_v1_delayed_context/`。每次只输出协议、小型进度/汇总、事前预测误差、同车区间和本次训练模型。`prequential.jsonl` 保存预测登记模型版本及完成/到达/评分时间；`physical_intervals.jsonl` 保存身份、起终站点、时间和执行动作，不导出全轨迹或 CSV。部分运行保留未到达标签，不伪造终止结果。早期环境接入失败的协议/错误文件也保留。

## 局限与下一步

当前是单进程在事件边界同步训练，图中的后台更新表示逻辑环节。默认 pilot 历史样本仅包含非终止边界事件，smoke 仅做两步离线初始化；不能作为完整训练或全天控制评价。正式实验需换用包含终止尾段的历史集并共用同一 IQL 初始化器、固定目标运行时长（`--max-target-batches 0`），再比较 A1/A2/A3；当前不启动完整研究批次。

目标是 SUMO 代理，沿用 v7 可见的计划到达率字段；未知需求识别需另行控制该字段权限。本地 SUMO 版本也不同于原冻结评价的 1.25.0。当前模型为事件预测模型，EMA 上下文并非 CaDM 复现；自动结构学习与局部机制更新 A4 尚未实现。同车区间是“本站服务完成后的决策就绪至下一站决策就绪”，包含 holding、运行/信号及下一站服务。四分支短运行只证明闭环和数据权限成立，尚未给出适应收益或因果优势结论。

下一步先用共同目标流比较事前预测，再用相同目标运行时间比较各自策略的闭环恢复表现；普通在线与上下文版本有效后，再推进自动机制分解。
