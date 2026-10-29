# v42：gate时段与issue前快照同步

只读已有57包日志、已存参数清单及v41分布缓存，不重放SGD、不恢复未保存快照、不启动原生前缀/分支，不回灌诊断真值。初始化50000、目标24000秒、F、trust和v40最终三线同窗/9630入库/285槽456更新方案不变。

`journal.py`分开两种证据：每包第一步的变化minibatch gate均值；同包同state的before/after deterministic proposal所确认的p=0.5跨越。指标来自该步actor优化前，不能当作包后actor快照。旧scores没有连续gate/mean/std，仅日志不能复原50024。`sync_plan.json`定义未来1800秒24次warmup后保存行为actor，issue时冻结其随机Hurdle→F，9630后每步按current actor对原行为分布重算RN；target critic同步不等于actor同步。

预检先精确复用v41基线，再固定旧目标和旧state，仅把假设行为的gate概率对拷成目标gate，保留原mean/std。这是理想gate删组，不是真实50024或采集候选。F后全部原子/连续测度及E_mu logc都按假设行为重算，不能沿用原mu期望或进入训练。真实post24→未来432更新对照保持unavailable；同快照c=1不代表更新后仍为1，原0.9混合路径传播仍在。

一个high任务预留2 CPU，稳定/冲击两场景实际fork并行；不加载checkpoint，仅读其文件清单和既有完成摘要。完整日志与分布记录留服务器，仅收≤24KB汇总。数值检查确认旧v41评分、F映射、重叠积分及Jensen关系；预算清单逐项保持v40最终值。当前没有采集或训练提交路径。

已完成结论见`RESULTS.md`与`decision.json`。t211495的两路计算均通过，父汇总超过24KB；t211496仅运行`finalize.py`消费现有结果，恢复23,240B摘要，无worker重跑。`failure.json`和本地`resident_failure.json`保留初次汇总失败记录，`analysis/finalization.json`记录恢复，不代表当前诊断未完成。`continue_batch.py`是首次任务监视器，恢复后不应再次运行它。
