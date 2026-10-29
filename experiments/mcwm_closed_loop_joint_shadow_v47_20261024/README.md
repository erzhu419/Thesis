# v47：单步M1/M8同流累计影子对照

保持v43初始化和原57包285槽456更新时序，λ=0；唯一改动是正动作Q−αlogπ联合项8样本均值。M1走原生路径，M8保留共同首抽样并以隔离持久流提供额外7次抽样。

先以稳定场景重放全部57包，核对源采样、指标、Adam步数、训练RNG及post24/post456完整模型/Adam。预检计算456步单列。通过后正式两场景四worker从同初始化并行，两臂参数和Adam跨包保留。

固定六条已有root记录、六个共同候选proposal及原issue记忆/时钟，每包记录价值水平、动作优势、actor提案、F后holding及别名；评分不入训练。没有新原生分支或标签，旧代码结果只读。完整权重、逐包日志和评分留服务器；主汇总≤48KB，预检与冻结补充小JSON另计。所有提交任务high。

真实在线路径已核实为一次combined256目标（历史64/fresh64/source128），统一h2oplus；offline_objective=iql指离线初始化。正式同场景两臂在初始评分后等待彼此ready，最终按真实训练时间交集确认并行。

已提交原57包预检t211552并通过；正式t211553–t211556，全部high。继续器先复用预检receipt，再自动汇总正式结果，不重复提交既有任务。

结果已完成，见[decision.md](decision.md)与decision.json。新增policy_summary.py读取post24/post456冻结模型补实际随机策略的门概率与期望Fholding；coverage_summary.py读取post456补实际策略动作区。两份补充与主结果分文件保存，不重训、不进回放。

t211570仍是high队列任务；调度器watch同锁自等待期间，已直接在服务器完成只读汇总，analyze.py会复用已有产物。原失败/恢复记录保留在recovery_notes.json与continuation.log。
