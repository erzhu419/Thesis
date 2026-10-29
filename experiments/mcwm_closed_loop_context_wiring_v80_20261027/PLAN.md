# v80：容量上下文的因果端点与学习接线

本轮只落实v79之后的采集合同和冻结接线预检，不新增模拟、反馈或优化器更新。旧代码、结果和旧门结论保持。

唯一候选改动是critic的容量动作交互：`Q_k=Q_original,k+theta_k*mask*z*(a_executed+1)/2`，theta逐head零初始化。z只使用issue前当前下一站兼容队列、车上当前leg计划下车人数与容量。actor/V观察仍为229维，F、trust和原混合backup不变。

`stream.py`沿用原EventStream.step。原issue产生ctx_s；下一真实全网issue提供ctx_ns，允许跨车和零时长，不用同车下一站冒充TD后继。终止后继显式missing。原数值转移、同tick动作顺序、thinning和30秒反馈延迟保持。

`learning.py`保留v73双池和原抽样顺序。协变量池只带ctx_s；fresh完整结果带同记录ctx_s/ns，no_fresh不读取目标r/ns/ctx_ns。历史和源域缺失上下文使用NaN/mask0，不回填v79诊断记录。critic的TD、target-Q、actor评分、V目标Q和CQL都接收对应上下文；actor/V/F本身仍接收原观察。

服务器冻结预检固定6208/6207 × fresh/no_fresh，使用既有初始化与档案的隔离测试副本。检查零上下文的抽样、RNG、目标、损失、旧参数梯度和Adam保真，并用手设非零参数检查新通路确实接通。测试副本不是新增当天监督；不执行continue_step或optimizer.step。

通过后，下一项工作是另立一次预算内全天对照：每臂285协变量，fresh另285完整结果，各类456更新、24000秒及原F/trust保持。两臂同时带同样上下文、实际并行；保留原v75绝对成本参照，报告信息维度扩展。首8步技术验证计入正式456步，不另加影子更新。全天乘客成本为主，并列早窗、稳定尾窗、等待/乘车分解和最坏格。v80本身不提供控制改善证据。
