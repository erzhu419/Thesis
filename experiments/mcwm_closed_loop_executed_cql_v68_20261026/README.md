# v68：按实际执行动作计算的 critic 保守正则

从原v65无KL、无延迟版本出发，唯一训练变动为 `online_cql_alpha=0.2`。先完整加载原50000步初始化和全部Adam，再启用正则。每臂285槽／57包、critic/V/actor/温度各456次更新、24000秒；F、fresh-V目标、采样和trust保持原样。

每步仅用本来就允许两臂使用的64条历史＋64条当前真实(s,a)。候选为零提案、10个均匀提案、10个当前actor正提案和1个确定性正提案，先经过相同F，执行动作完全相同的候选只计一次；事实动作已执行，不再F。全部Q头计算 `logmeanexp(Q候选)−Q事实`，系数0.2固定，温度1。随机候选在隔离Torch流生成，恢复主随机流；不额外采标签或增加optimizer步。

这是针对本项目F的有限候选CQL改法。F含floor/reserve平坦区域，因此不用提案密度冒充执行动作密度；不声称继承原CQL的理论下界。方法依据：[CQL原论文](https://arxiv.org/abs/2006.04779)。系数沿用原实现offline默认0.2作为预声明工程值，不根据本轮结果搜索。

开发固定6201/6203、稳定/冲击、fresh/no_fresh共8臂，high，成对实际并行。6201稳定首对先作5槽原生预检，每类optimizer各8步，继续完整日；通过后只展开余6臂。预检复用，不额外重复原生运行。alpha=0须在测试中精确复现原v64参数、Adam和主RNG；非零CQL则核实F接线、事实动作、候选去重及实际optimizer计数。

fresh/no_fresh均加同一正则，同时保留原v64/v65 no_fresh与fresh绝对参照。成本条件沿用前两轮：全天必须优于新旧no_fresh，稳定不得退化、冲击有收益，早窗和最坏组改善，稳定尾窗不高于旧基线。不能用做差的新基线、压零fresh或Q水平下降替代乘客成本收益。

仅运行这批固定开发对照，不自动扩种子或扫描正则系数。`continue_control.py --watch --interval 30`自动接续并收齐`results/development.json`与`results/policy_drift.json`，各≤48KB。旧代码和结果只读，不拉CSV、轨迹或权重。
