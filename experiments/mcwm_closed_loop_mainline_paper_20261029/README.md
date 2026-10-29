# 第六章主线与T-ITS独立论文

2026年10月29日起，按用户明确的新研究目标恢复冻结v73方法，以v75三份新初始化验证作为持续反馈的主结果，以v104作为历史控制器的同环境系统比较。主要评价为全天平均成本下降且观测条件中多数改善；逐格、逐时窗100%无损不再是继续研究的必需条件。v73旧尾窗门失败及其他候选的关闭记录全部保留，不追溯改判。

主结论：匹配no_fresh对照4/6改善、平均降低5.946秒／乘客；历史五actor×六环境比较20/30改善、平均降低7.199秒／乘客。尚未优于最佳历史actor。第一组用于新增结果监督的归因，第二组用于完整系统表现；当前方案不是完整历史MC-WM原样续训后只增加dripping。

| 交付 | 内容 |
|---|---|
| [第六章新版本](/home/erzhu419/mine_code/Thesis/experiments/mcwm_closed_loop_mainline_paper_20261029/CHAPTER6_MAINLINE_20261029.md) | 以20261010单章Markdown为底稿，保留原文，纳入6.3.5、6.4.6–6.4.7与表6-17至6-19；旧Word未覆盖 |
| [章节整合与Word衔接说明](/home/erzhu419/mine_code/Thesis/experiments/mcwm_closed_loop_mainline_paper_20261029/CHAPTER_INTEGRATION.md) | 三份母稿章节编号不同，给出正文与对应位置；不冒充已同步Word |
| [结果核对与开发消融](/home/erzhu419/mine_code/Thesis/experiments/mcwm_closed_loop_mainline_paper_20261029/RESULTS_AUDIT.md) | 六格、30项历史比较、三种子离散度、负例、信息预算与开发消融 |
| [英文独立论文源稿](/home/erzhu419/mine_code/Thesis/experiments/mcwm_closed_loop_mainline_paper_20261029/paper/main.tex) | 按T-ITS双栏准备的方法、实验和因果讨论初稿 |
| [英文论文PDF初稿](/home/erzhu419/mine_code/Thesis/experiments/mcwm_closed_loop_mainline_paper_20261029/paper/main.pdf) | 6页工作稿；不是已补齐独立消融、作者与公开材料的投稿定稿 |
| [因果解释及检验边界](/home/erzhu419/mine_code/Thesis/experiments/mcwm_closed_loop_mainline_paper_20261029/CAUSAL_ANALYSIS.md) | 观测／干预、机制迁移、状态信息与动作支持，附可入稿段落 |
| [Astra结论与退化报告索引](ASTRA_CONCLUSIONS_AND_DEGRADATION.md) | 汇集原评议、负例、容量支路关闭结论及因果假说，区别建议与实验证据 |
| [专刊与补充工作](/home/erzhu419/mine_code/Thesis/experiments/mcwm_closed_loop_mainline_paper_20261029/SUPPLEMENT_PLAN.md) | 已补内容、投稿前实质缺口与有限实验设计 |
| [表格数值来源](/home/erzhu419/mine_code/Thesis/experiments/mcwm_closed_loop_mainline_paper_20261029/results_tables.json) | 原小汇总的可审查摘录，不改旧结果 |

论证主线固定为：公交运行中稀疏的当前结果能在匹配继续学习之外提供平均服务收益；状态与结果选样分离是当前可复用的学习安排；负例显示这种收益有条件。因果视角用于解释可能的可迁移性缺口，不包装成已识别的混杂根因或已经实现的因果校正算法。

目标专刊为IEEE T-ITS的“Novel Methods for Modeling and Control of Mobility and Traffic Systems”；用户提供的征稿截止日为2026年10月31日。期刊官网通用作者规范已核对；专刊列表为动态嵌入，本轮未独立读取该条日期，提交时以系统实际选项为准。投稿动作尚未执行。

本目录是新增稿件与证据包。没有修改旧代码、协议、结果或论文Word；结果整理阶段没有新增模拟、训练、权重下载或服务器数据拉取。另在独立目录启动冻结六日组件消融，追加上限144000模拟秒，结果尚不属于本文已完成证据。后续任务继续high。尚缺的实验不能写成已完成；投稿前需按补充计划处理评价范围和作者／公开数据材料。

最终稿已补入v114／v118的不同续策与服务条件证据；这些例子用于机制讨论，不作为v75退化的因果证明。20261029最终编译通过：6页、摘要219词、引用完整、无overfull box；已目视检查新增讨论所在末页。六日消融首8步预检通过、全部原生任务运行中：t212286与t212288–t212292；汇总接续t212287。
