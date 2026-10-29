# Astra结论与退化问题归档索引

本页随2026年10月29日项目归档提交，汇集用户提供的Astra评议、已完成结果及当前主线决定。Astra建议属于研究判断，不作为本轮新增实验结果；各历史版本的关闭记录保留。

## 当前主线

恢复冻结v73方法，以v75为主验证。当前标准为全天平均成本下降且多数格改善，不再要求每格、每尾窗100%无损。

| 比较 | 改善比例 | 平均成本差，秒／乘客 |
|---|---:|---:|
| v75 fresh−匹配no_fresh | 4/6，66.7% | −5.946 |
| v75 fresh−历史五actor逐环境对照 | 20/30，66.7% | −7.199 |

历史面板复用六个fresh日，不是30次独立验证；尚未胜过最佳历史actor。匹配对照用于fresh增量，历史对照用于系统比较，两者不混合归因。

## 退化证据与Astra判断

- **v65**：真实反馈能纠正事实动作，却仍通过估计Q推动actor选择其他动作。保守actor、延迟actor和执行语义CQL是当时提出的三项路线；不能把建议本身写成成功结果。见[负例机制与方案](../mcwm_closed_loop_full_project_review_20261026/V65_LOSS_MECHANISM_AND_OPTIONS.md)。
- **v73／v75**：状态池与结果池解耦形成值得保留的平均收益，但6208稳定／冲击仍分别退化4.656／0.533秒。稳定日等待损失超过乘车收益，且在停更前已出现累计损失；不是只有episode末端才发生。见[v73](../mcwm_closed_loop_v73_review_20261027/RESULTS.md)、[v75](../mcwm_closed_loop_v75_review_20261027/RESULTS.md)、[主结果核对](RESULTS_AUDIT.md)。
- **v81／v83**：容量上下文全局动作支路与单支持TD屏蔽没有形成统一改善。冻结重评分的梯度压力不能当作成本损失贡献率；屏蔽改善稳定却损害冲击，不应继续拼接场景专用版本。见[v81](../mcwm_closed_loop_v81_review_20261027/RESULTS_AND_NEXT.md)、[v83](../mcwm_closed_loop_v83_review_20261027/RESULTS_AND_NEXT.md)。
- **最新因果解释**：更准确的假说是历史经验关系与当前动作效果不完全匹配。区分环境机制变化、行为与状态组成变化、未识别混杂；行为策略变化本身不等于混杂。有限fresh在共享参数与TD自举下向其他状态推广，可能改变原本较好的排序，但尚未识别为6208的确定病因。
- **v114／v118**：已有同状态干预支持holding效果依赖载客、接驳和服务条件。两批续策合同不同，不能池化，也不能当作v75负例的独立因果证明。见[因果分析及原结果链接](CAUSAL_ANALYSIS.md)。

## 原始评议与交付入口

- [Astra早期评议原文](../../markdown/Astra_diagnosis_20261023.md)
- [Astra研究计划整理](../mcwm_closed_loop_progress_review_20261023/ASTRA_REVIEW_AND_PLAN.md)
- [GPT诊断第一版](../../markdown/GPT_diagnosis.md)与[第二版](../../markdown/GPT_diagnosis_v2.md)
- [第六章新版本](CHAPTER6_MAINLINE_20261029.md)
- [独立论文源稿](paper/main.tex)、[参考文献](paper/references.bib)
- [补充工作与专刊计划](SUPPLEMENT_PLAN.md)

六日耦合／解耦消融已另立协议运行，结果未出前不计入上述收益。旧代码、结果与旧门结论不改；稿件仍为未投稿工作稿。
