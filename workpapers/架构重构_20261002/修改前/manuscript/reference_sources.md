# 引文来源说明

本文件为写作来源记录，不并入学位论文正文。统一文献库为`references.json`。当前库含154条，全部在正文引用；已确认同篇arXiv版本的124条采用预印本著录，其余30条保留正式出版或学位论文来源。下文2026-09-28各项是历史核验记录，版本选择以本节最新记录为准。原论文项目的文献库未改动。

## 2026-09-29 逐项语义复核

逐条对照154项的实际引用句及表1-1完整表行，判断见`workpapers/manuscript_audit/参考文献逐条落点对照_20260929.md`。一般研究概述以作者摘要核对；奖励函数、混合价值及分类器公式、鲁棒正则对应另读相关正文。第38、43项仅有DOI与作者机构书目支持，现稿只陈述研究对象，不使用未读取的机制或数值。其余非arXiv条目补查出版方摘要、作者原文及本地学位论文。未确认同篇预印本的条目保留正式来源，检索中的他文引用不算该论文的arXiv版本。

实质改动为修正KAN作者Soljačić残留乱码，并将公交分类器的事件时长输入明确归为本文实现。已核对[Wang和Sun原文式(3)](https://lijunsun.github.io/files/papers/2020-TRC-Bus.pdf)、[H2O+ v2第IV节](https://arxiv.org/html/2309.12716v2)、[DARC第5节式(3)](https://arxiv.org/html/2006.13916)、[Derman等第3至4节](https://arxiv.org/html/2303.06654)。两篇中文论文另据[赵琥等期刊摘要](https://zgglxb.chd.edu.cn/CN/10.19721/j.cnki.1001-7372.2021.07.003)及[李利华等当期目录摘要](https://zgglxb.chd.edu.cn/CN/Y2023/V36/I2)核对具体控制变量；刘鹏杰的场景资料对应本地PDF物理页40至60。

## 2026-09-29 全文引文校订

逐条取得124篇arXiv摘要页的题名、作者、版本年份和摘要；对其余有DOI的条目核对Crossref登记，并以出版方页面、作者机构原文或本地学位论文补充判断。记录存于`workpapers/manuscript_audit/reference_verification.json`。书目身份核验不等于通读全部论文；正文用途及读取层次另见本轮逐条引文对照。

SINDy预印本题名与正式版措辞略有区别，通过arXiv页面关联的DOI确认同篇，保留预印本题名与2015年。Cats停站策略论文补全出版方列出的副题。SUMO作者顺序仍以官方原文为准，不由Crossref覆盖。刘鹏杰论文题名按本地PDF封面著录。未检索到可确认同篇arXiv版本的公交文献不替换成主题相似的预印本。

正文修正COMBO的正则化对象、回报分布的术语及MuJoCo平台出处的位置；IQL、SAC和DARC/H2O的方法来源补入具体公式与算法定义处。摘要不放引文编号，相关来源均保留在正文。中文参考文献的作者省略词使用“等”，英文使用“et al.”。预印本不混入正式版卷期、页码或会议地点。

## 2026-09-28 文献扩写

新增预印本直接核对arXiv摘要页，不补查会议地址、卷期或页码。CSL以预印本类型处理，并保留arXiv链接；年份取首次提交年份。97个预印本候选中，4篇与本轮正文联系较弱，未纳入最终参考文献。题名不符的编号经arXiv实际记录改正后再导入。绪论与第2至6章分别补充公交控制、共享表征、价值估计、非平稳适应、仿真迁移、世界模型及固定数据学习的论述。

新增公交文献除DOI身份核对外，按以下原始来源阅读摘要或相关段落：

- Hickman随机停站：[出版页面](https://pubsonline.informs.org/doi/abs/10.1287/trsc.35.3.215.10150)。
- Dai动态控制点：[出版页面](https://www.sciencedirect.com/science/article/pii/S0191261518307069)。
- Bartholdi自行调节线路：[出版页面](https://www.sciencedirect.com/science/article/pii/S0191261511001676)，仅引用研究对象，不补写未经读取的实施细节。
- Gkiotsalitis站点控制综述：[作者机构原文](https://ris.utwente.nl/ws/files/247991196/Gkiotsalitis_2021_At_stop_control_measures_in_public_.pdf)。
- Rezazada串车综述：[出版页面](https://www.sciencedirect.com/org/science/article/pii/S0144164724000035)。
- Cats停站与频率容量两篇：使用出版方提交的Crossref摘要。
- Ibarra-Rojas公交规划综述：[出版页面](https://www.sciencedirect.com/science/article/pii/S0191261515000454)。
- Sánchez-Martínez动态停站：[出版页面](https://www.sciencedirect.com/science/article/abs/pii/S019126151500257X)。
- Daganzo与Pilachowski车际合作：[出版页面](https://www.sciencedirect.com/science/article/pii/S0191261510000858)。
- Wu超越与乘客分配：[作者机构存档](https://eprints.whiterose.ac.uk/id/eprint/118954/)。
- Xu公交强化学习配置评价：[作者机构成果列表](https://faculty.swjtu.edu.cn/xuzhandong/zh_CN/index.htm)。当前仅确认研究题名和书目，正文仅引用研究对象，不引用具体配置或结果。

He的Q学习停站、Yu的大语言模型奖励研究已取得arXiv版本，按预印本导入。补充内容均为相关方法梳理，不增加任何本文实验结果。

## 公交控制与交通仿真

- `daganzo2009headway`：[作者所在机构页面](https://its.berkeley.edu/node/4645)。实时车头时距控制及线路稳定性。
- `xuan2011dynamic`：[出版页面](https://www.sciencedirect.com/science/article/pii/S0191261511001093)。虚拟时刻表、动态停站与线性控制。
- `delgado2012holding`：[出版页面](https://www.sciencedirect.com/science/article/pii/S0191261512000653)。停站、限制上客与乘客总延误。
- `andres2017predictive`：[出版页面](https://www.sciencedirect.com/science/article/abs/pii/S019126151630892X)。预测与动态停站结合。
- `alesiani2018reinforcement`：[作者上传原文](https://www.researchgate.net/publication/329591205_Reinforcement_Learning-Based_Bus_Holding_for_High-Frequency_Services)。高频公交的强化学习停站；该页面明确注明由作者Gkiotsalitis上传。
- `wang2020real`：[作者提供原文](https://lijunsun.github.io/files/papers/2020-TRC-Bus.pdf)。多智能体动态停站。
- `wang2021asynchronous`：[IJCAI正式论文](https://www.ijcai.org/proceedings/2021/60)。异步控制与图评价器。
- `wang2023robust`：[作者预印本](https://arxiv.org/abs/2111.01946)。回报分布、元学习与运行扰动。本轮改用预印本著录，不再混用2023年正式卷期。
- `wang2023multiobjective`：[出版页面](https://www.sciencedirect.com/science/article/abs/pii/S0968090X2300298X)。多线路及偏好权重。
- `rodriguez2023cooperative`：[出版页面](https://www.sciencedirect.com/science/article/pii/S0968090X23002978)。合作停站与跳站。
- `yu2024hierarchical`：[作者成果页面](https://fbs.ustc.edu.cn/lcx/list.htm)。作者和研究题名可确认，DOI登记确认25(8):9675-9692；IEEE原文此次未能直接读取，绪论只引用其“层级多智能体公交串车控制”研究对象，不据二手摘要写具体上下层机制或效果。
- `li2026graphaware`：[出版页面](https://www.sciencedirect.com/science/article/pii/S1366554525006313)。多线路排队与图表示。正式卷期为2026年207卷，文章号104603，DOI中的2025不作为发表年。
- `zhang2026single`：[正式出版页面](https://academic.oup.com/tse/article/8/2/tdag005/8530551)。作者公交共享控制成果；正式版8(2):tdag005已核实，本轮按要求改用[同篇预印本](https://arxiv.org/abs/2508.20784)。具体实验仍以用户提供原稿和当前结果为依据。
- `resac2026`：[公开预印本](https://arxiv.org/abs/2603.18396)。公开版本书目与当前本地研究稿分别处理，后续诊断结果来自本地已完成记录。
- `lopez2018microscopic`：[SUMO官方引用说明](https://sumo.dlr.de/userdoc/Publications.html)、[DLR原文](https://elib.dlr.de/127994/1/08569938.pdf)。平台出处及作者顺序。Crossref将Wiessner置于第二位，与原文和官方引用不符，保留原文中末位顺序。

## 价值学习、非平稳性与约束

- `haarnoja2018soft`：[PMLR](https://proceedings.mlr.press/v80/haarnoja18b.html)。SAC熵正则化演员评论家。
- `Fujimoto2018TD3`：[PMLR](https://proceedings.mlr.press/v80/fujimoto18a.html)。双评价器与延迟更新。
- `Osband2016BootstrappedDQN`：[NeurIPS](https://proceedings.neurips.cc/paper_files/paper/2016/hash/8d8818c8e140c64c743113f563cf750f-Abstract.html)。随机价值函数与深度探索。
- `chen2021redq`：[作者预印本](https://arxiv.org/abs/2101.05982)。高更新数据比、集成及随机子集目标；正式发表信息为ICLR 2021。
- `Depeweg2018Decomposition`：[PMLR](https://proceedings.mlr.press/v80/depeweg18a.html)。模型不确定性分解与风险敏感学习。
- `Bellemare2017Distributional`：[PMLR](https://proceedings.mlr.press/v70/bellemare17a.html)。回报分布表述。
- `Geist2019TheoryRegularized`：[PMLR](https://proceedings.mlr.press/v97/geist19a.html)。正则化MDP与贝尔曼算子。
- `Iyengar2005RobustDP`：[INFORMS](https://pubsonline.informs.org/doi/10.1287/moor.1040.0129)。具有结构条件的不确定集合与鲁棒动态规划。
- `adams2007bocd`：[作者预印本](https://arxiv.org/abs/0710.3742)。在线变化点检测及运行长度分布。
- `madsen2020nau`：[作者预印本](https://arxiv.org/abs/2001.05016)、[ICLR原文](https://openreview.net/pdf?id=H1gNOeHKPS)。NAU与NMU结构。
- `arjovsky2019irm`：[作者预印本](https://arxiv.org/abs/1907.02893)。跨环境预测目标的一致性。
- `achiam2017cpo`：[PMLR](https://proceedings.mlr.press/v70/achiam17a.html)。期望累计成本约束；仅为相关方法背景，不表示本文采用了CPO。
- `sutton1999between`：[论文书目及原文入口](https://www.cs.utexas.edu/~shivaram/readings/b2hd-SuttonPS1999.html)。DOI核对正式1999年112(1-2):181-211；用于一般半马尔可夫时间抽象背景，不把公交事件接口归为该文提出的方法。

## 离线与跨动力学学习

- `kumar2020cql`：[NeurIPS](https://proceedings.neurips.cc/paper/2020/hash/0d2b2061826a5df3221116a5085a6052-Abstract.html)。保守离线价值学习。
- `kostrikov2022iql`：[作者预印本](https://arxiv.org/abs/2110.06169)。隐式策略改进与优势加权策略提取；正式发表信息为ICLR 2022。
- `ball2023efficient`：[PMLR](https://proceedings.mlr.press/v202/ball23a.html)。RLPD数据混合。
- `zhou2025wsrl`：[ICLR正式论文](https://proceedings.iclr.cc/paper_files/paper/2025/hash/504491292cb71e7681eedfe0e602b72f-Abstract-Conference.html)。在线预热与后续微调；移除原本地条目中未由正式页面确认的“volume=2025”和页码32343-32368。
- `eysenbach2021off`：[作者预印本](https://arxiv.org/abs/2006.13916)。DARC分类器奖励修正；正式发表信息为ICLR 2021。
- `niu2022h2o`：[作者预印本](https://arxiv.org/abs/2206.13464)。目标日志与源仿真混合训练。
- `niu2025h2oplus`：[作者预印本v2](https://arxiv.org/abs/2309.12716v2)。H2O+通用框架；采用2025年修订版本，不混入ICRA 2025的会议著录，也不把该版本年份混作2023年的首次提交年份。
- `peng2018`：[作者预印本](https://arxiv.org/abs/1710.06537)。动力学随机化及机器人迁移。

## 世界模型与误差传播

- `chua2018pets`：[NeurIPS](https://proceedings.neurips.cc/paper/2018/hash/3de568f8597b94bda53149c7d7f5958c-Abstract.html)。概率集成和轨迹采样。
- `janner2019mbpo`：[NeurIPS](https://proceedings.neurips.cc/paper/2019/hash/5faf461eff3099671ad63c6f3f094f7f-Abstract.html)。模型偏差与短滚动。
- `hafner2020dreamer`：[作者预印本](https://arxiv.org/abs/1912.01603)。潜在空间中的模型想象与行为学习。
- `ross2012agnostic`：[ICML原文](https://icml.cc/2012/papers/833.pdf)。系统辨识的分布差异与策略性能。
- `Asadi2018Lipschitz`：[PMLR](https://proceedings.mlr.press/v80/asadi18a.html)。Lipschitz模型误差传播。
- `brunton2016sindy`：[PNAS](https://www.pnas.org/doi/10.1073/pnas.1517384113)。候选函数库及稀疏动力学识别。
- `lanier2026redraw`：[PMLR](https://proceedings.mlr.press/v331/lanier26a.html)。潜在动力学的残差世界模型适应。
- `todorov2012mujoco`：[正式DOI](https://doi.org/10.1109/IROS.2012.6386109)。物理引擎出处；本文不据此引出新的迁移效果。

## 格式与使用范围

中文合稿以CSL官方样式库中的GB/T 7714-2015顺序编码样式为基础，源文件保留原有作者、来源及许可说明。当前纳入中英文文献，英文作者省略词使用“et al.”，中文在Word构建时输出“等”。与DOI字段重复的DOI解析网址不再同时输出；能提供独立内容的出版页可以与DOI并列。该样式是当前阶段稿的引文实现，学校完整规范仍以用户提供的正式要求为准。

作者姓名保留原文重音符号，例如Muñoz；不因Crossref省略重音而改写。TD3的作者Herke van Hoof按复合姓氏“van Hoof”处理，修正BibTeX转换产生的姓氏前缀后移。DOI登记用于核对，不自动覆盖原论文或正式出版页面。所有研究结果、理论证明和局限仍按对应本地材料组织，文献题名相同不代表不同版本的实证结论相同。
