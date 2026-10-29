# 附录A RE-SAC贝尔曼算子的数学证明

[P3394] 附录A RE-SAC贝尔曼算子的数学证明

[P3395] 本附录先给出固定贝尔曼算子的共同性质，再说明RE-SAC分析所用的最大值贝尔曼更新与熵正则化理想更新如何满足这些条件。实际固定策略采样目标与后继惩罚的对应见第3.3.3节。附录C的固定置信混合和附录D的固定符号惩罚采用同一结论，各自的模式更新与策略响应另行分析。

[P3396 M003712,M003713,M003714,M003715,M003716,M003717,M003718] A.1 基本条件。本附录沿用第3.3节的熵温度系数 $\alpha_{ent}$（即正文中的 $\alpha$）、惩罚系数 $\lambda_{epi}$ 与 $\lambda_{ale}$。记 $R_{pen}$ 为吸收固定惩罚后的等效即时项，$c_{shift}$ 为常数平移量，$\mathcal{V}$ 为一般动作价值聚合映射。

[P3397 M003719,M003720,M003721] 设状态集合 $\mathcal{S}$ 和动作集合 $\mathcal{A}$ 均有限非空，动作价值函数空间为 $\mathcal{Q =}\mathbb{R}^{\mathcal{S \times A}}$，赋予无穷范数（上确界范数）

[P3399 式(A-1) M003722] $\parallel Q \parallel_{\infty} = \max_{s,a}|Q(s,a)|.$

[P3400 式(A-1)] (A-1)

[P3401 M003723,M003724] 这是有限维完备赋范空间。记 $Q_{1} \leq Q_{2}$ 为逐点不等式，$Q + c_{shift}$ 表示在每个状态动作对上加同一常数。

[P3402 M003725,M003726,M003727,M003728,M003729,M003730,M003731] 转移概率满足 $P(s' \mid s,a) \geq 0$、$\sum_{s'}^{}P(s' \mid s,a) = 1$，折扣因子满足 $0 \leq \gamma < 1$。给定奖励 $R(s,a)$、惩罚函数 $\Gamma_{epi}(s,a)$、非负系数 $\lambda_{epi}$ 及标量 $\kappa_{ale}$。

[P3403 M003732,M003733,M003734] “固定”指比较同一算子作用于 $Q_{1}$ 和 $Q_{2}$ 时，上述各量均保持不变。RE-SAC将 $\kappa_{ale}$ 与目标网络权重正则量相联系，即

[P3405 式(A-2) M003735] $\kappa_{ale} = \lambda_{ale}\sum_{l}^{} \parallel W_{l} \parallel_{1}.$

[P3406 式(A-2)] (A-2)

[P3407] 本附录将该标量及目标集成网络给出的惩罚函数视为单步更新内的给定参数。

[P3408 M003736] 最大值聚合与固定温度 $\alpha_{ent} > 0$ 下的软聚合分别为

[P3410 式(A-3) M003737] $V_{Q}(s) = \max_{a}Q(s,a),$

[P3411 式(A-3) M003738] $V_{Q}^{\alpha_{ent}}(s) = \alpha_{ent}\log\sum_{a}^{}\exp\left( \frac{Q(s,a)}{\alpha_{ent}} \right).$

[P3412 式(A-3)] (A-3)

[P3413 M003739] 令 $c_{pen}(s,a)$ 为固定的有效惩罚，并将其吸收至即时项：

[P3415 式(A-4) M003740] $R_{pen}(s,a) = R(s,a) - \gamma c_{pen}(s,a).$

[P3416 式(A-4)] (A-4)

[P3417 M003741,M003742,M003743] 若直接采用当前状态动作对上的惩罚，可取 $c_{pen}(s,a) = \lambda_{epi}\Gamma_{epi}(s,a) + \kappa_{ale}$；与第3.3.3节实际后继状态惩罚对应时，则取固定策略下的条件期望 $\Lambda(s,a)$（其中 $\kappa_{ale} = \lambda_{ale}\kappa_{k}$）。

[P3418] 两类理想改进算子统一写为

[P3420 式(A-5) M003744] $\begin{array}{r}
\mathcal{(T}Q)(s,a) = R_{pen}(s,a) + \gamma\sum_{s'}^{}P(s' \mid s,a)V_{Q}(s'), \\
(\mathcal{T}_{\alpha_{ent}}Q)(s,a) = R_{pen}(s,a) + \gamma\sum_{s'}^{}P(s' \mid s,a)V_{Q}^{\alpha_{ent}}(s').
\end{array}$

[P3421 式(A-5)] (A-5)

[P3422] 式(A-5)以式(A-4)吸收固定惩罚，后续证明只需考察动作聚合算子对输入价值函数扰动的响应。

[P3423 M003745] 固定策略的熵正则评价可另取 $\mathcal{V}_{\pi}Q(s) = \mathbb{E}_{a \sim \pi}\lbrack Q(s,a) - \alpha_{ent}\log\pi(a \mid s)\rbrack$。在策略及熵项固定且期望有限时，该聚合同样满足A.2节的单调性和平移相容性，因而适用A.3节的共同收缩结论。

[P3424 M003746] A.2 动作评价。考虑映射 $\mathcal{V:Q \rightarrow}\mathbb{R}^{\mathcal{S}}$。若它单调并与常数平移相容，即

[P3426 式(A-6) M003747] $Q_{1} \leq Q_{2}\  \Rightarrow \ \mathcal{V}Q_{1}\mathcal{\leq V}Q_{2},\quad\quad\mathcal{V(}Q + c_{shift}\mathcal{) = V}Q + c_{shift},$

[P3427 式(A-6)] (A-6)

[P3428] 则可控制其对输入扰动的响应。

[P3429] 引理A.1。 满足上述两项条件的聚合映射在无穷范数下非扩张：

[P3431 式(A-7) M003748] $\mathcal{\parallel V}Q_{1}\mathcal{- V}Q_{2} \parallel_{\infty} \leq \parallel Q_{1} - Q_{2} \parallel_{\infty}.$

[P3432 式(A-7)] (A-7)

[P3433 M003749,M003750] 证明。 令 $\varepsilon_{dist} = \parallel Q_{1} - Q_{2} \parallel_{\infty}$。由 $Q_{2} - \varepsilon_{dist} \leq Q_{1} \leq Q_{2} + \varepsilon_{dist}$，依次使用单调性和平移相容性，得到

[P3435 式(A-8) M003751] $\mathcal{V}Q_{2} - \varepsilon_{dist}\mathcal{\leq V}Q_{1}\mathcal{\leq V}Q_{2} + \varepsilon_{dist}.$

[P3436 式(A-8)] (A-8)

[P3437] 逐点取绝对值再取上确界即得结论。证毕。

[P3438 M003752,M003753,M003754] 最大值聚合满足这两项条件：逐动作不等式在取最大值后保持方向；每个动作同时增加 $c_{shift}$，最大值也增加 $c_{shift}$。因此 $V_{Q}$ 满足引理A.1。软聚合的对应性质见A.4节。

[P3439 M003755,M003756,M003757] A.3 基本性质。设 $R_{pen}(s,a)$ 为任意固定实函数，$P$ 为固定随机转移核，$\mathcal{V}$ 满足A.2节的两项条件，定义通用算子

[P3441 式(A-9) M003758] $\mathcal{(B}Q)(s,a) = R_{pen}(s,a) + \gamma\sum_{s'}^{}P(s' \mid s,a\mathcal{)(V}Q)(s').$

[P3442 式(A-9)] (A-9)

[P3443 M003759,M003760] 定理A.1。 算子 $\mathcal{B}$ 单调，满足折扣平移性质，并为 $\gamma$-压缩映射：

[P3445 式(A-10) M003761] $\begin{array}{r}
Q_{1} \leq Q_{2}\  \Rightarrow \ \mathcal{B}Q_{1}\mathcal{\leq B}Q_{2}, \\
\mathcal{B(}Q + c_{shift})\mathcal{= B}Q + \gamma c_{shift}, \\
\mathcal{\parallel B}Q_{1}\mathcal{- B}Q_{2} \parallel_{\infty} \leq \gamma \parallel Q_{1} - Q_{2} \parallel_{\infty}.
\end{array}$

[P3446 式(A-10)] (A-10)

[P3447 M003762,M003763,M003764] 证明。 单调性由 $\mathcal{V}$ 的单调性、$P \geq 0$ 和 $\gamma \geq 0$ 得到。对于常数平移，利用概率归一化有

[P3449 式(A-11) M003765] $\mathcal{B(}Q + c_{shift}\mathcal{) - B}Q = \gamma c_{shift}\sum_{s'}^{}P(s' \mid s,a) = \gamma c_{shift}.$

[P3450 式(A-11)] (A-11)

[P3451] 固定项在两个输入相减时精确抵消。由引理A.1，

[P3453 式(A-12) M003766] $\begin{array}{r}
\mathcal{|(B}Q_{1}\mathcal{- B}Q_{2})(s,a)| \leq \gamma\sum_{s'}^{}P(s' \mid s,a\mathcal{)|(V}Q_{1}\mathcal{- V}Q_{2})(s')| \\
 \leq \gamma \parallel Q_{1} - Q_{2} \parallel_{\infty}.
\end{array}$

[P3454 式(A-12)] (A-12)

[P3455] 再对状态动作对取最大值，即得式(A-10)的收缩结论。证毕。

[P3456 M003767,M003768] 将A.1节中的 $R_{pen}$ 和最大值聚合代入，立得RE-SAC算子 $\mathcal{T}$ 的三项性质。固定惩罚仅改变不动点的位置，而不增大算子在单步输入比较中的距离收缩系数。附录C、D只需进一步说明其混合转移仍为合法随机核、合并项仍与输入价值函数无关，即可直接应用同一定理。

[P3457] A.4 熵正则化。

[P3458 M003769,M003770] 引理A.2。 对固定 $\alpha_{ent} > 0$，软聚合 $V_{Q}^{\alpha_{ent}}$ 单调且与常数平移相容，因而具有非扩张性。

[P3459 M003771,M003772] 证明。 有限非空动作集中的指数项均为正，对数良定义。指数、有限求和与对数函数均保持单调次序，故 $Q_{1} \leq Q_{2}$ 蕴含 $V_{Q_{1}}^{\alpha_{ent}} \leq V_{Q_{2}}^{\alpha_{ent}}$。直接计算有

[P3461 式(A-13) M003773] $\begin{array}{r}
V_{Q + c_{shift}}^{\alpha_{ent}}(s) = \alpha_{ent}\log\left\lbrack exp(c_{shift}/\alpha_{ent})\sum_{a}^{}\exp(Q(s,a)/\alpha_{ent}) \right\rbrack \\
 = c_{shift} + V_{Q}^{\alpha_{ent}}(s).
\end{array}$

[P3462 式(A-13)] (A-13)

[P3463] 应用引理A.1即得非扩张性。证毕。

[P3464 M003774] 定理A.2。 固定温度下的RE-SAC软算子 $\mathcal{T}_{\alpha_{ent}}$ 单调，满足

[P3466 式(A-14) M003775] $\mathcal{T}_{\alpha_{ent}}\left( Q + c_{shift} \right) = \mathcal{T}_{\alpha_{ent}}Q + \gamma c_{shift},\quad\quad$

[P3467 式(A-14) M003776] $\parallel \mathcal{T}_{\alpha_{ent}}Q_{1} - \mathcal{T}_{\alpha_{ent}}Q_{2} \parallel_{\infty} \leq \gamma \parallel Q_{1} - Q_{2} \parallel_{\infty}.$

[P3468 式(A-14)] (A-14)

[P3469 M003777] 证明。 由引理A.2，取定理A.1中的 $\mathcal{V}Q = V_{Q}^{\alpha_{ent}}$，并结合A.1节的固定项与随机转移核性质即得。证毕。

[P3470] A.5 收敛性。

[P3471 M003778,M003779,M003780,M003781] 推论A.1。 对满足定理A.1条件的 $\mathcal{B}$，存在唯一不动点 $Q^{*}$。从任意 $Q_{0}$ 出发的精确迭代 $Q_{n + 1}\mathcal{= B}Q_{n}$ 收敛，且

[P3473 式(A-15) M003782] $\parallel Q_{n} - Q^{*} \parallel_{\infty} \leq \gamma^{n} \parallel Q_{0} - Q^{*} \parallel_{\infty}.$

[P3474 式(A-15)] (A-15)

[P3475 M003783,M003784,M003785,M003786] 证明。 当 $\gamma = 0$ 时，算子为常值映射，一次迭代即到达不动点。以下设 $0 < \gamma < 1$。收缩性给出 $\parallel Q_{j + 1} - Q_{j} \parallel_{\infty} \leq \gamma^{j} \parallel Q_{1} - Q_{0} \parallel_{\infty}$。对 $m > n$ 求和有

[P3477 式(A-16) M003787] $\parallel Q_{m} - Q_{n} \parallel_{\infty} \leq \frac{\gamma^{n}}{1 - \gamma} \parallel Q_{1} - Q_{0} \parallel_{\infty}.$

[P3478 式(A-16)] (A-16)

[P3479 M003788,M003789] 当 $n \rightarrow \infty$ 时右端趋于零，空间完备性保证迭代序列收敛。由收缩映射的连续性，极限满足不动点方程；两个不动点的距离不超过其自身的 $\gamma$ 倍，故不动点唯一。将迭代序列与该不动点反复比较，即得式(A-15)。证毕。

[P3480 M003790] 令式(A-16)中 $m \rightarrow \infty$，还可得由初始一步迭代差给出的误差上界：

[P3482 式(A-17) M003791] $\parallel Q_{n} - Q^{*} \parallel_{\infty} \leq \frac{\gamma^{n}}{1 - \gamma} \parallel Q_{1} - Q_{0} \parallel_{\infty}.$

[P3483 式(A-17)] (A-17)

[P3484 M003792,M003793] 该推论统一适用于 $\mathcal{T}$、固定温度的 $\mathcal{T}_{\alpha_{ent}}$，以及附录C、D中满足共同条件的固定算子。

[P3485 M003794] A.6 反馈影响。考虑单状态单动作系统，价值函数退化为实数 $Q_{sc}$。定义含输入相关反馈的算子

[P3487 式(A-18) M003795] $\mathcal{T}_{fb}(Q_{sc}) = (\gamma + \lambda_{fb})Q_{sc},\quad\quad\gamma,\lambda_{fb} \geq 0.$

[P3488 式(A-18)] (A-18)

[P3489 M003796,M003797] 命题A.1。 该算子的精确距离扩张因子为 $\gamma + \lambda_{fb}$；当 $\gamma + \lambda_{fb} \geq 1$ 时，算子不再是严格压缩映射。

[P3490] 证明。 对任意两个输入作差，有

[P3492 式(A-19) M003798] $|\mathcal{T}_{fb}(Q_{sc,1}) - \mathcal{T}_{fb}(Q_{sc,2})| = (\gamma + \lambda_{fb})|Q_{sc,1} - Q_{sc,2}|.$

[P3493 式(A-19)] (A-19)

[P3494 M003799,M003800,M003801] 若 $\gamma + \lambda_{fb} \geq 1$，取 $Q_{sc,1} = 1,Q_{sc,2} = 0$ 即排除任何小于1的收缩系数；当 $\gamma + \lambda_{fb} > 1$ 时，任意不同输入间的距离均被严格放大。证毕。

[P3495 M003802] 这一构造说明，当惩罚项直接依赖当前在训输入价值函数时，该项不再于相减时抵消，其反馈增益会直接改变算子的收缩系数；固定目标网络惩罚则是保持原有 $\gamma$-收缩因子的充分条件。

[P3496] A.7 风险评价。第3.2.5节从最不利运行后果引出保守价值学习。本节梳理三种数学关系：指定分布集合对应的精确对偶正则项、Wasserstein概率距离约束给出的可计算下界，以及CVaR对应的风险包络对偶表示。

[P3497 M003803,M003804,M003805,M003806,M003807] 固定状态动作对，设 $\mathcal{U}$ 是有限后继状态空间上的非空概率分布集合，$p^{\circ}\mathcal{\in U}$ 为名义分布，$V$ 为有限值状态价值向量。支持函数定义为 $\sigma_{\mathcal{U}}(f) = \sup_{p\mathcal{\in U}}\langle p,f\rangle$。令 $\Omega_{\mathcal{U}}(V) = \langle p^{\circ},V\rangle + \sigma_{\mathcal{U}}( - V)$。

[P3498 M003808,M003809] 命题A.2。 最不利期望等于名义期望减去正则项 $\Omega_{\mathcal{U}}(V)$，且 $\Omega_{\mathcal{U}}(V) \geq 0$。

[P3499] 证明。 根据上确界与下确界的关系，

[P3501 式(A-20) M003810] $\begin{array}{r}
\langle p^{\circ},V\rangle - \Omega_{\mathcal{U}}(V) = - \sigma_{\mathcal{U}}( - V) = \inf_{p\mathcal{\in U}}\langle p,V\rangle, \\
\Omega_{\mathcal{U}}(V) = \langle p^{\circ},V\rangle - \inf_{p\mathcal{\in U}}\langle p,V\rangle \geq 0.
\end{array}$

[P3502 式(A-20)] (A-20)

[P3503 M003811] 最后一步使用了 $p^{\circ}\mathcal{\in U}$。证毕。

[P3504 M003812,M003813] 在状态动作矩形集合下，将各处的 $\mathcal{U}$ 取为 $\mathcal{U}_{s,a}$，上述恒等式逐点进入鲁棒贝尔曼更新。Derman等在指定集合条件下进一步建立了奖励、转移鲁棒性与策略、价值正则化之间的对偶对应关系[118]。

[P3505 M003814,M003815,M003816,M003817,M003818,M003819,M003820,M003821] 设有限状态空间配备距离度量 $d_{\mathcal{S}}$。一阶Wasserstein距离 $W_{1}(p,q)$ 为所有边缘分布分别为 $p,q$ 的耦合分布中，状态距离 $d_{\mathcal{S}}(x,y)$ 的最小期望。设 $V$ 满足 $|V(x) - V(y)| \leq L_{V}d_{\mathcal{S}}(x,y)$，$L_{V}$ 为其Lipschitz常数上界，$\varepsilon_{ipm} \geq 0$ 为分布距离半径。

[P3506 M003822] 命题A.3。 对 $\mathcal{U = \{}p:W_{1}(p,p^{\circ}) \leq \varepsilon_{ipm}\}$，有

[P3508 式(A-21) M003823] $\inf_{p\mathcal{\in U}}\mathbb{E}_{p}\lbrack V\rbrack \geq \mathbb{E}_{p^{\circ}}\lbrack V\rbrack - \varepsilon_{ipm}L_{V}.$

[P3509 式(A-21)] (A-21)

[P3510 M003824,M003825] 证明。 对任意边缘分布为 $p,p^{\circ}$ 的耦合 $\chi$，有

[P3512 式(A-22) M003826] $\left| \mathbb{E}_{p}\lbrack V\rbrack - \mathbb{E}_{p^{\circ}}\lbrack V\rbrack \right| = \left| \mathbb{E}_{(x,y) \sim \chi}\lbrack V(x) - V(y)\rbrack \right| \leq L_{V}\mathbb{E}_{\chi}\lbrack d_{\mathcal{S}}(x,y)\rbrack.$

[P3513 式(A-22)] (A-22)

[P3514 M003827,M003828,M003829,M003830] 先对耦合 $\chi$ 取下确界得到Wasserstein距离 $W_{1}(p,p^{\circ}) \leq \varepsilon_{ipm}$，再对集合 $\mathcal{U}$ 内的分布 $p$ 取下确界，即得式(A-21)。证毕。

[P3515 M003831,M003832,M003833] 式(A-21)给出了从名义期望价值中扣除敏感度惩罚项的理论依据。对于采用1-Lipschitz激活函数的前馈神经网络，其全局输入敏感度受各层诱导矩阵范数乘积 $\prod_{l}^{} \parallel W_{l} \parallel$ 控制；RE-SAC在工程实现中采用计算简便的逐层权重 $\mathcal{l}_{1}$ 范数之和 $\sum_{l}^{} \parallel W_{l} \parallel_{1}$ 作为结构化正则代理。

[P3516 M003834,M003835,M003836,M003837,M003838] 以 $C_{traj}$ 表示一条运行轨迹的有界可测累计成本，$\mathbb{P}_{0}$ 为固定策略和名义环境诱导的轨迹概率分布。取 $\beta_{tail} \in (0,1\rbrack$ 表示所关注的最差尾部概率质量，$\eta_{risk}$ 为成本阈值，$(x)_{+} = max(x,0)$。按这一尾部概率约定，定义

[P3518 式(A-23) M003839] ${CVaR}_{\beta_{tail}}(C_{traj}) = \inf_{\eta_{risk}\mathbb{\in R}}\left\{ \eta_{risk} + \frac{1}{\beta_{tail}}\mathbb{E}_{\mathbb{P}_{0}}\lbrack(C_{traj} - \eta_{risk})_{+}\rbrack \right\}.$

[P3519 式(A-23)] (A-23)

[P3520 M003840] 令 $\zeta_{risk}$ 为可测的非负轨迹重加权似然比密度，定义风险包络集合

[P3522 式(A-24) M003841] $\mathcal{Z}_{\beta_{tail}} = \left\{ \zeta_{risk}:\ 0 \leq \zeta_{risk} \leq \frac{1}{\beta_{tail}},\quad\mathbb{E}_{\mathbb{P}_{0}}\lbrack\zeta_{risk}\rbrack = 1 \right\}.$

[P3523 式(A-24)] (A-24)

[P3524] 命题A.4。 在上述定义下，有如下对偶表示：

[P3526 式(A-25) M003842] ${CVaR}_{\beta_{tail}}(C_{traj}) = \sup_{\zeta_{risk} \in \mathcal{Z}_{\beta_{tail}}}\mathbb{E}_{\mathbb{P}_{0}}\lbrack\zeta_{risk}C_{traj}\rbrack.$

[P3527 式(A-25)] (A-25)

[P3528 M003843,M003844,M003845,M003846,M003847] 证明。 对任意可行密度 $\zeta_{risk} \in \mathcal{Z}_{\beta_{tail}}$ 和任意实数阈值 $\eta_{risk}$，由于 $\zeta_{risk}$ 在阈值以下产生非正项、在阈值以上不超过 $1/\beta_{tail}$，结合 $\mathbb{E}_{\mathbb{P}_{0}}\lbrack\zeta_{risk}\rbrack = 1$ 有

[P3530 式(A-26) M003848] $\begin{array}{r}
\mathbb{E}_{\mathbb{P}_{0}}\lbrack\zeta_{risk}C_{traj}\rbrack = \eta_{risk} + \mathbb{E}_{\mathbb{P}_{0}}\lbrack\zeta_{risk}(C_{traj} - \eta_{risk})\rbrack \\
 \leq \eta_{risk} + \frac{1}{\beta_{tail}}\mathbb{E}_{\mathbb{P}_{0}}\lbrack(C_{traj} - \eta_{risk})_{+}\rbrack.
\end{array}$

[P3531 式(A-26)] (A-26)

[P3532 M003849,M003850,M003851,M003852,M003853,M003854,M003855,M003856,M003857,M003858,M003859] 当 $\beta_{tail} \in (0,1)$ 时，累计成本分布函数单调右连续，故存在分位阈值 $\eta_{risk}^{*}$，使得 $\mathbb{P}_{0}(C_{traj} > \eta_{risk}^{*}) \leq \beta_{tail} \leq \mathbb{P}_{0}(C_{traj} \geq \eta_{risk}^{*})$。在 $C_{traj} > \eta_{risk}^{*}$ 处取 $\zeta_{risk} = 1/\beta_{tail}$，在 $C_{traj} < \eta_{risk}^{*}$ 处取 $\zeta_{risk} = 0$，并在 $C_{traj} = \eta_{risk}^{*}$ 处取区间 $\lbrack 0,1/\beta_{tail}\rbrack$ 内的常数以补足期望1，所得密度可行且使式(A-26)取等号；当 $\beta_{tail} = 1$ 时，所有可行密度几乎处处等于1，取不大于成本下界的阈值时两端同等于期望成本 $\mathbb{E}_{\mathbb{P}_{0}}\lbrack C_{traj}\rbrack$。证毕。

[P3533 M003860] 命题A.4表明，最小化尾部风险 ${CVaR}_{\beta_{tail}}$ 等价于在有界似然比重加权分布集合上最小化最坏期望成本[109]。

[P3534] A.8 适用范围。本附录的价值算子结论建立在有限状态动作空间、精确条件期望及单步冻结参数条件之上；当神经网络参数、策略或熵温度随梯度更新演进时，训练过程对应一列时变算子，其渐近误差由式(3-36)的逼近误差与算子漂移共同界定。