# 附录A RE-SAC贝尔曼算子的数学证明

本附录先给出固定贝尔曼算子的共同性质，再说明RE-SAC分析所用的最大值贝尔曼更新与熵正则化理想更新如何满足这些条件。实际固定策略采样目标与后继惩罚的对应见第3.3.3节。附录C的固定置信混合和附录D的固定符号惩罚采用同一结论，各自的模式更新与策略响应另行分析。

**A.1 基本条件。**本附录沿用第3.3节的$\alpha_{ent}$、$\lambda_{epi}$与$\lambda_{ale}$。$R_{pen}$为包含固定惩罚后的即时项，$c_{shift}$为常数平移，$\mathcal{V}$为一般动作聚合映射；它们均不是新增的随机环境量。

设状态集合 $\mathcal{S}$ 和动作集合 $\mathcal{A}$ 均有限非空，动作价值函数空间为 $\mathcal{Q =}\mathbb{R}^{\mathcal{S} \times \mathcal{A}}$，赋予最大范数

$$\parallel Q \parallel_{\infty} = \max_{s,a}|Q(s,a)|. \tag{A-1}$$

这是有限维完备赋范空间。记 $Q_{1} \leq Q_{2}$ 为逐点不等式，$Q + c_{shift}$ 表示在每个状态动作上加同一常数。

转移概率满足 $P(s' \mid s,a) \geq 0$、$\sum_{s'}^{}P(s' \mid s,a) = 1$，折扣因子满足 $0 \leq \gamma < 1$。给定奖励 $R(s,a)$、惩罚函数 $\Gamma_{epi}(s,a)$、非负系数 $\lambda_{epi}$ 及标量 $\kappa_{ale}$。

“固定”指比较同一算子作用于 $Q_{1}$ 和 $Q_{2}$ 时，这些量均不改变。RE-SAC将 $\kappa_{ale}$ 与网络权重正则量联系起来，例如

$$\kappa_{ale} = \lambda_{ale}\sum_{l}^{} \parallel W_{l} \parallel_{1}. \tag{A-2}$$

本附录把该值及集成网络给出的惩罚函数视为给定参数。

最大值聚合与固定温度 $\alpha_{ent} > 0$ 下的软聚合分别为

$$\begin{aligned}
V_{Q}(s) = \max_{a}Q(s,a),\quad\quad \\
V_{Q}^{\alpha_{ent}}(s) = \alpha_{ent}\log\sum_{a}^{}\exp\left( \frac{Q(s,a)}{\alpha_{ent}} \right).
\end{aligned} \tag{A-3}$$

令$c_{pen}(s,a)$为固定的有效惩罚，并将其吸收到即时项：

$$R_{pen}(s,a) = R(s,a) - \gamma c_{pen}(s,a). \tag{A-4}$$

若直接采用当前状态动作的理想惩罚，可取$c_{pen}(s,a) = \lambda_{epi}\Gamma_{epi}(s,a) + \kappa_{ale}$；与第3章实际后继惩罚对应时，则须取第3.3.3节在固定策略下定义的条件期望$c_{j}^{\pi}(s,a)$，其中$\kappa_{ale} = \lambda_{ale}\kappa_{j}$。不能把后继惩罚未经条件平均就替换为当前状态的集成分歧。

两类理想改进算子统一写为

$$\begin{aligned}
\mathcal{(T}Q)(s,a) & = R_{pen}(s,a) + \gamma\sum_{s'}^{}P(s' \mid s,a)V_{Q}(s'), \\
(\mathcal{T}_{\alpha_{ent}}Q)(s,a) & = R_{pen}(s,a) + \gamma\sum_{s'}^{}P(s' \mid s,a)V_{Q}^{\alpha_{ent}}(s').
\end{aligned} \tag{A-5}$$

式(A-5)以式(A-4)吸收固定惩罚，后续证明只需控制动作聚合对输入价值的响应。

固定策略的熵正则评价可另取$\mathcal{V}_{\pi}Q(s) = \mathbb{E}_{a \sim \pi}\lbrack Q(s,a) - \alpha_{ent}\log\pi(a \mid s)\rbrack$。在策略及熵项固定且期望有限时，该聚合同样满足A.2的单调性和平移相容性，因而适用A.3的共同收缩结论；它与上式的最大值或软聚合承担不同的评价或改进角色。

**A.2 动作评价。**考虑映射 $\mathcal{V:Q} \rightarrow \mathbb{R}^{\mathcal{S}}$。若它单调并与常数平移相容，即

$$Q_{1} \leq Q_{2}\  \Rightarrow \ \mathcal{V}Q_{1}\mathcal{\leq V}Q_{2},\quad\quad\mathcal{V(}Q + c_{shift}\mathcal{) = V}Q + c_{shift}, \tag{A-6}$$

则可控制其对输入扰动的响应。

**引理A.1。** 满足上述两项条件的聚合映射在最大范数下非扩张：

$$\mathcal{\parallel V}Q_{1}\mathcal{- V}Q_{2} \parallel_{\infty} \leq \parallel Q_{1} - Q_{2} \parallel_{\infty}. \tag{A-7}$$

**证明。** 令 $\varepsilon_{dist} = \parallel Q_{1} - Q_{2} \parallel_{\infty}$。由 $Q_{2} - \varepsilon_{dist} \leq Q_{1} \leq Q_{2} + \varepsilon_{dist}$，依次使用单调性和平移相容性，得到

$$\mathcal{V}Q_{2} - \varepsilon_{dist}\mathcal{\leq V}Q_{1}\mathcal{\leq V}Q_{2} + \varepsilon_{dist}. \tag{A-8}$$

逐点取绝对值再取最大值即得结论。证毕。

最大值聚合满足两项条件：逐动作不等式在取最大值后保持方向；每个动作同时增加 $c_{shift}$，最大值也增加 $c_{shift}$。因此 $V_{Q}$ 满足引理A.1。软聚合的对应性质见A.4。

**A.3 基本性质。**设 $R_{pen}(s,a)$ 为任意固定实函数，$P$ 为固定随机核，$\mathcal{V}$ 满足A.2的两项条件，定义

$$\mathcal{(B}Q)(s,a) = R_{pen}(s,a) + \gamma\sum_{s'}^{}P(s' \mid s,a\mathcal{)(V}Q)(s'). \tag{A-9}$$

**定理A.1。** 算子 $\mathcal{B}$ 单调，满足折扣平移性质，并为 $\gamma$ 收缩：

$$\begin{aligned}
Q_{1} \leq Q_{2} & \  \Rightarrow \ \mathcal{B}Q_{1}\mathcal{\leq B}Q_{2}, \\
\mathcal{B(}Q + c_{shift}) & \mathcal{= B}Q + \gamma c_{shift}, \\
\mathcal{\parallel B}Q_{1}\mathcal{- B}Q_{2} \parallel_{\infty} & \leq \gamma \parallel Q_{1} - Q_{2} \parallel_{\infty}.
\end{aligned} \tag{A-10}$$

**证明。** 单调性由 $\mathcal{V}$ 的单调性、$P \geq 0$ 和 $\gamma \geq 0$ 得到。对于常数平移，利用概率归一化有

$$\mathcal{B(}Q + c_{shift}\mathcal{) - B}Q = \gamma c_{shift}\sum_{s'}^{}P(s' \mid s,a) = \gamma c_{shift}. \tag{A-11}$$

固定项在两个输入相减时抵消。由引理A.1，

$$\begin{aligned}
\mathcal{|(B}Q_{1}\mathcal{- B}Q_{2})(s,a)| & \leq \gamma\sum_{s'}^{}P(s' \mid s,a\mathcal{)|(V}Q_{1}\mathcal{- V}Q_{2})(s')| \\
 & \leq \gamma \parallel Q_{1} - Q_{2} \parallel_{\infty}.
\end{aligned} \tag{A-12}$$

再对状态动作取最大值，得到式(A-10)的收缩结论。证毕。

将A.1中的 $R_{pen}$ 和最大值聚合代入，立即得到RE-SAC算子 $\mathcal{T}$ 的三项性质。固定惩罚改变固定点的位置，但在这次输入比较中不增加距离放大系数。附录C、D只需进一步说明其混合转移仍为随机核、合并项仍与输入价值函数无关，即可应用同一定理。

**A.4 熵正则化。****引理A.2。** 对固定 $\alpha_{ent} > 0$，软聚合 $V_{Q}^{\alpha_{ent}}$ 单调且与常数平移相容，因而非扩张。

**证明。** 有限非空动作集中的指数项均为正，对数有定义。指数、有限求和与对数均保持次序，故 $Q_{1} \leq Q_{2}$ 蕴含 $V_{Q_{1}}^{\alpha_{ent}} \leq V_{Q_{2}}^{\alpha_{ent}}$。直接计算有

$$\begin{aligned}
V_{Q + c_{shift}}^{\alpha_{ent}}(s) & = \alpha_{ent}\log\left\lbrack exp(c_{shift}/\alpha_{ent})\sum_{a}^{}\exp(Q(s,a)/\alpha_{ent}) \right\rbrack \\
 & = c_{shift} + V_{Q}^{\alpha_{ent}}(s).
\end{aligned} \tag{A-13}$$

应用引理A.1即得非扩张性。证毕。

**定理A.2。** 固定温度下的RE-SAC软算子 $\mathcal{T}_{\alpha_{ent}}$ 单调，满足

$$\mathcal{T}_{\alpha_{ent}}(Q + c_{shift}) = \mathcal{T}_{\alpha_{ent}}Q + \gamma c_{shift},\quad\quad \parallel \mathcal{T}_{\alpha_{ent}}Q_{1} - \mathcal{T}_{\alpha_{ent}}Q_{2} \parallel_{\infty} \leq \gamma \parallel Q_{1} - Q_{2} \parallel_{\infty}. \tag{A-14}$$

**证明。** 由引理A.2，取定理A.1中的 $\mathcal{V}Q = V_{Q}^{\alpha_{ent}}$，并使用A.1的固定项与随机核即可。证毕。

**A.5 收敛性。****推论A.1。** 对满足定理A.1条件的 $\mathcal{B}$，存在唯一固定点 $Q^{*}$。从任意 $Q_{0}$ 出发的精确迭代 $Q_{n + 1} = \mathcal{B}Q_{n}$ 收敛，且

$$\parallel Q_{n} - Q^{*} \parallel_{\infty} \leq \gamma^{n} \parallel Q_{0} - Q^{*} \parallel_{\infty}. \tag{A-15}$$

**证明。** 当 $\gamma = 0$ 时，算子为常值映射，一次迭代即到达固定点。以下设 $0 < \gamma < 1$。收缩性给出 $\parallel Q_{j + 1} - Q_{j} \parallel_{\infty} \leq \gamma^{j} \parallel Q_{1} - Q_{0} \parallel_{\infty}$。对 $m > n$ 求和有

$$\parallel Q_{m} - Q_{n} \parallel_{\infty} \leq \frac{\gamma^{n}}{1 - \gamma} \parallel Q_{1} - Q_{0} \parallel_{\infty}. \tag{A-16}$$

右端趋于零，完备性保证迭代收敛。由收缩映射的连续性，极限满足固定点方程；两个固定点的距离不超过其自身的$\gamma$倍，故固定点唯一。将迭代与该固定点反复比较，即得式(A-15)。证毕。

令前述 $m \rightarrow \infty$，还得到可由初始迭代差计算的界

$$\parallel Q_{n} - Q^{*} \parallel_{\infty} \leq \frac{\gamma^{n}}{1 - \gamma} \parallel Q_{1} - Q_{0} \parallel_{\infty}. \tag{A-17}$$

该推论统一适用于 $\mathcal{T}$、固定温度的 $\mathcal{T}_{\alpha_{ent}}$，以及附录C、D中满足共同条件的固定算子。

**A.6 反馈影响。**考虑单状态单动作系统，价值函数退化为实数 $Q_{sc}$。定义

$$\mathcal{T}_{fb}(Q_{sc}) = (\gamma + \lambda_{fb})Q_{sc},\quad\quad\gamma,\lambda_{fb} \geq 0. \tag{A-18}$$

**命题A.1。** 该算子的精确距离因子为 $\gamma + \lambda_{fb}$；当此值不小于1时，不是严格收缩。

**证明。** 两个输入相减，有

$$|\mathcal{T}_{fb}(Q_{sc,1}) - \mathcal{T}_{fb}(Q_{sc,2})| = (\gamma + \lambda_{fb})|Q_{sc,1} - Q_{sc,2}|. \tag{A-19}$$

若 $\gamma + \lambda_{fb} \geq 1$，取 $Q_{sc,1} = 1,Q_{sc,2} = 0$ 即排除任何小于1的收缩系数；大于1时，任意不同输入间距离均被放大。证毕。

这一构造说明，输入相关项不再通过相减抵消，其反馈强度会改变算子性质。但输入依赖并不必然破坏收缩，例如 $\gamma = 0.5,\lambda_{fb} = 0.1$ 时距离因子为0.6。因而固定惩罚是保持原有收缩因子的充分条件，不是所有收缩算子的必要条件。此处采用正向反馈 $+ \lambda_{fb}Q_{sc}$；具体减法惩罚的输入依赖性仍需按其定义分析。

**A.7 风险评价。**第3.2.5节从最不利运行后果引出保守价值学习。本节区分三种数学关系：指定分布集合对应的精确正则项、概率距离约束给出的可计算下界，以及CVaR对应的特定风险包络。它们回答评价目标如何转换，与前文固定贝尔曼更新是否收缩是两个相互衔接的问题。

固定状态动作对，设$\mathcal{U}$是有限后继状态空间上的非空概率分布集合，$p^{\circ} \in \mathcal{U}$为名义分布，$V$为有限值状态价值向量。支持函数定义为$\sigma_{\mathcal{U}}(f) = \sup_{p \in \mathcal{U}}\langle p,f\rangle$。令$\Omega_{\mathcal{U}}(V) = \langle p^{\circ},V\rangle + \sigma_{\mathcal{U}}( - V)$。

**命题A.2。** 最不利期望等于名义期望减去该正则项，且$\Omega_{\mathcal{U}}(V) \geq 0$。

**证明。** 根据上确界与下确界的关系，

$$\begin{aligned}
\langle p^{\circ},V\rangle - \Omega_{\mathcal{U}}(V) & = - \sigma_{\mathcal{U}}( - V) = \inf_{p\mathcal{\in U}}\langle p,V\rangle, \\
\Omega_{\mathcal{U}}(V) & = \langle p^{\circ},V\rangle - \inf_{p\mathcal{\in U}}\langle p,V\rangle \geq 0.
\end{aligned} \tag{A-20}$$

最后一步使用$p^{\circ} \in \mathcal{U}$。证毕。

在状态动作矩形集合下，将各处的$\mathcal{U}$取为$\mathcal{U}_{s,a}$，上述恒等式逐点进入鲁棒贝尔曼更新。相应正则项依赖后继价值和允许的分布集合；只有当该支持函数能够有效计算时，改写才同时带来计算上的便利。Derman等在指定集合条件下进一步建立了奖励、转移鲁棒性与策略、价值正则化之间的对应。[derman2023twice]

设有限状态空间配备距离$d_{\mathcal{S}}$。一阶Wasserstein距离$W_{1}(p,q)$为所有边缘分别为$p,q$的耦合分布中，状态距离$d_{\mathcal{S}}(x,y)$的最小期望。设$V$满足$|V(x) - V(y)| \leq L_{V}d_{\mathcal{S}}(x,y)$，$L_{V}$为其Lipschitz常数上界，$\varepsilon_{ipm} \geq 0$为分布距离半径。

**命题A.3。** 对$\mathcal{U = \{}p:W_{1}(p,p^{\circ}) \leq \varepsilon_{ipm}\}$，有

$$\inf_{p\mathcal{\in U}}\mathbb{E}_{p}\lbrack V\rbrack \geq \mathbb{E}_{p^{\circ}}\lbrack V\rbrack - \varepsilon_{ipm}L_{V}. \tag{A-21}$$

**证明。** 对任意边缘为$p,p^{\circ}$的耦合$\chi$，有

$$\left| \mathbb{E}_{p}\lbrack V\rbrack - \mathbb{E}_{p^{\circ}}\lbrack V\rbrack \right| = \left| \mathbb{E}_{(x,y) \sim \chi}\lbrack V(x) - V(y)\rbrack \right| \leq L_{V}\mathbb{E}_{\chi}\lbrack d_{\mathcal{S}}(x,y)\rbrack. \tag{A-22}$$

由前式的绝对值界，目标分布下的价值期望不小于名义期望减去距离成本与价值敏感度的乘积。先对耦合取下确界得到Wasserstein距离，再使用给定的距离半径界，再对集合内的$p$取下确界，即得式(A-21)。证毕。

这个不等式给出了名义价值减去敏感度惩罚的理论依据，等号还要求允许的分布移动能够达到相应最不利变化。对采用1-Lipschitz激活的前馈网络，其输入敏感度可由相容诱导矩阵范数的乘积$\prod_{l}^{} \parallel W_{l} \parallel$上界。RE-SAC采用的逐层元素绝对值之和$\sum_{l}^{} \parallel W_{l} \parallel_{1}$是另一种参数量，不能直接替换这个乘积。例如，两层标量权重均为5、正半轴上激活为恒等映射时，函数斜率为25，而两层权重绝对值之和为10。因此，第3章将该范数和用于固定目标偏移，算法性质按实际定义分析；命题A.3本身不赋予任意范数和一个精确鲁棒下界的含义。

以$C_{traj}$表示一个运行轨迹的有界可测累计成本，$\mathbb{P}_{0}$为固定策略和名义环境诱导的轨迹概率分布。取$\beta_{tail} \in (0,1\rbrack$表示关注的最差尾部概率质量，$\eta_{risk}$为成本阈值，$(x)_{+} = max(x,0)$。按这一尾部概率约定，定义

$${CVaR}_{\beta_{tail}}(C_{traj}) = \inf_{\eta_{risk}\mathbb{\in R}}\left\{ \eta_{risk} + \frac{1}{\beta_{tail}}\mathbb{E}_{\mathbb{P}_{0}}\lbrack(C_{traj} - \eta_{risk})_{+}\rbrack \right\}. \tag{A-23}$$

令$\zeta_{risk}$为可测的非负轨迹重加权密度，定义风险包络

$$\mathcal{Z}_{\beta_{tail}} = \left\{ \zeta_{risk}:\ 0 \leq \zeta_{risk} \leq \frac{1}{\beta_{tail}},\quad\mathbb{E}_{\mathbb{P}_{0}}\lbrack\zeta_{risk}\rbrack = 1 \right\}. \tag{A-24}$$

**命题A.4。** 在上述定义下，

$${CVaR}_{\beta_{tail}}(C_{traj}) = \sup_{\zeta_{risk} \in \mathcal{Z}_{\beta_{tail}}}\mathbb{E}_{\mathbb{P}_{0}}\lbrack\zeta_{risk}C_{traj}\rbrack. \tag{A-25}$$

**证明。** 对任意可行$\zeta_{risk}$和阈值$\eta_{risk}$，非负密度在阈值以下产生非正项，在阈值以上至多等于$1/\beta_{tail}$，故

$$\begin{aligned}
\mathbb{E}_{\mathbb{P}_{0}}\lbrack\zeta_{risk}C_{traj}\rbrack & = \eta_{risk} + \mathbb{E}_{\mathbb{P}_{0}}\lbrack\zeta_{risk}(C_{traj} - \eta_{risk})\rbrack \\
 & \leq \eta_{risk} + \frac{1}{\beta_{tail}}\mathbb{E}_{\mathbb{P}_{0}}\lbrack(C_{traj} - \eta_{risk})_{+}\rbrack.
\end{aligned} \tag{A-26}$$

尾部概率小于1时，累计成本的分布函数单调且右连续，在两端分别趋于0和1，因而存在一个分位阈值$\eta_{risk}^{*}$，使严格超过它的概率不大于$\beta_{tail}$，大于等于它的概率不小于$\beta_{tail}$。在严格超过阈值的轨迹上令$\zeta_{risk} = 1/\beta_{tail}$，在低于阈值的轨迹上令$\zeta_{risk} = 0$，并在等于阈值处选择区间$\lbrack 0,1/\beta_{tail}\rbrack$内的常数补足期望1。该密度可测且可行；密度有界、成本有界，故相应乘积可积，并使上式取得等号。若阈值处无概率质量，严格上尾概率即为$\beta_{tail}$，不需要补足。因此上、下两个优化问题具有相同值。若尾部概率等于1，所有可行密度均在几乎处处等于1；取不大于成本下界的阈值时，原优化目标也等于期望成本。因此该端点的风险包络表示同样成立。证毕。

由于每个可行密度都定义一个轨迹分布，命题A.4把尾部成本评价写成指定分布集合内的最坏期望。CVaR决策研究据此连接风险敏感目标与有预算的模型扰动。[chow2015risk] 这里的约束施加于完整轨迹概率；若转成逐阶段递推，还需保留风险预算的演化，不能任意改成各阶段独立使用同一尾部水平的更新。本文的跨种子尾部统计用于描述学习结果，RE-SAC训练仍采用第3章给出的保守目标，而非直接优化这一轨迹CVaR。

**A.8 适用范围。**本附录的价值算子结论针对有限状态动作、精确期望及固定参数；第A.7节的风险包络表示则适用于一般概率空间上的有界可测累计成本。RE-SAC实际使用函数逼近、随机小批量、目标网络及策略更新；网络更新使惩罚、策略或温度发生变化时，训练对应一列不同算子，其收敛还取决于逼近误差与算子漂移。权重范数作为固定标量进入证明，也不赋予它识别环境随机方差的统计含义。

软聚合证明采用有限动作集。公交连续动作策略的参数化采样与优化不由这一证明直接覆盖。第3章的理论说明固定惩罚对理想价值算子的作用，实际运行效果由公交实验判断。

第A.7节说明三类评价的联系及条件，不把任意熵项、权重惩罚或风险指标视为同一个目标。特别地，固定目标范数的收缩证明、Wasserstein敏感度下界和CVaR轨迹重加权分别有不同假设；这些假设决定了理论解释能够覆盖的算法部分。

# 附录B BAPR动作补偿与模式误判的数学证明

本附录对应 BAPR 的持久执行器模式与策略重参数化机制。证明围绕一个核心问题展开：若不同环境模式仅改变动作坐标，是否能够保留同一参考模式下的策略，通过坐标补偿恢复参考模式的执行；当模式识别不完全正确时，动作误差又如何进入长期回报差异。

论证依次包括有界动作的补偿可行性、极性变换下的逐路径等价、已有信息与模式可辨识性、动作加权的模式误差，以及有限噪声模型中的性能差恒等式与条件上界。

**B.1 基本条件。**设动作维数为 $n_{a}$，命令空间为

$$\mathcal{A = \lbrack -}1,1\rbrack^{n_{a}}. \tag{B-1}$$

逐坐标截断映射记为 $clip$，其第 $j$ 个分量为

$$clip(a)_{j} = {clip}_{1}(a_{j}),\quad 其中\quad{clip}_{1}(x) = max\{ - 1,min\{ x,1\}\}. \tag{B-2}$$

模式$z$的对角执行器增益为$G_{z} = diag(g_{z,1},\ldots,g_{z,n_{a}})$，其对角元素按动作坐标编号。名义信号与加噪后的实际执行信号均为无量纲执行器量，不表示公交驻站秒数。命令$a$先形成名义信号$v_{nom} = G_{z}a$，再与执行噪声$\varepsilon$相加并截断，得到实际执行量

$$u_{exec} = clip(v_{nom} + \varepsilon). \tag{B-3}$$

环境状态转移和奖励分别由 $F_{phys}(x_{phys},u_{exec},e)$ 与 $r(x_{phys},u_{exec},e)$ 给出，其中 $e$ 为其他外生事件。不同模式共用 $F_{phys}$ 和 $r$，模式差异由 $G_{z}$ 表达。状态映射和奖励映射在给定外生事件后确定相应后果，观测映射只提供参考模式下的策略可获得的输入。这里$x_{phys}$表示物理状态，$F_{phys}$表示物理状态转移映射。

设观测映射为 $o(x_{phys})$，参考模式下的策略为 $\mu_{ref}$，参考模式记为$ref$。参考模式下的动作记为

$$a^{ref}(x_{phys}) = \mu_{ref}(o(x_{phys}\mathcal{)) \in A.} \tag{B-4}$$

在参考模式直接执行时，名义信号为 $G_{ref}a^{ref}(x_{phys})$。在另一模式 $z$ 下，希望通过选择命令 $a_{z}(x_{phys})$，使

$$G_{z}a_{z}(x_{phys}) = G_{ref}a^{ref}(x_{phys}). \tag{B-5}$$

这一等式只描述名义信号恢复。若补偿命令越出动作边界，环境实际接收的截断命令可能不再满足它，因此需要先讨论可行性。

**B.2 补偿条件。****引理B.1。** 对任意实数 $x$，有

$${clip}_{1}(x) = x\quad \Leftrightarrow \quad - 1 \leq x \leq 1. \tag{B-6}$$

**证明。**截断的值域为$\lbrack - 1,1\rbrack$，且在该区间内保持输入不变，两个方向均由定义直接得到。证毕。

逐坐标应用该引理，立即得到

$$clip(a) = a\quad \Leftrightarrow \quad a\mathcal{\in A.} \tag{B-7}$$

**定理B.1。** 设 $G_{act} = diag(g_{1},\ldots,g_{n_{a}})$，每个 $g_{j} \neq 0$，希望实现名义目标信号 ${v_{nom}}^{*}$。令逆命令为

$$a_{inv} = {G_{act}}^{- 1}{v_{nom}}^{*},\quad\quad a_{inv,j} = v_{nom,j}^{*}/g_{j}. \tag{B-8}$$

则

$$G_{act}clip(a_{inv}) = {v_{nom}}^{*}\quad \Leftrightarrow \quad a_{inv}\mathcal{\in A.} \tag{B-9}$$

**证明。** 因 $g_{j} \neq 0$，逐坐标有 $g_{j}(v_{nom,j}^{*}/g_{j}) = v_{nom,j}^{*}$，即 $G_{act}a_{inv} = {v_{nom}}^{*}$。若 $G_{act}clip(a_{inv}) = {v_{nom}}^{*}$，则 $G_{act}clip(a_{inv}) = G_{act}a_{inv}$，从而

$$g_{j}\left( {clip}_{1}(a_{inv,j}) - a_{inv,j} \right) = 0 \tag{B-10}$$

对每个 $j$ 成立。非零增益意味着 ${clip}_{1}(a_{inv,j}) = a_{inv,j}$，由引理B.1得到 $a_{inv,j} \in \lbrack - 1,1\rbrack$，即 $a_{inv} \in \mathcal{A}$。

反之，若 $a_{inv} \in \mathcal{A}$，截断不改变 $a_{inv}$，所以 $G_{act}clip(a_{inv}) = G_{act}a_{inv} = {v_{nom}}^{*}$。证毕。

代入 ${v_{nom}}^{*} = G_{ref}a^{ref}$，精确补偿条件为

$$G_{z}^{- 1}G_{ref}a^{ref} \in \lbrack - 1,1\rbrack^{n_{a}}. \tag{B-11}$$

可逆性与可行性在这里是两个不同要求。例如单维增益 $g_{z} = 1/2$、参考增益为1、参考模式下的动作 $a^{ref} = 1$，逆命令为2。截断后命令为1，实际名义信号为 $1/2$，无法恢复目标信号1。这是定理的直接数值例子。

将对角元素只取 $+ 1$ 或 $- 1$ 的增益称为极性变换。对任意这样的 $G_{z}$，以$I_{n_{a}}$表示动作维数对应的单位矩阵，有

$$G_{z}^{2} = I_{n_{a}},\quad\quad G_{z}^{- 1} = G_{z}. \tag{B-12}$$

**定理B.2。** 若当前模式 $G_{z}$ 与参考模式 $G_{ref}$ 均为极性变换，且 $a^{ref} \in \mathcal{A}$，则补偿命令

$$a_{z} = G_{z}G_{ref}a^{ref} \tag{B-13}$$

仍在 $\mathcal{A}$ 中，并满足

$$G_{z}a_{z} = G_{ref}a^{ref}. \tag{B-14}$$

**证明。** 每个分量为 $a_{z,j} = g_{z,j}g_{ref,j}{a^{ref}}_{j}$。两个符号的乘积为 $+ 1$ 或 $- 1$，所以 $|a_{z,j}| = |{a^{ref}}_{j}| \leq 1$，即补偿命令不越界。另一方面，

$$g_{z,j}a_{z,j} = g_{z,j}^{2}g_{ref,j}{a^{ref}}_{j} = g_{ref,j}{a^{ref}}_{j}. \tag{B-15}$$

逐坐标合并得到信号等式。证毕。

极性结构的作用因此有两层：自逆性提供代数补偿，保持逐维动作上下界提供执行可行性。二者共同使正确模式下的补偿无需额外近似。

**B.3 补偿效果。****引理B.2。** 若两个名义信号 $v_{nom,1} = v_{nom,2}$，且使用同一执行噪声 $\varepsilon$，则

$$clip(v_{nom,1} + \varepsilon) = clip(v_{nom,2} + \varepsilon). \tag{B-16}$$

**证明。** 两个截断输入相同，函数作用结果也相同。证毕。

结合定理B.2，正确极性补偿与参考模式不仅在名义信号上相同，加上共同噪声并截断后的实际执行量也相同。

固定执行噪声序列 $\{\varepsilon_{k}\}$ 与外生事件序列 $\{ e_{k}\}$。给定名义信号规则 $v_{nom}(k,x_{phys})$，从状态 $x_{phys,k}$ 出发定义

$$u_{exec,k} = clip(v_{nom}(k,x_{phys,k}) + \varepsilon_{k}),\quad\quad x_{phys,k + 1} = F_{phys}(x_{phys,k},u_{exec,k},e_{k}). \tag{B-17}$$

剩余 $n$ 步的累计奖励为

$$J_{n,k}^{v_{nom}}(x_{phys,k}) = \sum_{i = 0}^{n - 1}r(x_{phys,k + i},u_{exec,k + i},e_{k + i}),\quad\quad J_{0,k}^{v_{nom}} = 0. \tag{B-18}$$

下面在有限时域内进一步说明，相同的名义信号规则如何使两条轨迹的状态与累计奖励保持一致。

**引理B.3。** 若两个信号规则对所有 $k,x_{phys}$ 满足 $v_{nom,1}(k,x_{phys}) = v_{nom,2}(k,x_{phys})$，则从同一初始状态、使用同一噪声与事件序列运行任意有限步数后，末状态及累计奖励相同。

**证明。**零步结论成立；若当前状态相同，相同名义信号和共同外生量给出相同执行量、奖励与后继状态。逐步归纳，末状态与累计奖励均相同。证毕。

**定理B.3。** 设 $z_{k}$ 为任意给定模式序列，所有 $G_{z}$ 均为极性变换。比较以下两个控制过程：参考过程始终产生名义信号 $G_{ref}a^{ref}(x_{phys,k})$；切换过程在每步正确获知 $z_{k}$，执行命令 $G_{z_{k}}G_{ref}a^{ref}(x_{phys,k})$。若初始状态及外生噪声相同，则两过程在任意有限时域内逐路径具有相同状态、执行量、奖励和累计回报。

**证明。** 对任意时间与状态，切换过程的名义信号为

$$G_{z_{k}}\left( G_{z_{k}}G_{ref}a^{ref}(x_{phys}) \right) = G_{ref}a^{ref}(x_{phys}), \tag{B-19}$$

与参考过程相同。由引理B.3，任意有限步数的末状态及回报相同；由于该结论对每个前缀长度都成立，整个状态序列逐项相同。执行量与奖励的一致性随之由共同状态、共同信号和共同外生量得到。证毕。

若再对同一外生噪声分布取平均，逐路径相等直接推出回报分布及期望相等。上述代数恢复不要求模式切换之间有最小间隔；模式持续性主要影响基于已完成转移的模式识别能否及时提供正确模式。

**B.4 模式识别。**本附录中的“因果”仅指动作使用当前及历史信息，而不使用未来观测；它不表示已经识别环境的真实因果结构。下文据此称为基于历史信息的控制器。

设初始置信为 $\rho_{0}$，转移记录为 $\xi_{0},\xi_{1},\ldots$，更新映射为 $U$。递归更新映射由已完成转移和旧置信形成新置信，动作选择映射再由当前置信及状态形成命令。第 $k$ 个动作仅使用前 $k$ 个已完成转移：

$$\rho_{k} = U(\cdots U(U(\rho_{0},\xi_{0}),\xi_{1})\cdots,\xi_{k - 1}),\quad 其中\quad a_{k} = S(\rho_{k},s_{k}). \tag{B-20}$$

**命题B.1。** 若两条记录流的前 $k$ 个转移完全相同，当前状态也相同，则由上述规则生成的第 $k$ 个动作相同，无论后续记录如何变化。

**证明。**相同记录前缀依次产生相同置信，再由相同选择映射得到相同动作。证毕。

这一性质给出控制器的信息时序：产生 $a_{k}$ 时尚未使用由 $a_{k}$ 形成的后继转移。它与模式识别精度是不同命题，按上述信息时序执行本身不意味着一定能识别模式。

**命题B.2。** 给定相同的命令规则 $a(k,x_{phys})$，若两条模式序列对所有 $k,x_{phys}$ 满足

$$G_{z_{k}}a(k,x_{phys}) = G_{z_{k}'}a(k,x_{phys}), \tag{B-21}$$

则在共同初始状态和共同外生噪声下，两个系统的有限时域末状态与累计奖励相同。

**证明。** 两系统对每个时间、状态提供相同名义信号，直接应用引理B.3即可。证毕。

例如两个极性模式只在某一坐标的符号上不同，而命令始终将该坐标置零，则执行信号无法暴露这一符号差异。已知变换可逆并不能消除缺少激励导致的信息不足。

设有限噪声样本空间为 $\Omega$，质量函数 $p(\omega) \geq 0$ 且 $\sum_{\omega}^{}p(\omega) = 1$。两模式对应观察 $Y_{1}(\omega)$、$Y_{2}(\omega)$，且对每个 $\omega$ 都相同。令判别器 $D_{cls}(Y) \in \{ 1,2\}$。两个观测映射分别给出两种模式下的观测，判别器根据观测输出模式标签。

**定理B.4。** 在两模式先验概率均为 $1/2$ 时，判别器平均正确率为 $1/2$。

**证明。** 正确率可写为

$$\begin{aligned}
p_{correct} = & \frac{1}{2}\sum_{\omega}^{}p(\omega)\mathbf{1}\{ D_{cls}(Y_{1}(\omega)) = 1\} \\
 & + \frac{1}{2}\sum_{\omega}^{}p(\omega)\mathbf{1}\{ D_{cls}(Y_{2}(\omega)) = 2\}.
\end{aligned} \tag{B-22}$$

由观察相同，两个指标在每个 $\omega$ 上互补，相加为1。于是

$$p_{correct} = \frac{1}{2}\sum_{\omega}^{}p(\omega) = \frac{1}{2}. \tag{B-23}$$

证毕。该结论适用于确定性二元判别器；若判别器使用独立有限随机种子，可将种子并入共同样本空间得到相同结论。

设历史变量为 $H_{obs}$，下一观察为 $Y$，二者取值集合有限。两种模式假设下，历史分布分别为 $p(h_{obs})$、$q(h_{obs})$，下一观察条件核为 $P_{obs,1}(y \mid h_{obs})$、$P_{obs,2}(y \mid h_{obs})$。本节假设这些概率严格为正，并按各自分布归一化。历史的实现值和下一观察的实现值分别是本节有限求和的局部自变量；两个条件核分别对应两种模式假设。

定义有限分布的Kullback-Leibler散度（Kullback-Leibler Divergence，KL散度），亦称相对熵

$$D_{KL}(p \parallel q) = \sum_{h_{obs}}^{}p(h_{obs})log\frac{p(h_{obs})}{q(h_{obs})}. \tag{B-24}$$

**定理B.5。** 联合分布满足

$$\begin{array}{r}
D_{KL}(pP_{obs,1} \parallel qP_{obs,2}) = D_{KL}(p \parallel q) \\
 + \sum_{h_{obs}}^{}p(h_{obs}) \\
 \times D_{KL}\left( P_{obs,1}( \cdot \mid h_{obs}) \parallel P_{obs,2}( \cdot \mid h_{obs}) \right).
\end{array} \tag{B-25}$$

**证明。** 严格正性保证对数的商与乘积展开有效。对每个 $(h_{obs},y)$，

$$\log\frac{p(h_{obs})P_{obs,1}(y \mid h_{obs})}{q(h_{obs})P_{obs,2}(y \mid h_{obs})} = log\frac{p(h_{obs})}{q(h_{obs})} + log\frac{P_{obs,1}(y \mid h_{obs})}{P_{obs,2}(y \mid h_{obs})}. \tag{B-26}$$

乘以 $p(h_{obs})P_{obs,1}(y \mid h_{obs})$ 后对 $y$ 求和，第一项变为

$$p(h_{obs})log\frac{p(h_{obs})}{q(h_{obs})}\sum_{y}^{}P_{obs,1}(y \mid h_{obs}) = p(h_{obs})log\frac{p(h_{obs})}{q(h_{obs})}. \tag{B-27}$$

第二项为

$$p(h_{obs})\sum_{y}^{}P_{obs,1}(y \mid h_{obs})log\frac{P_{obs,1}(y \mid h_{obs})}{P_{obs,2}(y \mid h_{obs})}. \tag{B-28}$$

最后对 $h_{obs}$ 求和即得结论。证毕。

在固定模式驻留区间内逐次应用该等式，并使用共同初始历史分布，可将轨迹相对熵展开为条件相对熵的期望和：

$$D_{KL}(\mathsf{P}_{z}^{\pi_{beh}} \parallel \mathsf{P}_{z'}^{\pi_{beh}}) = \sum_{k = 0}^{N_{hor} - 1}\mathbb{E}_{\mathsf{P}_{z}^{\pi_{beh}}}\left\lbrack D_{KL}(P_{z}( \cdot \mid H_{k},a_{k}) \parallel P_{z'}( \cdot \mid H_{k},a_{k})) \right\rbrack. \tag{B-29}$$

这里 $\pi_{beh}$ 是两个假设下使用的同一基于已有信息的行为规则，$H_{k}$ 为已获得历史。逐步的共同动作选择在给定历史后不额外提供模式标签。若访问到的条件观察核相同，则各项为零，轨迹不能提供区分这两种模式的信息。正信息量说明有可区分信号，但其大小与有限驻留时间内能否可靠识别仍是不同问题。

**B.5 动作误差。**设真实模式为 $z$，用于补偿的估计模式为 $\widehat{z}$。基于历史信息的控制器实际名义信号与参考信号分别为

$${v_{nom}}^{C} = G_{z}G_{\widehat{z}}G_{ref}a^{ref},\quad\quad{v_{nom}}^{O} = G_{ref}a^{ref}. \tag{B-30}$$

二者之差记为 $\Delta v_{nom} = {v_{nom}}^{C} - {v_{nom}}^{O}$，其第 $j$ 个分量为

$$\Delta v_{nom,j} = (g_{z,j}g_{\widehat{z},j} - 1)g_{ref,j}{a^{ref}}_{j}. \tag{B-31}$$

该表达式比较同一个状态下的两个信号，尚未将未来状态变化包含在内。

**定理B.6。** 若真实、估计及参考增益均为极性变换，则

$$(\Delta v_{nom,j})^{2} = \left\{ \begin{matrix}
0, & g_{z,j} = g_{\widehat{z},j}, \\
4{a^{ref}}_{j}^{2}, & g_{z,j} \neq g_{\widehat{z},j},
\end{matrix} \right.\  \tag{B-32}$$

从而

$$\parallel \Delta v_{nom} \parallel_{2}^{2} = 4\sum_{j = 1}^{n_{a}}{a^{ref}}_{j}^{2}\mathbf{1}\{ g_{z,j} \neq g_{\widehat{z},j}\}, \tag{B-33}$$

以及

$$\parallel \Delta v_{nom} \parallel_{2} = 2\sqrt{\sum_{j = 1}^{n_{a}}{a^{ref}}_{j}^{2}\mathbf{1}\{ g_{z,j} \neq g_{\widehat{z},j}\}}. \tag{B-34}$$

**证明。** 若两模式在第 $j$ 维符号相同，则乘积为1，差分因子为0。若符号不同，则乘积为 $- 1$，差分因子为 $- 2$。参考符号平方为1，所以第二种情形的误差平方为 $( - 2)^{2}g_{ref,j}^{2}{a^{ref}}_{j}^{2} = 4{a^{ref}}_{j}^{2}$。对坐标求和得到平方范数，再利用非负平方根及 $\sqrt{4x} = 2\sqrt{x}$ 得到范数表达式。证毕。

当误判发生在参考模式下的动作接近零的坐标时，名义信号误差较小；同样的模式误判发生在大幅动作坐标时，误差更大。因此未加权模式准确率不能完整描述控制误差。下一节进一步说明，长期回报还取决于这些局部动作差异所处状态的价值敏感性。

**B.6 路径分析。**为将含记忆的控制规则表示为当前状态的函数，自本节起用$x_{aug}$表示增广状态。它包含物理状态、已完成的观测与命令历史$H_{k}$以及置信$\rho_{k}$；真实执行模式不能由时刻与物理状态确定时，也作为分析用隐变量纳入，但控制器不读取该隐变量。第B.3节比较的是物理轨迹，本节的增广还需保留控制器用于后续判断的信息。

名义信号$v_{nom}$在真实模式$z$下对应命令$a = G_{z}^{- 1}v_{nom}$，实际执行信号则为$clip(v_{nom} + \varepsilon_{k})$。本节只考虑满足命令边界的名义信号；极性补偿自动满足这一条件，一般增益须满足式(B-11)。一次转移完成后，以实际观测、所发命令和对应反馈构造记录$\xi_{k}$，更新$H_{k + 1} = H_{k} \frown \xi_{k}$及$\rho_{k + 1} = U(\rho_{k},\xi_{k})$，其中$\frown$表示追加记录。

将物理转移、命令记录、置信更新及模式演化共同记为$F_{aug}(x_{aug},v_{nom},e_{k})$。本节固定的外生序列提供$e_{k}$与执行噪声$\varepsilon_{k}$。这一映射的动作输入是加噪和截断前的名义信号，而不是截断后的执行信号：截断可能使不同命令产生相同执行结果，仅保留执行结果不足以重建命令历史。映射的物理状态分量仍按$F_{phys}(x_{phys},clip(v_{nom} + \varepsilon_{k}),e_{k})$更新，奖励仍由物理状态与实际执行信号计算。

固定同一外生序列，在相同真实模式演化和信息记录规则下，比较待评价控制器的名义信号$v_{nom}^{C}(k,x_{aug})$与参照规则$v_{nom}^{O}(k,x_{aug})$。参照可以取正确模式补偿后的参考信号，不要求它是全局最优策略。以$V_{n,k}^{O,path}$表示按参照规则继续执行$n$步的路径回报；先执行一个可行名义信号，再执行参照规则$n$步，定义

$$\begin{aligned}
Q_{n,k}^{O,path}(x_{aug},v_{nom}) = & r(x_{phys},clip(v_{nom} + \varepsilon_{k}),e_{k}) \\
 & + V_{n,k + 1}^{O,path}\left( F_{aug}(x_{aug},v_{nom},e_{k}) \right).
\end{aligned} \tag{B-35}$$

其中$x_{phys}$是$x_{aug}$的物理分量。动作价值下标$n$只计首步之后执行参照规则的步数，因此$V_{n + 1,k}^{O,path}$对应$Q_{n,k}^{O,path}$。

定义当前信号相对参照信号的路径动作价值差为

$$\begin{aligned}
\Delta_{n,k}^{O}(x_{aug}) = & Q_{n,k}^{O,path}(x_{aug},{v_{nom}}^{C}(k,x_{aug})) \\
 & - Q_{n,k}^{O,path}(x_{aug},{v_{nom}}^{O}(k,x_{aug})).
\end{aligned} \tag{B-36}$$

**定理B.7。** 从同一 $x_{aug,k}$ 出发，令 $x_{aug,k + i}^{C}$ 为基于历史信息的控制器形成的状态。运行 $N$ 步时，

$$J_{N,k}^{C}(x_{aug,k}) - J_{N,k}^{O}(x_{aug,k}) = \sum_{i = 0}^{N - 1}\Delta_{N - 1 - i,k + i}^{O}(x_{aug,k + i}^{C}). \tag{B-37}$$

**证明。** 记 ${\Delta J}_{n,k}(x_{aug}) = J_{n,k}^{C}(x_{aug}) - J_{n,k}^{O}(x_{aug})$。零步时为0。考虑 $n + 1$ 步，并令

$$x_{aug,C}' = F_{aug}\left( x_{aug},v_{nom}^{C}(k,x_{aug}),e_{k} \right). \tag{B-38}$$

展开两个路径回报，再在待评价控制器分支中加入并减去 $V_{n,k + 1}^{O,path}(x_{aug,C}')$，得到

$$\begin{aligned}
{\Delta J}_{n + 1,k}(x_{aug}) = & Q_{n,k}^{O,path}(x_{aug},{v_{nom}}^{C}(k,x_{aug})) - Q_{n,k}^{O,path}(x_{aug},{v_{nom}}^{O}(k,x_{aug})) \\
 & + {\Delta J}_{n,k + 1}(x_{aug,C}').
\end{aligned} \tag{B-39}$$

逐步对最后一项展开，直到剩余步数为0，全部项都在待评价控制器轨迹访问的状态上计算，得到所述求和。证毕。

该恒等式不要求两种控制器第一步之后仍处于相同状态。每一项都在待评价控制器实际访问的状态上，比较先执行两种名义信号、再按参照规则运行的回报。因此，该分解已考虑动作不同所引起的后续状态差异。

**推论B.1。** 若存在局部界 $b_{Q,loc}(k,x_{aug})$，使任意相关剩余步数满足

$$|Q_{n,k}^{O,path}(x_{aug},{v_{nom}}^{C}(k,x_{aug})) - Q_{n,k}^{O,path}(x_{aug},{v_{nom}}^{O}(k,x_{aug}))| \leq b_{Q,loc}(k,x_{aug}), \tag{B-40}$$

则

$$|J_{N,k}^{C}(x_{aug,k}) - J_{N,k}^{O}(x_{aug,k})| \leq \sum_{i = 0}^{N - 1}b_{Q,loc}(k + i,x_{aug,k + i}^{C}). \tag{B-41}$$

**证明。** 对定理B.7右侧应用有限和的三角不等式，再逐项应用局部界即可。也可以按剩余步数归纳得到同一累计界。证毕。

**B.7 期望分析。**设外生事件集合 $\mathcal{E}$ 有限，质量函数满足 $p(e) \geq 0$、$\sum_{e}^{}p(e) = 1$，每步独立按该质量函数抽取事件。事件同时决定执行噪声 $\varepsilon(e)$、奖励及转移中的外生量。

沿用第B.6节以名义信号为动作输入的增广转移。期望分析中，外生事件$e$同时确定执行噪声$\varepsilon(e)$及其余随机量，定义

$$\begin{aligned}
F_{dyn,e}(x_{aug},v_{nom}) & = F_{aug}(x_{aug},v_{nom},e), \\
r_{e}(x_{aug},v_{nom}) & = r(x_{phys},clip(v_{nom} + \varepsilon(e)),e).
\end{aligned} \tag{B-42}$$

上述转移同时更新物理状态与命令历史，奖励由物理状态及实际执行信号确定。以下价值递推和性能差分析均基于这两个定义。任意名义信号规则 ${v_{nom}}^{\pi}(k,x_{aug})$ 的期望价值递归为

$$V_{0,k}^{\pi}(x_{aug}) = 0, \tag{B-43}$$

$$V_{n + 1,k}^{\pi}(x_{aug}) = \sum_{e}^{}p(e)\left\lbrack r_{e}(x_{aug},{v_{nom}}^{\pi}(k,x_{aug})) + V_{n,k + 1}^{\pi}(F_{dyn,e}(x_{aug},{v_{nom}}^{\pi}(k,x_{aug}))) \right\rbrack. \tag{B-44}$$

先执行给定名义信号、此后按参照规则继续运行的期望动作价值为

$$Q_{n,k}^{O}(x_{aug},v_{nom}) = \sum_{e}^{}p(e)\left\lbrack r_{e}(x_{aug},v_{nom}) + V_{n,k + 1}^{O}(F_{dyn,e}(x_{aug},v_{nom})) \right\rbrack. \tag{B-45}$$

本节的 $V,Q$ 为对外生事件分布求平均的价值量；B.6带path上标的价值量则固定单条外生序列。状态 $x_{aug}$ 可以包含实现控制所需的历史或置信，使因果选择写成时间和当前增广状态的函数。

令

$$A_{n,k}^{O}(x_{aug}) = Q_{n,k}^{O}(x_{aug},{v_{nom}}^{C}(k,x_{aug})) - Q_{n,k}^{O}(x_{aug},{v_{nom}}^{O}(k,x_{aug})). \tag{B-46}$$

**定理B.8。** 对任意有限时域 $N$，有

$$V_{N,k}^{C}(x_{aug}) - V_{N,k}^{O}(x_{aug}) = \mathbb{E}_{C}\left\lbrack \sum_{i = 0}^{N - 1}A_{N - 1 - i,k + i}^{O}(X_{aug,k + i})\  \middle| \ X_{aug,k} = x_{aug} \right\rbrack. \tag{B-47}$$

其中期望沿基于历史信息的控制器 $C$ 诱导的状态分布计算。

**证明。** 定义 ${\Delta V}_{n,k}(x_{aug}) = V_{n,k}^{C}(x_{aug}) - V_{n,k}^{O}(x_{aug})$。按价值递归展开，并在每个待评价控制器的后继状态加入再减去参照价值，得到

$$\begin{aligned}
{\Delta V}_{n + 1,k}(x_{aug}) = & A_{n,k}^{O}(x_{aug}) \\
 & + \sum_{e}^{}p(e){\Delta V}_{n,k + 1}(F_{dyn,e}(x_{aug},{v_{nom}}^{C}(k,x_{aug}))).
\end{aligned} \tag{B-48}$$

零步差为0。假设剩余 $n$ 步的差已经等于相应局部差累计期望，则最后一项是在第一步待评价控制器转移后，对后续累计差再次取期望。有限求和的分配律将其与当前局部差合并，得到 $n + 1$ 步结论。对 $n$ 归纳即得定理。证毕。

该等式不需要价值函数对动作光滑，也不需要模式识别准确率假设。它首先给出精确分解，再将如何界定单步差的问题留给下一步。

**定理B.9。** 若对所有相关剩余步数、时间和状态都有

$$|A_{n,k}^{O}(x_{aug})| \leq b_{Q,loc}(k,x_{aug}), \tag{B-49}$$

则

$$|V_{N,k}^{C}(x_{aug}) - V_{N,k}^{O}(x_{aug})| \leq \mathbb{E}_{C}\left\lbrack \sum_{i = 0}^{N - 1}b_{Q,loc}(k + i,X_{aug,k + i})\  \middle| \ X_{aug,k} = x_{aug} \right\rbrack. \tag{B-50}$$

**证明。** 可直接对定理B.8应用期望与有限和的三角不等式。累计界从零步开始递归，局部界在基于历史信息的控制器访问的各状态上求和取期望。为给出递归形式，定义

$$B_{ret,0,k}(x_{aug}) = 0, \tag{B-51}$$

$$B_{ret,n + 1,k}(x_{aug}) = b_{Q,loc}(k,x_{aug}) + \sum_{e}^{}p(e)B_{ret,n,k + 1}(F_{dyn,e}(x_{aug},{v_{nom}}^{C}(k,x_{aug}))). \tag{B-52}$$

零步时 $|{\Delta V}_{0,k}| = B_{ret,0,k} = 0$。若 $|{\Delta V}_{n,k + 1}({x_{aug}}')| \leq B_{ret,n,k + 1}({x_{aug}}')$ 对所有后继状态成立，由前述差分递归有

$$\begin{aligned}
|{\Delta V}_{n + 1,k}(x_{aug})| & \leq |A_{n,k}^{O}(x_{aug})| + \left| \sum_{e}^{}p(e){\Delta V}_{n,k + 1}(F_{dyn,e}(x_{aug},{v_{nom}}^{C}(k,x_{aug}))) \right| \\
 & \leq b_{Q,loc}(k,x_{aug}) + \sum_{e}^{}p(e)|{\Delta V}_{n,k + 1}(F_{dyn,e}(x_{aug},{v_{nom}}^{C}(k,x_{aug})))| \\
 & \leq b_{Q,loc}(k,x_{aug}) + \sum_{e}^{}p(e)B_{ret,n,k + 1}(F_{dyn,e}(x_{aug},{v_{nom}}^{C}(k,x_{aug}))) \\
 & = B_{ret,n + 1,k}(x_{aug}).
\end{aligned} \tag{B-53}$$

第二步使用 $p(e) \geq 0$，第三步使用归纳假设。展开 $B_{ret}$ 就是待评价控制器的状态分布下的累计局部界期望，故结论成立。证毕。

**B.8 回报损失。**取参考名义信号

$${v_{nom}}^{O}(k,x_{aug}) = G_{ref}a^{ref}(x_{aug}), \tag{B-54}$$

待评价控制器信号为

$${v_{nom}}^{C}(k,x_{aug}) = G_{z_{k}}G_{\widehat{z}(k,x_{aug})}G_{ref}a^{ref}(x_{aug}). \tag{B-55}$$

设 $L_{Q,act}(k,x_{aug}) \geq 0$ 是给定的局部敏感性系数，并假设对所比较的两个名义信号及相关剩余步数，

$$|Q_{n,k}^{O}(x_{aug},{v_{nom}}^{C}(k,x_{aug})) - Q_{n,k}^{O}(x_{aug},{v_{nom}}^{O}(k,x_{aug}))| \leq L_{Q,act}(k,x_{aug}) \parallel {v_{nom}}^{C}(k,x_{aug}) - {v_{nom}}^{O}(k,x_{aug}) \parallel_{2}. \tag{B-56}$$

这个条件约束包含后续参照价值的动作价值差，而非只约束一步物理转移。

**定理B.10。** 在B.7的有限概率模型、极性变换与上述局部敏感性条件下，

$$\begin{aligned}
 & |V_{N,k}^{C}(x_{aug}) - V_{N,k}^{O}(x_{aug})| \\
 & \quad \leq \mathbb{E}_{C}\left\lbrack \sum_{i = 0}^{N - 1}2L_{Q,act}(k + i,X_{aug,k + i})\sqrt{\sum_{j = 1}^{n_{a}}{a^{ref}}_{j}(X_{aug,k + i})^{2}I_{i,j}}\  \middle| \ X_{aug,k} = x_{aug} \right\rbrack,
\end{aligned} \tag{B-57}$$

其中

$$I_{i,j} = \mathbf{1}\left\{ g_{z_{k + i},j} \neq g_{\widehat{z}(k + i,X_{aug,k + i}),j} \right\}. \tag{B-58}$$

**证明。** 定理B.6在每个时间与状态给出

$$\parallel {v_{nom}}^{C}(k,x_{aug}) - {v_{nom}}^{O}(k,x_{aug}) \parallel_{2} = 2\sqrt{\sum_{j}^{}{a^{ref}}_{j}(x_{aug})^{2}\mathbf{1}\{ g_{z_{k},j} \neq g_{\widehat{z}(k,x_{aug}),j}\}}. \tag{B-59}$$

将其代入局部敏感性条件，得到定理B.9所需的局部界

$$b_{Q,loc}(k,x_{aug}) = 2L_{Q,act}(k,x_{aug})\sqrt{\sum_{j}^{}{a^{ref}}_{j}(x_{aug})^{2}\mathbf{1}\{ g_{z_{k},j} \neq g_{\widehat{z}(k,x_{aug}),j}\}}. \tag{B-60}$$

再应用定理B.9即可。证毕。

该界把模式误判、参考模式下的动作幅度与局部价值敏感性联系起来。系数2和动作平方权重来自极性误差的精确代数，而 $L_{Q,act}(k,x_{aug})$ 仍是需要成立的外部敏感性条件。识别准确率即使较高，也不能单独替代这些状态与动作信息。

**B.9 适用范围。**本附录的精确恢复依赖不同模式共享状态转移与奖励机制，差异仅通过已知执行器坐标进入。一般可逆对角增益还需满足逆命令可行性，极性变换则自动保持动作边界。因此这些结论直接解释 BAPR 的执行器模式机制，不直接把客流增加或道路通行时间变化归结为同一种动作变换。

逐路径等价以正确模式为条件，基于历史信息的控制器则必须从已完成转移中识别模式。无激励和相同观察的结果说明已知变换并不保证可辨识；KL链式结果在有限且严格正的概率核下成立，所证明的单步等式也不直接给出有限驻留时长中的识别精度。

性能差恒等式本身精确成立，其符号没有被预先确定。正确模式参照恢复参考模式下的策略，并非任意动作序列上的全局最优上界。回报绝对差界进一步依赖局部动作价值敏感性。对于包含接触和约束切换的动力学，需要分别确定所考察区域内的敏感性常数；模式识别准确率本身不足以给出数值性能保证。

# 附录C BAPR置信加权算子的数学证明

BAPR研究围绕分段平稳环境中的置信更新与保守价值学习展开[zhang2026bapr]。其理论对象是固定置信加权的模式贝尔曼算子，与附录B的执行器坐标补偿具有不同机制。本附录给出置信加权价值学习的数学基础，为置信更新、联合模式表示及结构化适应方法提供明确的算子解释。

分析依次讨论固定权重混合、概率归一化及边缘分布，并进一步考察价值依赖权重对收缩性的影响，以及持续证据对模式置信的作用。

**C.1 基本定义。**$\mathcal{Z}$为候选模式集合，$z$为其元素；$\rho$为单模式置信，$\rho^{joint}$为联合模式置信，$\rho^{(2)}$为第二模式变量的边缘分布。${\overline{R}}_{pen,\rho}$表示按置信加权的固定奖励与惩罚项。本附录固定标量惩罚是给定的合并量，区别于第3章未加权的网络权重范数。

设状态集 $\mathcal{S}$、动作集 $\mathcal{A}$ 和模式集 $\mathcal{Z}$ 均有限非空。模式 $z$ 可以表示候选运行模式或距上次变化的持续长度。每个模式具有奖励 $R_{z}(s,a)$、转移核 $P_{z}(s' \mid s,a)$ 和固定惩罚 $\Gamma_{z}(s,a)$，满足

$$P_{z}(s' \mid s,a) \geq 0,\quad\quad\sum_{s'}^{}P_{z}(s' \mid s,a) = 1. \tag{C-1}$$

令 $0 \leq \gamma < 1$，$\lambda_{epi} \geq 0$，$\kappa_{fix}\mathbb{\in R}$ 为固定标量，动作最大值为

$$V_{Q}(s) = \max_{a}Q(s,a). \tag{C-2}$$

模式条件算子定义为

$$(\mathcal{T}_{z}Q)(s,a) = R_{z}(s,a) + \gamma\left\lbrack \sum_{s'}^{}P_{z}(s' \mid s,a)V_{Q}(s') - \lambda_{epi}\Gamma_{z}(s,a) - \kappa_{fix} \right\rbrack. \tag{C-3}$$

各模式共享输入价值函数 $Q$，但具有各自转移、奖励及惩罚。在比较同一算子的两个输入时，上述参数不改变。

设置信权重 $\rho(z)$ 满足

$$\rho(z) \geq 0,\quad\quad\sum_{z}^{}\rho(z) = 1. \tag{C-4}$$

定义置信加权算子

$$(\mathcal{T}_{\rho}Q)(s,a) = \sum_{z\mathcal{\in Z}}^{}\rho(z)(\mathcal{T}_{z}Q)(s,a). \tag{C-5}$$

$\rho$ 是算子的给定参数，在本次比较中不随输入 $Q$ 改变。其可以由前一阶段数据产生，也可以在后续阶段更新；当前定理先研究固定权重下的作用。

**C.2 等价表示。**固定置信混合可以合并为一个随机核和一个与价值输入无关的函数。定义

$$\begin{aligned}
{\overline{P}}_{\rho}(s' \mid s,a) & = \sum_{z}^{}\rho(z)P_{z}(s' \mid s,a), \\
{\overline{R}}_{pen,\rho}(s,a) & = \sum_{z}^{}\rho(z)\left\lbrack R_{z}(s,a) - \gamma\lambda_{epi}\Gamma_{z}(s,a) - \gamma\kappa_{fix} \right\rbrack.
\end{aligned} \tag{C-6}$$

**命题C.1。** ${\overline{P}}_{\rho}$ 为合法随机核，且置信加权算子可写为

$$(\mathcal{T}_{\rho}Q)(s,a) = {\overline{R}}_{pen,\rho}(s,a) + \gamma\sum_{s'}^{}{\overline{P}}_{\rho}(s' \mid s,a)V_{Q}(s'). \tag{C-7}$$

**证明。** 核的非负性来自非负加权和，归一化满足

$$\sum_{s'}^{}{\overline{P}}_{\rho}(s' \mid s,a) = \sum_{z}^{}\rho(z)\sum_{s'}^{}P_{z}(s' \mid s,a) = \sum_{z}^{}\rho(z) = 1. \tag{C-8}$$

将C.1中的模式算子展开并交换两个有限求和，即得所述表示。所有模式共享同一个 $V_{Q}$，因此可以把模式平均收入转移核。证毕。

**C.3 基本性质。****定理C.1。** 在C.1的固定置信与固定惩罚条件下，$\mathcal{T}_{\rho}$ 单调，并满足

$$\begin{aligned}
\mathcal{T}_{\rho}\left( Q + c_{shift} \right) = \mathcal{T}_{\rho}Q + \gamma c_{shift},\quad\quad \\
\parallel \mathcal{T}_{\rho}Q_{1} - \mathcal{T}_{\rho}Q_{2} \parallel_{\infty} \leq \gamma \parallel Q_{1} - Q_{2} \parallel_{\infty}.
\end{aligned} \tag{C-9}$$

**证明。** 命题C.1给出合法随机核 ${\overline{P}}_{\rho}$ 和固定项 ${\overline{R}}_{pen,\rho}$；最大值聚合满足引理A.1的条件。将三者代入定理A.1即可。单一模式 $\mathcal{T}_{z}$ 对应把置信集中于该模式，因而具有相同性质。由推论A.1，固定的 $\mathcal{T}_{\rho}$ 具有唯一固定点及精确迭代的几何误差界。证毕。

固定置信使模式平均成为当前模型的一部分。模式是否可辨识、平稳阶段持续多久，影响置信估计与随时间变化的外层过程，而不进入这一次固定算子的收缩系数。

若同时跟踪持续长度 $z \in \mathcal{Z}$ 与运行类别 $z^{(2)} \in \mathcal{Z}$，并给定合法联合置信 $\rho^{joint}(z,z^{(2)})$，则

$$\mathcal{T}_{\rho^{joint}}Q = \sum_{z,z^{(2)}}^{}\rho^{joint}(z,z^{(2)})\mathcal{T}_{z,z^{(2)}}Q \tag{C-10}$$

具有相同结论。其依据是有限乘积集 $\mathcal{Z} \times \mathcal{Z}$ 仍有限非空，可以作为命题C.1及定理C.1中的模式集。

**C.4 更新性质。**证据分数只要求非负；归一化常数是各模式证据分数的置信加权和。设原权重 $\rho(z) \geq 0$，证据分数 $L(z) \geq 0$，归一化常数为

$$Z_{norm} = \sum_{z}^{}\rho(z)L(z) > 0. \tag{C-11}$$

更新权重定义为

$$\rho'(z) = \frac{\rho(z)L(z)}{Z_{norm}}. \tag{C-12}$$

**命题C.2。** 更新权重非负且和为1。

**证明。**非负分子除以正归一化常数仍非负，且

$$\sum_{z}^{}\rho'(z) = \frac{\sum_{z}^{}\rho(z)L(z)}{Z_{norm}} = \frac{Z_{norm}}{Z_{norm}} = 1. \tag{C-13}$$

证毕。证据分数本身无需预先归一化。

**推论C.1。** 用上述 $\rho'$ 替代 $\rho$，并在下一次贝尔曼更新比较中固定 $\rho'$，则对应算子仍为 $\gamma$ 收缩。

**证明。** 命题C.2提供定理C.1的全部置信条件，直接代入即可。证毕。

若 $L$ 来自恰当概率模型中的似然，上述式子具有相应后验更新解释；从本命题本身能够得到的是非负权重的归一化性质。

给定合法联合置信 $\rho^{joint}(z,z^{(2)})$，定义

$$\rho(z) = \sum_{z^{(2)}}^{}\rho^{joint}(z,z^{(2)}),\quad\quad\rho^{(2)}(z^{(2)}) = \sum_{z}^{}\rho^{joint}(z,z^{(2)}). \tag{C-14}$$

**命题C.3。** $\rho$ 和 $\rho^{(2)}$ 均非负且归一化。

**证明。**有限非负和保持非负；两种边缘的总质量分别为

$$\sum_{z}^{}\rho(z) = \sum_{z}^{}{\sum_{z^{(2)}}^{}\rho^{joint}}(z,z^{(2)}) = 1. \tag{C-15}$$

以及

$$\sum_{z^{(2)}}^{}\rho^{(2)}(z^{(2)}) = \sum_{z^{(2)}}^{}{\sum_{z}^{}\rho^{joint}}(z,z^{(2)}) = \sum_{z}^{}{\sum_{z^{(2)}}^{}\rho^{joint}}(z,z^{(2)}) = 1. \tag{C-16}$$

证毕。因此，仅用 $z$ 条件算子进行混合时，可以将提取出的 $\rho$ 代入定理C.1，得到对应收缩性质。

边缘分布合法并不意味着两个边缘能够恢复任意联合分布。取二元 $\mathcal{Z = \{}0,1\}$，考虑

$$B_{1} = \begin{pmatrix}
1/2 & 0 \\
0 & 1/2
\end{pmatrix},\quad\quad B_{2} = \begin{pmatrix}
0 & 1/2 \\
1/2 & 0
\end{pmatrix}. \tag{C-17}$$

两个联合分布的行边缘与列边缘都为 $(1/2,1/2)$，但一个集中于相同索引，另一个集中于不同索引。因此仅保留双边缘会丢失相关结构。

如果模式算子本来就与 $z^{(2)}$ 无关，即 $\mathcal{T}_{z,z^{(2)}} = \mathcal{T}_{z}$，则可按有限求和直接化简

$$\sum_{z,z^{(2)}}^{}\rho^{joint}(z,z^{(2)})\mathcal{T}_{z,z^{(2)}}Q = \sum_{z}^{}\left( \sum_{z^{(2)}}^{}\rho^{joint}(z,z^{(2)}) \right)\mathcal{T}_{z}Q = \mathcal{T}_{\rho}Q. \tag{C-18}$$

该相等性依赖算子对被边缘化索引不敏感。联合收缩与边缘收缩分别成立，并不自动提供一般联合条件控制器的信息等价性。

**C.5 保守性。**定义保守评价系数

$$\beta_{eff} = \beta_{base} - \lambda_{w}c_{bel}, \tag{C-19}$$

其中 $\lambda_{w} \geq 0$、$c_{bel} \geq 0$。基础系数表示置信调节前的取值，有效系数表示调节后的取值。比较不同调节强度时，基础系数和置信调节量均保持不变。$c_{bel}$为非负的置信调节量，与合成价值惩罚$c_{pen}(s,a)$区分。

**命题C.4。** 有 $\beta_{eff} \leq \beta_{base}$。若 $\lambda_{w,1} \leq \lambda_{w,2}$，则

$$\beta_{eff}(\lambda_{w,2}) \leq \beta_{eff}(\lambda_{w,1}). \tag{C-20}$$

**证明。** 第一式由 $\lambda_{w}c_{bel} \geq 0$ 得到。第二式由

$$\beta_{eff}(\lambda_{w,2}) - \beta_{eff}(\lambda_{w,1}) = - (\lambda_{w,2} - \lambda_{w,1})c_{bel} \leq 0 \tag{C-21}$$

得到。证毕。

在固定集成均值与非负标准差下，评价 $\bar{Q} + \beta_{eff}\sigma_{Q}$ 随该系数减小而不增。这给出评价式保守方向的代数解释。实际控制是否安全或回报是否改善，还取决于动作选择与环境后果。

**C.6 共享表示。**给定任意模式函数族 $\{ Q_{z}(s,a)\}_{z \in \mathcal{Z}}$，定义

$$Q_{shared}(z,s,a) = Q_{z}(s,a). \tag{C-22}$$

这是将模式索引作为函数输入的定义，联合模式也可按相同方式组织。固定某一模式后，共享函数在该模式下的取值与原模式函数逐点相同，因此代入同一算子时得到相同结果。它说明的是模式条件函数的组织方式，不给出有限神经网络无误差表示任意函数族的保证。

**C.7 反馈影响。**考虑单状态单动作，两个模式奖励为 $R_{1},R_{2}$，共同后继项为 $\gamma Q_{sc}$。若代数权重取 $\rho_{1}(Q_{sc}) = \lambda_{fb}Q_{sc}$，则混合展开为

$$\begin{aligned}
\mathcal{F(}Q_{sc}) & = \lambda_{fb}Q_{sc}(\gamma Q_{sc} + R_{1}) + (1 - \lambda_{fb}Q_{sc})(\gamma Q_{sc} + R_{2}) \\
 & = (\gamma + \lambda_{fb}\Delta R)Q_{sc} + R_{2},\quad 其中\quad\Delta R = R_{1} - R_{2}.
\end{aligned} \tag{C-23}$$

以下以该仿射函数为分析对象。在全实轴上，$\lambda_{fb}Q_{sc}$ 不一定在 $\lbrack 0,1\rbrack$ 内，因此下列完整实数域定理首先是仿射反馈算子的结论。

**定理C.2。** 对任意 ${Q_{sc}}_{1},{Q_{sc}}_{2}$，

$$\mathcal{F(}{Q_{sc}}_{1}\mathcal{) - F(}{Q_{sc}}_{2}) = (\gamma + \lambda_{fb}\Delta R)({Q_{sc}}_{1} - {Q_{sc}}_{2}). \tag{C-24}$$

若 $\gamma,\lambda_{fb},\Delta R \geq 0$，则

$$\mathcal{|F(}{Q_{sc}}_{1}\mathcal{) - F(}{Q_{sc}}_{2})| = (\gamma + \lambda_{fb}\Delta R)|{Q_{sc}}_{1} - {Q_{sc}}_{2}|. \tag{C-25}$$

**证明。** 两个仿射函数值相减，常数 $R_{2}$ 抵消；提取公共因子后，再利用非负系数的绝对值等于其本身即可。证毕。

记 $c_{fb} = \gamma + \lambda_{fb}\Delta R$。若 $c_{fb} < 1$，该式给出严格收缩；若 $c_{fb} \geq 1$，取 ${Q_{sc}}_{1} = 1,{Q_{sc}}_{2} = 0$，输出距离至少等于输入距离，不能存在统一小于1的收缩因子；若 $c_{fb} > 1$，任意不同输入的距离都被放大。

因此该非负参数仿射族的阈值为

$$\lambda_{fb}\Delta R = 1 - \gamma. \tag{C-26}$$

该乘积增大时，算子的距离放大系数也增大。在本节的非负参数条件下，乘积低于式(C-26)的阈值时仍保持严格收缩，达到或超过阈值时则不再严格收缩。

在 $0 \leq \lambda_{fb}Q_{sc} \leq 1$ 的输入范围，前述混合还具有合法概率权重解释。例如取 $\gamma = 0.5$、$\lambda_{fb} = 1$、$\Delta R = 0.6$，在 ${Q_{sc}}_{1} = 1,{Q_{sc}}_{2} = 0$ 处权重分别为1和0，均合法，但输出距离为1.1，大于输入距离1。

这一有限输入对说明，即使比较点上的权重合法，随价值改变的模式加权也可能放大差异。它不把线性权重在全实轴上的每个取值都解释为概率，也不构成关于所有贝叶斯置信更新的普遍不可能性结论。

**C.8 持续证据下的后验阈值。**设初始旧模式与新模式概率之比为 $\Omega_{prior} > 0$，并假定每一步有利于新模式的两个相对证据因子分别为 $L_{evid}$ 与 ${L_{evid}}^{- 1}$，其中 $L_{evid} > 1$。在这种给定形式下，单步新旧证据之比为 ${L_{evid}}^{2}$，累计新旧概率比定义为

$$O_{belief,n} = \frac{1}{\Omega_{prior}}{L_{evid}}^{2n}. \tag{C-27}$$

式(C-27)在$n = 0$时给出$O_{belief,0} = 1/\Omega_{prior}$，与两个概率比的方向一致。单步新旧似然比为$L_{evid}^{2}$。由此可得后验比率的阈值条件及单调性。

**定理C.3。本节容差用于规定新模式后验概率的目标下界，并非控制映射的函数误差。** 若 $\delta_{post} > 0$ 且

$${L_{evid}}^{2n} \geq \frac{\Omega_{prior}}{\delta_{post}}, \tag{C-28}$$

则 $O_{belief,n} \geq 1/\delta_{post}$。

**证明。** 将式(C-28)乘以正数 $1/\Omega_{prior}$，便有

$$\frac{1}{\Omega_{prior}}{L_{evid}}^{2n} \geq \frac{1}{\Omega_{prior}}\frac{\Omega_{prior}}{\delta_{post}} = \frac{1}{\delta_{post}}. \tag{C-29}$$

证毕。该结论的输入已经包含累计证据大小条件。

若确有两模式归一化概率，$0 < \delta_{post} < 1$，则 $O_{belief,n} \geq 1/\delta_{post}$ 还给出

$$\rho_{n}(new) = \frac{O_{belief,n}}{1 + O_{belief,n}} \geq \frac{1}{1 + \delta_{post}} \geq 1 - \delta_{post}. \tag{C-30}$$

后一不等式来自 $1/(1 + \delta_{post}) - (1 - \delta_{post}) = {\delta_{post}}^{2}/(1 + \delta_{post}) \geq 0$。这是本附录对概率比的进一步数学解释。

**命题C.5。** 若 $\Omega_{prior} > 0$、$L_{evid} \geq 1$，则 $O_{belief,n + 1} \geq O_{belief,n}$。

**证明。** ${L_{evid}}^{2(n + 1)} = {L_{evid}}^{2n}{L_{evid}}^{2} \geq {L_{evid}}^{2n}$，再乘以正数 $1/\Omega_{prior}$ 即得结论。证毕。

当 $L_{evid} > 1$ 时，对定理C.3的充分条件取对数，可得到一个整数步数选择：

$$n \geq max\left\{ 0,\left\lceil \frac{log(\Omega_{prior}/\delta_{post})}{2\log L_{evid}} \right\rceil \right\}. \tag{C-31}$$

在初始概率比$\Omega_{prior}$与单步证据强度$L_{evid} > 1$固定、$\delta_{post}$趋于0时，上述充分步数的阶为$O(log(1/\delta_{post}))$。这是给定持续同向证据条件下的后验阈值分析，不是带噪声、重置和模式切换的检测器时延定理；实际时间还取决于观测间隔。

**C.9 适用范围。**固定置信混合保持收缩，更新后合法置信也可重新形成收缩算子；但随时间变化的一列收缩算子不因此自动共享固定点，也不因此保证整个学习过程收敛。定理没有将某个实际残差或权重范数证明为正确后验似然。

联合与边缘分布的概率性质不等于信息完备性，抽象共享函数存在也不等于有限神经网络的表示或训练保证。保守系数的单调性只规定评价式方向，不能从中推出误报变化点不会损害控制安全或性能。

价值依赖反馈的反例准确刻画了给定仿射族的阈值，固定置信是保持原收缩因子的充分条件，而非所有收缩算子的必要条件。证据比增长结果则依赖事先指定的幂次形式。上述适用条件用于区分BAPR的算子性质与实际模式识别、策略训练及公交控制结果。

# 附录D CS-BAPR策略响应与贝尔曼算子的数学证明

本附录分别分析完整策略的动作误差、结构化函数及其复合的正则性，以及固定惩罚价值算子的性质。第D.1节先给出最终公交软符号配置的实现对应；第D.2至D.5节及第D.8至D.10节给出条件误差分析；第D.6至D.7节单独讨论价值目标与固定算子的关系。NAU输出结构的性质不能直接用于不含NAU的软符号策略。

首先讨论标量函数和赋范空间中的可微映射，再将分析扩展到有限状态动作空间中的价值算子。

**D.1 基本条件。**设策略映射与参考映射为 $\mu_{\det},f_{ref}:\mathbb{R \rightarrow R}$，考察集合为 $X_{eval}$。这里策略符号表示确定性完整动作映射，参考符号表示同坐标下的比较映射；考察域是两映射共同定义的输入区域。给定非负常数 $\varepsilon_{der}$，导数一致性条件为

$$|{\mu_{\det}}'(x) - {f_{ref}}'(x)| \leq \varepsilon_{der},\quad 其中\quad x \in X_{eval}. \tag{D-1}$$

给定非负常数 $L_{D\mu}$，策略导数的Lipschitz条件为

$$|{\mu_{\det}}'(x) - {\mu_{\det}}'(y)| \leq L_{D\mu}|x - y|. \tag{D-2}$$

这两个条件分别刻画雅可比一致性与导数变化的正则性。前者比较两个函数在同一点的响应，后者比较同一函数在不同点的响应。可微性在后续定理中单独列为假设。

设 $X_{in},X_{out}$ 为实赋范空间，$D\mu_{\det}(x):X_{in} \rightarrow X_{out}$ 表示Fréchet导数，导数范数均为算子范数。相应条件为

$$\parallel D\mu_{\det}(x) - Df_{ref}(x) \parallel \leq \varepsilon_{der} \tag{D-3}$$

以及

$$\parallel D\mu_{\det}(x) - D\mu_{\det}(y) \parallel \leq L_{D\mu} \parallel x - y \parallel . \tag{D-4}$$

当 $X_{in} = \mathbb{R}^{n}$、$X_{out} = \mathbb{R}^{m}$ 时，导数可以表示为雅可比矩阵，矩阵范数应与所采用的输入、输出范数相容。输入特征的归一化会影响这些常数，因此所有距离和导数界都相对于同一坐标表示。

**最终公交软符号配置及其分析坐标。** 第4.4.5节采用15维归一化状态$s$和一个附加驻站动作。坐标从1编号，其中$s_{6}$、$s_{7}$、$s_{9}$分别表示采用同一尺度归一化的前向时距、后向时距和计划时距。对当前输入，仅将$s_{6}$、$s_{7}$替换为$s_{9}$，其余坐标不变，得到参考输入$\Pi(s)$。这一操作用于比较网络输出，不要求所得输入对应能够实际出现的物理平衡状态。

以下完整策略与符号目标对应正文第4.2.3节的模型定义，用于本附录的导数误差与外推条件分析。

设$m_{\theta}$为第4.2.3节定义的策略网络，其输出尚未经过最后的动作压缩；$b_{\theta}$为初值为0的可学习标量。完整确定性动作及秒数换算为

$$\begin{aligned}
q_{\theta}(s) & = b_{\theta} + m_{\theta}(s) - m_{\theta}(\Pi(s)), \\
\mu_{det,\theta}(s) & = tanhq_{\theta}(s), \\
u_{\theta}(s) & = 30\lbrack 1 + \mu_{det,\theta}(s)\rbrack.
\end{aligned} \tag{D-4a}$$

训练时，在$q_{\theta}(s)$上加入由策略标准差缩放的高斯噪声，再进行$\tanh$变换；固定策略评价采用式(D-4a)。由于$\Pi(\Pi(s)) = \Pi(s)$，参考输入处的神经修正为0，完整归一化动作却等于$\tanh b_{\theta}$。因此，$b_{\theta} = 0$时参考输入处的基准驻站为30秒，而不是0秒。该结构也不保证完整动作等于符号参考，下文的基准动作误差$\varepsilon_{val}$仍须保留。

符号代价模型的输入、函数库、辨识数据及拟合过程见第4.2.3节。以下只保留构造动作参考和分析导数误差所需的定义。

以${\widehat{c}}_{l}(s,a)$表示线路—方向组合$l$的标准化一步代价预测。候选动作集$\mathcal{A}_{g} = \{ - 1, - 5/6,\ldots,1\}$为第4.2.3节使用的驻站候选网格。令$\tau_{g} = 0.25$，以较低预测代价赋予较大权重，构造动作参考

$$f_{ref}(s) = {clip}_{\lbrack - 0.999,0.999\rbrack}\left\lbrack \frac{\sum_{a \in \mathcal{A}_{g}}^{}aexp\lbrack - {\widehat{c}}_{l}(s,a)/\tau_{g}\rbrack}{\sum_{a \in \mathcal{A}_{g}}^{}\exp\lbrack - {\widehat{c}}_{l}(s,a)/\tau_{g}\rbrack} \right\rbrack. \tag{D-4b}$$

该参考是候选动作的代价加权平均，不是直接将最小代价动作作为策略输出。各初始化对应的符号模型在辨识完成后固定，不在后续策略训练或评价中重新拟合。

策略与参考的导数均针对同一归一化输入计算，输出采用$\lbrack - 1,1\rbrack$中的完整确定性动作，不采用秒数。记$J_{\mu}(s) = D_{s}\mu_{det,\theta}(s)$、$J_{f}(s) = D_{s}f_{ref}(s)$，导数误差及策略损失为

$$\begin{aligned}
\mathcal{l}_{sym}(s) & = \frac{\parallel J_{\mu}(s) - J_{f}(s) \parallel_{F}^{2}}{1 + \parallel J_{f}(s) \parallel_{F}^{2}}, \\
\mathcal{L}_{\pi} & = \mathcal{L}_{\pi,base} + \frac{0.01}{B_{batch}}\sum_{i = 1}^{B_{batch}}\mathcal{l}_{sym}(s_{i}).
\end{aligned} \tag{D-4c}$$

其中$\parallel \cdot \parallel_{F}$为Frobenius范数，$B_{batch} = 256$，$\mathcal{L}_{\pi,base}$为第4.2.3节式(4-12)的策略基础目标。最终配置不另加策略动作与参考动作之间的平方误差损失，行为克隆系数也为0。符号项比较的是完整归一化策略与参考映射对输入状态的导数。

价值目标采用当前小批量中全部后继状态的平均导数误差：

$${\bar{\mathcal{l}}}_{sym}^{+} = \frac{1}{B_{batch}}\sum_{i = 1}^{B_{batch}}\mathcal{l}_{sym}(s_{i}'),\quad 其中\quad\lambda_{sym} = 0.01. \tag{D-4d}$$

$$\begin{aligned}
y_{i} & = \frac{r_{i}}{\sigma_{r}} + (1 - d_{i}^{term})\gamma\lbrack B_{SAC,i}^{-} - \lambda_{sym}sg({\bar{\mathcal{l}}}_{sym}^{+})\rbrack, \\
B_{SAC,i}^{-} & = \min_{1 \leq j \leq 3}Q_{\phi_{j}^{-}}(s_{i}',a_{i}') - \alpha_{ent}\log\pi_{\theta}(a_{i}' \mid s_{i}').
\end{aligned} \tag{D-4e}$$

其中$B_{SAC,i}^{-}$仅用于简写同一目标中的后继软价值。$\sigma_{r} > 0$为奖励标准差缩放量，$d_{i}^{term}$为终止标记，$a_{i}'$由当前策略在后继状态采样，$\gamma = 0.99$；$sg$表示计算梯度时把该量视为常数。目标中的符号项是整批共享的标量，不是针对每条转移单独计算的惩罚，也不通过价值目标更新策略。策略正则则按式(D-4c)，使用当前状态上的导数误差。最终配置关闭价值目标中的权重范数项。

实现对15维数值编码计算雅可比，其中包括类别编号的数值延拓偏导。这些项只能解释为对所用数值表示的正则化，不能解释成线路或站点类别连续变化的物理效应。将下文导数界用于公交连续运行量时，应固定离散类别，在同一归一化坐标下考察连续截面，并避开参考截断的不可微边界。样本上的平均导数损失较小，不自动给出整条外推路径上的统一误差上界。采用ReLU的对照配置，其激活边界仍需分别处理，不能直接使用tanh软符号配置的正则性结论。

**D.2 标量误差。****命题D.1。区间两端点是本命题的局部实数变量；误差函数定义为策略动作与参考动作之差。** 设 $a \leq b$，$\mu_{\det},f_{ref}$ 可微，且对 $x \in \lbrack a,b)$ 有

$$|{\mu_{\det}}'(x) - {f_{ref}}'(x)| \leq \varepsilon_{der},\quad\quad\varepsilon_{der} \geq 0. \tag{D-5}$$

则

$$|\lbrack\mu_{\det}(b) - f_{ref}(b)\rbrack - \lbrack\mu_{\det}(a) - f_{ref}(a)\rbrack| \leq \varepsilon_{der}(b - a). \tag{D-6}$$

**证明。** 令 $e_{map}(x) = \mu_{\det}(x) - f_{ref}(x)$。当 $a = b$ 时结论直接成立。当 $a < b$ 时，$e_{map}$ 在闭区间上连续、开区间上可微，由中值定理存在 $\xi \in (a,b)$，使

$$e_{map}(b) - e_{map}(a) = {e_{map}}'(\xi)(b - a). \tag{D-7}$$

取绝对值并代入导数界即可。证毕。

若基准误差满足 $|e_{map}(a)| \leq \varepsilon_{val}$，则由三角不等式有

$$|e_{map}(b)| \leq \varepsilon_{val} + \varepsilon_{der}(b - a). \tag{D-8}$$

该界表明，导数差异沿外推路径累积；基准点的函数误差与路径上的累积误差共同决定最终误差。

**命题D.2。** 设 $\varepsilon_{val},\varepsilon_{der},L \geq 0$，$\mu_{\det},f_{ref}$ 可微，$a \leq b$，并满足

$$|e_{map}(a)| \leq \varepsilon_{val},\quad\quad|{e_{map}}'(x)| \leq \varepsilon_{der} + L(x - a),\quad x \in \lbrack a,b). \tag{D-9}$$

则

$$|e_{map}(b)| \leq \varepsilon_{val} + \varepsilon_{der}(b - a) + \frac{L}{2}(b - a)^{2}. \tag{D-10}$$

**证明。** 定义边界函数

$$B_{err}(x) = \varepsilon_{val} + \varepsilon_{der}(x - a) + \frac{L}{2}(x - a)^{2}. \tag{D-11}$$

其导数为 ${B_{err}}'(x) = \varepsilon_{der} + L(x - a)$。由 ${e_{map}}'(x) \leq {B_{err}}'(x)$ 可知 $e_{map} - B_{err}$ 在区间上非增，且 $e_{map}(a) - B_{err}(a) \leq 0$，所以 $e_{map}(b) \leq B_{err}(b)$。同理，由 $- {e_{map}}'(x) \leq {B_{err}}'(x)$ 可知 $- e_{map} - B_{err}$ 非增，得到 $- e_{map}(b) \leq B_{err}(b)$。合并两式即有 $|e_{map}(b)| \leq B_{err}(b)$。证毕。

该证明只需要导数比较，不需要额外假设 ${e_{map}}'$ 连续。

**命题D.3。** 若

$$\begin{aligned}
|{\mu_{\det}}'(a) - {f_{ref}}'(a)| & \leq \varepsilon_{der}, \\
|{\mu_{\det}}'(x) - {\mu_{\det}}'(a)| & \leq L_{D\mu}|x - a|, \\
|{f_{ref}}'(x) - {f_{ref}}'(a)| & \leq L_{Df}|x - a|,
\end{aligned} \tag{D-12}$$

则

$$|{\mu_{\det}}'(x) - {f_{ref}}'(x)| \leq \varepsilon_{der} + (L_{D\mu} + L_{Df})|x - a|. \tag{D-13}$$

**证明。** 在导数差中加减基准点的两个导数：

$$\begin{aligned}
{\mu_{\det}}'(x) - {f_{ref}}'(x) = & \lbrack{\mu_{\det}}'(x) - {\mu_{\det}}'(a)\rbrack + \lbrack{\mu_{\det}}'(a) - {f_{ref}}'(a)\rbrack \\
 & + \lbrack{f_{ref}}'(a) - {f_{ref}}'(x)\rbrack.
\end{aligned} \tag{D-14}$$

取绝对值，逐项应用三角不等式和已知上界即可。证毕。

将命题D.3代入命题D.2，得到

$$|\mu_{\det}(b) - f_{ref}(b)| \leq \varepsilon_{val} + \varepsilon_{der}(b - a) + \frac{L_{D\mu} + L_{Df}}{2}(b - a)^{2}. \tag{D-15}$$

当参考函数的导数为常数时，取 $L_{Df} = 0$ 即得到相应特殊情形。

**D.3 向量误差。****命题D.4。** 设 $\mu_{\det},f_{ref}:X_{in} \rightarrow X_{out}$ 可微，$\varepsilon_{der} \geq 0$，并在连接 $x_{0}$ 与 $x$ 的线段上满足

$$\parallel D\mu_{\det}(z) - Df_{ref}(z) \parallel \leq \varepsilon_{der}. \tag{D-16}$$

若 $\parallel \mu_{\det}(x_{0}) - f_{ref}(x_{0}) \parallel \leq \varepsilon_{val}$，则

$$\parallel \mu_{\det}(x) - f_{ref}(x) \parallel \leq \varepsilon_{val} + \varepsilon_{der} \parallel x - x_{0} \parallel . \tag{D-17}$$

**证明。** 记 $e_{map} = \mu_{\det} - f_{ref}$，$\Delta x = x - x_{0}$，路径 $z(t) = x_{0} + t\Delta x$。由赋范空间上的均值不等式，

$$\parallel e_{map}(x) - e_{map}(x_{0}) \parallel \leq \sup_{0 \leq t \leq 1} \parallel De_{map}(z(t)) \parallel \, \parallel \Delta x \parallel \leq \varepsilon_{der} \parallel \Delta x \parallel . \tag{D-18}$$

再用三角不等式得到结论。也可以对每个范数不超过1的连续线性泛函 $\mathcal{l}_{fun}:X_{out}\mathbb{\rightarrow R}$，将标量中值定理应用于 $\mathcal{l}_{fun} \circ e_{map} \circ z$，最后利用范数的对偶表示获得相同结果。证毕。

上述结论只需在线段上控制导数误差；若全空间均满足同一条件，该结论自然成立。

**命题D.5。** 设 $X_{out}$ 为实Banach空间，$e_{path}:\lbrack 0,1\rbrack \rightarrow X_{out}$ 连续，且在 $\lbrack 0,1)$ 的每一点存在右导数。路径导数界的常数分别表示起点函数误差、路径上的初始导数幅度及其随路径参数线性增长的斜率。给定 $\varepsilon_{val},\varepsilon_{path},L_{path} \geq 0$，若

$$\parallel e_{path}(0) \parallel \leq \varepsilon_{val},\quad\quad \parallel e_{path, +}'(t) \parallel \leq \varepsilon_{path} + L_{path}t, \tag{D-19}$$

则

$$\parallel e_{path}(1) \parallel \leq \varepsilon_{val} + \varepsilon_{path} + \frac{L_{path}}{2}. \tag{D-20}$$

**证明。** 令 $B_{err}(t) = \varepsilon_{val} + \varepsilon_{path}t + L_{path}t^{2}/2$。对任意连续线性泛函 $\mathcal{l}_{fun}$，$\parallel \mathcal{l}_{fun} \parallel \leq 1$，有

$$(\mathcal{l}_{fun} \circ e_{path})_{+}'(t) = \mathcal{l}_{fun}(e_{path, +}'(t)) \leq {B_{err}}'(t). \tag{D-21}$$

给定 $\eta > 0$，令 $v_{cmp}(t) = \mathcal{l}_{fun}(e_{path}(t)) - B_{err}(t) - \eta t$，则 $v_{cmp, +}'(t) \leq - \eta$，且 $v_{cmp}(0) \leq 0$。假设 $v_{cmp}(1) > 0$，连续性保证 $v_{cmp}$ 在 $\lbrack 0,1\rbrack$ 上取得最小值；由于 $v_{cmp}(1) > v_{cmp}(0)$，最小点 $t_{*}$ 严格小于1。在该点，右侧差商非负，故右导数非负，与 $v_{cmp, +}'(t_{*}) \leq - \eta$ 矛盾。因此 $v_{cmp}(1) \leq 0$。

令 $\eta \downarrow 0$，得 $\mathcal{l}_{fun}(e_{path}(1)) \leq B_{err}(1)$。利用范数的对偶表示，对全部 $\parallel \mathcal{l}_{fun} \parallel \leq 1$ 取上确界，即得所需结论。证毕。

连续可微路径满足上述右导数条件。

**推论D.6。** 设 $\mu_{\det},f_{ref}:X_{in} \rightarrow X_{out}$ 在包含线段 $\lbrack x_{0},x\rbrack$ 的开集上可微，$X_{out}$ 完备。若

$$\begin{aligned}
 \parallel \mu_{\det}(x_{0}) - f_{ref}(x_{0}) \parallel & \leq \varepsilon_{val}, \\
 \parallel D\mu_{\det}(x_{0}) - Df_{ref}(x_{0}) \parallel & \leq \varepsilon_{der}, \\
 \parallel D\mu_{\det}(z) - D\mu_{\det}(x_{0}) \parallel & \leq L_{D\mu} \parallel z - x_{0} \parallel , \\
 \parallel Df_{ref}(z) - Df_{ref}(x_{0}) \parallel & \leq L_{Df} \parallel z - x_{0} \parallel 
\end{aligned} \tag{D-22}$$

在线段上成立，则

$$\begin{aligned}
 \parallel \mu_{\det}(x) - f_{ref}(x) \parallel \leq & \varepsilon_{val} + \varepsilon_{der} \parallel x - x_{0} \parallel \\
 & + \frac{L_{D\mu} + L_{Df}}{2} \parallel x - x_{0} \parallel^{2}.
\end{aligned} \tag{D-23}$$

**证明。** 令 $\Delta x = x - x_{0}$，构造真实误差路径

$$e_{path}(t) = \mu_{\det}(x_{0} + t\Delta x) - f_{ref}(x_{0} + t\Delta x). \tag{D-24}$$

由链式法则，

$${e_{path}}'(t) = \lbrack D\mu_{\det}(x_{0} + t\Delta x) - Df_{ref}(x_{0} + t\Delta x)\rbrack\Delta x. \tag{D-25}$$

在算子差中加减 $D\mu_{\det}(x_{0})$ 和 $Df_{ref}(x_{0})$，得到

$$\begin{aligned}
 \parallel {e_{path}}'(t) \parallel & \leq \lbrack\varepsilon_{der} + (L_{D\mu} + L_{Df})t \parallel \Delta x \parallel \rbrack \parallel \Delta x \parallel \\
 & = \varepsilon_{der} \parallel \Delta x \parallel + (L_{D\mu} + L_{Df}) \parallel \Delta x \parallel^{2}t.
\end{aligned} \tag{D-26}$$

取 $\varepsilon_{path} = \varepsilon_{der} \parallel \Delta x \parallel$、$L_{path} = (L_{D\mu} + L_{Df}) \parallel \Delta x \parallel^{2}$，应用命题D.5即可。证毕。

由线段构造、链式法则和导数误差界，向量情形归结为上述路径命题。

若在整条路径上使用统一的导数误差上界，则还可得到

$$\parallel \mu_{\det}(x) - f_{ref}(x) \parallel \leq \varepsilon_{val} + \varepsilon_{der} \parallel \Delta x \parallel + (L_{D\mu} + L_{Df}) \parallel \Delta x \parallel^{2}. \tag{D-27}$$

这是先把半径 $\parallel \Delta x \parallel$ 范围内的导数误差统一上界为 $\varepsilon_{der} + (L_{D\mu} + L_{Df}) \parallel \Delta x \parallel$，再应用命题D.4所得。它的二次项系数为1，路径比较所得系数为 $1/2$。两者使用误差沿路径变化的信息精细程度不同，因此应保留其区别。

取标量函数 $\mu_{\det}(x) = c_{quad}x^{2}$，参考函数为其在 $a$ 点的一阶展开

$$f_{ref}(x) = c_{quad}a^{2} + 2c_{quad}a(x - a). \tag{D-28}$$

此时基准函数值和导数误差均为0，$L_{D\mu} = 2|c_{quad}|$，$L_{Df} = 0$，且

$$|\mu_{\det}(x) - f_{ref}(x)| = |c_{quad}||x - a|^{2} = \frac{L_{D\mu}}{2}|x - a|^{2}. \tag{D-29}$$

因此，在这组条件下二次项中的 $1/2$ 不能普遍进一步减小。若参考函数本来就是策略自身的局部线性化，那么该界刻画的是策略偏离其线性化的程度，而非相对于未知最优策略的误差。

**D.4 输出结构。**本节分析NAU线性输出与NMU启发的二次修正结构的数学性质。以NAU为例，其理想化线性映射由各输入分量的加权和构成，权重取−1、0或1：

$$F_{NAU}(x) = \sum_{i = 1}^{n}c_{NAU,i}x_{i},\quad\quad c_{NAU,i} \in \{ - 1,0,1\}. \tag{D-30}$$

其中各坐标的NAU权重记为 $c_{NAU,i}$。固定其余坐标，对第$j$ 个坐标求导，有

$$\frac{\partial F_{NAU}}{\partial x_{j}} = c_{NAU,j}. \tag{D-31}$$

因此，该线性层的雅可比为常数，雅可比Lipschitz常数可取0。该性质对任意固定实权重同样成立，不要求训练权重已经精确离散为三个值。固定任意其余坐标只增加常数项，因此每个坐标方向的一维截面具有相同性质。

对 $F_{quad}(x) = c_{quad}x^{2}$，有 ${F_{quad}}'(x) = 2c_{quad}x$，因此下一式给出该二次函数的导数变化界。二次系数及随后引入的缩放系数均为固定实数。

$$|{F_{quad}}'(x) - {F_{quad}}'(y)| = 2|c_{quad}||x - y|. \tag{D-32}$$

导数Lipschitz常数可取 $2|c_{quad}|$。进一步，对 $G_{quad}(x) = \omega_{quad}c_{quad}x^{2}$，常数可取 $2|\omega_{quad}c_{quad}|$。线性缩放相应地缩放了导数的Lipschitz常数。这里$c_{quad}$为二次项系数，$\omega_{quad}$为乘性缩放系数。

上述结论精确适用于标量二次函数。更高次乘积的导数条件取决于次数和定义域。例如 $x^{3}$ 的导数在全实线上不是Lipschitz的，但在任意给定有界区间上具有有限Lipschitz常数。

**命题D.7。** 令 $r_{ReLU}(x) = max\{ 0,x\}$。不存在有限非负 $L$，使在全部非零点上有

$$|{r_{ReLU}}'(x) - {r_{ReLU}}'(y)| \leq L|x - y|. \tag{D-33}$$

**证明。** 当 $L = 0$ 时，取 $x = - 1$、$y = 1$，左侧为1，右侧为0，矛盾。当 $L > 0$ 时，取

$$x = - \frac{1}{4L},\quad\quad y = \frac{1}{4L}. \tag{D-34}$$

由于负半轴导数为0、正半轴导数为1，左侧等于1，而右侧等于 $1/2$，仍矛盾。证毕。

证明没有使用原点处的导数约定。

**命题D.8。** 若 ${f_{ref}}'$ 在原点左、右邻域分别恒等于 $d_{left}$ 和 $d_{right}$，且 $d_{left} \neq d_{right}$，则不存在覆盖该邻域的有限导数Lipschitz常数。两个常数分别是原点左侧与右侧邻域中的导数值；证明中的正数是到原点的局部距离。

**证明。** 假设存在常数 $L \geq 0$。取足够小的 $t > 0$，使 $- t$ 和 $t$ 位于给定左右邻域，并满足 $2Lt < |d_{left} - d_{right}|$；当 $L = 0$ 时任取这样的 $t$。则

$$|d_{left} - d_{right}| = |{f_{ref}}'( - t) - {f_{ref}}'(t)| \leq 2Lt < |d_{left} - d_{right}|, \tag{D-35}$$

矛盾。证毕。

导数跳跃与函数本身是否Lipschitz是两个问题。ReLU本身是1-Lipschitz函数，有限权重的有限层ReLU网络仍有有限函数Lipschitz常数；其输出幅度因而至多随输入范数线性增长。导数正则性定理不适用，并不等价于函数值必然出现超多项式增长。

**D.5 复合结构。****命题D.9。** 若标量函数 $g$ 可微，且 $|g'(x)| \leq B_{g}$ 对所有 $x$ 成立，则

$$|g(x) - g(y)| \leq B_{g}|x - y|. \tag{D-36}$$

**证明。** 当 $x = y$ 时显然成立；否则在二者之间应用中值定理。证毕。

函数的Lipschitz性质在此由导数幅度上界保证。导数Lipschitz本身不蕴含函数在全实线上Lipschitz，例如 $g(x) = x^{2}$ 的导数Lipschitz常数为2，而导数幅度不受全局常数控制。

**命题D.10。** 设 $f_{out},g$ 可微，并满足

$$\begin{aligned}
|{f_{out}}'(u) - {f_{out}}'(v)| & \leq L_{out}|u - v|, \\
|g(x) - g(y)| & \leq K_{g}|x - y|, \\
|g'(x)| & \leq B_{g},\quad\quad|{f_{out}}'(u)| \leq B_{out}, \\
|g'(x) - g'(y)| & \leq L_{g}|x - y|.
\end{aligned} \tag{D-37}$$

则 $f_{out} \circ g$ 的导数Lipschitz常数可以取

$$L_{f_{out} \circ g} = L_{out}K_{g}B_{g} + B_{out}L_{g}. \tag{D-38}$$

**证明。** 链式法则给出 $(f_{out} \circ g)'(x) = {f_{out}}'(g(x))g'(x)$。加减 ${f_{out}}'(g(y))g'(x)$，有

$$\begin{aligned}
 & (f_{out} \circ g)'(x) - (f_{out} \circ g)'(y) \\
 = & \lbrack{f_{out}}'(g(x)) - {f_{out}}'(g(y))\rbrack g'(x) \\
 & + {f_{out}}'(g(y))\lbrack g'(x) - g'(y)\rbrack.
\end{aligned} \tag{D-39}$$

取绝对值并分别估计，得到

$$\begin{aligned}
 & |(f_{out} \circ g)'(x) - (f_{out} \circ g)'(y)| \\
 \leq & L_{out}|g(x) - g(y)|B_{g} + B_{out}L_{g}|x - y| \\
 \leq & (L_{out}K_{g}B_{g} + B_{out}L_{g})|x - y|.
\end{aligned} \tag{D-40}$$

证毕。当 $B_{out} = 1$ 时，该上界化为 $L_{out}K_{g}B_{g} + L_{g}$。

对向量映射，链式法则为 $D(f_{out} \circ g)(x) = Df_{out}(g(x))Dg(x)$。按上述方式加减中间项，并使用算子范数的次乘性，可得到相同常数形式。这是对标量论证的直接数学推广。

特征函数的Lipschitz常数不能替代其导数Lipschitz常数。取 $f_{out}(u) = u$、$g(x) = max\{ 0,x\}$，则 $L_{out} = 0$，$K_{g} = 1$，但 $f_{out} \circ g$ 的导数在原点两侧跳跃。由 $L_{out}K_{g}$ 或 $L_{out}K_{g}^{2}$ 单独构造复合导数上界都会错误地得到0。

外层平滑饱和也不能一般性消除内部导数跳跃。若 $g$ 为左右斜率不同的分段线性函数，且 $g(0) = 0$，由于 $\tanh'(0) = 1$，$\tanh \circ g$ 在原点两侧保留不同的一侧导数。只有完整映射确实满足相应条件时，才能应用前述二次误差界。

**D.6 基本性质。**前述分析关注完整策略对状态的响应。下面单独研究固定惩罚价值算子。实际公交配置同时使用式(D-4c)的策略正则和式(D-4e)的价值目标，两种训练操作不相互替代。本节以固定参数、精确期望和有限集合为条件分析理想价值更新，不将其直接等同于随机梯度训练。

设状态、动作和模式集合 $\mathcal{S,A,Z}$ 有限非空。令 $0 \leq \gamma < 1$，模式转移核满足

$$P_{z}(s' \mid s,a) \geq 0,\quad\quad\sum_{s'}^{}P_{z}(s' \mid s,a) = 1. \tag{D-41}$$

给定固定奖励 $R_{z}$、认知惩罚 $\Gamma_{epi,z}$、符号惩罚 $\Gamma_{sym,z}$ 以及非负系数 $\lambda_{epi},\lambda_{sym}$。定义

$$V_{Q}(s) = \max_{a}Q(s,a), \tag{D-42}$$

$$\begin{aligned}
(\mathcal{T}_{z}Q)(s,a) = R_{z}(s,a) + \gamma\lbrack & \sum_{s'}^{}P_{z}(s' \mid s,a)V_{Q}(s') \\
 & - \lambda_{epi}\Gamma_{epi,z}(s,a) \\
 & - \lambda_{sym}\Gamma_{sym,z}(s,a) - \kappa_{fix}\rbrack.
\end{aligned} \tag{D-43}$$

给定固定置信 $\rho(z) \geq 0$、$\sum_{z}^{}\rho(z) = 1$，混合算子为

$$(\mathcal{T}_{CS}Q)(s,a) = \sum_{z}^{}\rho(z)(\mathcal{T}_{z}Q)(s,a). \tag{D-44}$$

范数取有限状态动作空间上的最大范数。固定是指比较两个输入 $Q_{1},Q_{2}$ 时，上述奖励、惩罚、置信和转移均相同。

使用C.2的混合随机核 ${\overline{P}}_{\rho}$，将固定项记为

$$\begin{aligned}
{\overline{R}}_{pen,CS}(s,a) = \sum_{z}^{}\rho(z)\lbrack & R_{z}(s,a) - \gamma\lambda_{epi}\Gamma_{epi,z}(s,a) \\
 & - \gamma\lambda_{sym}\Gamma_{sym,z}(s,a) - \gamma\kappa_{fix}\rbrack.
\end{aligned} \tag{D-45}$$

**命题D.11。** 在D.6的条件下，$\mathcal{T}_{CS}$ 单调，满足折扣平移性质，且

$$\parallel \mathcal{T}_{CS}Q_{1} - \mathcal{T}_{CS}Q_{2} \parallel_{\infty} \leq \gamma \parallel Q_{1} - Q_{2} \parallel_{\infty}. \tag{D-46}$$

**证明。** 按有限求和重排，$\mathcal{T}_{CS}Q = {\overline{R}}_{pen,CS} + \gamma{\overline{P}}_{\rho}V_{Q}$。命题C.1保证 ${\overline{P}}_{\rho}$ 是随机核，符号惩罚在此次比较中固定，因而并入 ${\overline{R}}_{pen,CS}$ 后满足定理A.1的条件。三项结论直接成立，唯一固定点及精确迭代的几何误差界由推论A.1给出。证毕。

对一次给定的小批量，固定策略、符号参考和奖励缩放量后，式(D-4d)的平均导数误差可以作为式(D-43)中的固定符号项，并对各模式使用同一数值。该项在同一小批量内由全部记录共享，但可随下一次采样或策略更新改变；停止梯度不意味着它在整个训练过程中保持不变。

最终公交配置的符号系数为0.01，价值目标中的权重范数项关闭。实际更新对三个目标价值网络取较小估计，并从当前策略采样后继动作；式(D-42)则使用理想最大值聚合，公交实现也不显式构造模式转移核后再混合。因此，命题D.11用于说明固定加性惩罚不改变所分析价值算子的收缩性质，不将理想模式算子与实际小批量更新视为同一算法，也不据此证明策略与价值网络的联合训练收敛。

**D.7 反馈影响。**给定非负证据 $\mathcal{L}(z)$，若 $Z_{norm} = \sum_{z}^{}\rho(z)\mathcal{L}(z) > 0$，则更新 $\rho^{+}(z) = \rho(z)\mathcal{L}(z)/Z_{norm}$ 由命题C.2保持非负且归一化。在下一次输入比较中固定 $\rho^{+}$，便可继续使用命题D.11。

考虑单状态单动作模型，令符号惩罚随输入值变化：$\Gamma(Q_{sc}) = C_{pen} - \mu_{fb}Q_{sc}$。其中固定截距取实数，反馈斜率非负；沿用非负符号惩罚系数与严格折扣条件。把与 $Q_{sc}$ 无关的常数合并为固定即时项，该项不随被比较的动作价值输入变化，算子为

$$T_{symfb}(Q_{sc}) = R_{0} + \gamma(1 + \lambda_{sym}\mu_{fb})Q_{sc}. \tag{D-47}$$

记 $c_{symfb} = \gamma(1 + \lambda_{sym}\mu_{fb}) \geq 0$，则

$$|T_{symfb}(Q_{sc,1}) - T_{symfb}(Q_{sc,2})| = c_{symfb}|Q_{sc,1} - Q_{sc,2}|. \tag{D-48}$$

当 $c_{symfb} < 1$ 时该算子仍是收缩；当 $c_{symfb} = 1$ 时为非扩张但不是严格收缩；当 $c_{symfb} > 1$ 时任意不同输入间距离都扩大。后两种情形取 $Q_{sc,1} = 1,Q_{sc,2} = 0$ 就足以排除小于1的全局收缩系数。

若 $0 < \gamma < 1$、$\lambda_{sym} > 0$，临界反馈强度为

$$\mu_{fb,crit} = \frac{1/\gamma - 1}{\lambda_{sym}}. \tag{D-49}$$

例如 $\gamma = 0.99$、$\lambda_{sym} = 0.01$ 时，临界值约为1.0101。若要求所考察两个输入上的惩罚非负，可取 $C_{pen} \geq \mu_{fb}$，则 $\Gamma(0),\Gamma(1) \geq 0$，仍得到相同反例。

该模型说明固定惩罚是收缩性的充分条件，而不是必要条件。输入依赖反馈是否破坏收缩，取决于其增益与折扣共同决定的距离放大系数。

**D.8 误差关系。**本节在动作映射层面引入一个有限基函数模型，分析它与给定参考动作及学习策略之间的导数误差。设$\varphi_{i}:X_{in} \rightarrow X_{out}$为可微基函数，$i = 1,\ldots,n$，实系数向量为$c_{coef}$，相应模型记为$f_{c_{coef}}$。

这里的有限线性组合是用于误差分解的一般动作模型，不等同于第D.1节预测一步代价的${\widehat{c}}_{l}(s,a)$。第D.1节先拟合代价，再经过指数加权及截断构造动作参考$f_{ref}$，因而不能把代价模型的系数误差或预测误差直接当作动作参考的导数误差。将本节结论用于该配置时，需要对完整参考构造另外建立相应误差条件。以下先在这些导数误差界已经给定的条件下展开分析。符号模型写为

$$f_{c_{coef}}(x) = \sum_{i = 1}^{n}c_{coef,i}\varphi_{i}(x). \tag{D-50}$$

有限和与标量乘法保持可微性，因而

$$Df_{c_{coef}}(x) = \sum_{i = 1}^{n}c_{coef,i}D\varphi_{i}(x). \tag{D-51}$$

可微性只依赖基函数及有限线性组合。系数向量的稀疏性则需要通过估计时的约束或正则化获得。

**命题D.12。** 假设同一个系数向量 $c_{coef}$ 在区域 $X_{eval}$ 满足

$$\begin{aligned}
 \parallel Df_{ref}(x) - Df_{c_{coef}}(x) \parallel & \leq \varepsilon_{sym}, \\
 \parallel D\mu_{\det}(x) - Df_{c_{coef}}(x) \parallel & \leq \varepsilon_{pol}.
\end{aligned} \tag{D-52}$$

其中，$\varepsilon_{sym}$为符号模型与参考映射的导数误差上界，$\varepsilon_{pol}$为策略与同一符号模型的导数误差上界。则

$$\parallel D\mu_{\det}(x) - Df_{ref}(x) \parallel \leq \varepsilon_{sym} + \varepsilon_{pol},\quad\quad x \in X_{eval}. \tag{D-53}$$

**证明。** 加减 $Df_{c_{coef}}(x)$，取范数并使用三角不等式即可。证毕。

使用同一 $c_{coef}$ 是这一组合关系的实质条件。如果策略拟合的是另一模型 $f_{\widetilde{c_{coef}}}$，还需要额外控制 $\parallel Df_{c_{coef}} - Df_{\widetilde{c_{coef}}} \parallel$，不能直接省略这一误差。

若上述两个误差界覆盖从 $x_{0}$ 到 $x$ 的整条线段，并满足基准函数值误差界，则由命题D.4有

$$\parallel \mu_{\det}(x) - f_{ref}(x) \parallel \leq \varepsilon_{val} + (\varepsilon_{sym} + \varepsilon_{pol}) \parallel x - x_{0} \parallel . \tag{D-54}$$

若两个误差条件在全空间成立，上式对任意外推路径成立。符号模型在训练区域的误差界，与其在整条外推路径的误差界，是不同的信息。

**D.9 环境一致性。**以 $\mathcal{E}$，列出所考察的环境。每个环境 $e$ 对应参考映射 $f_{e}$ 和区域 $D_{e}$。对这些环境使用同一系数向量，其导数 $c_{coef}$ 的一致性条件定义为

$$\parallel Df_{e}(x) - Df_{c_{coef}}(x) \parallel \leq \varepsilon_{IRM},\quad 其中\quad e\mathcal{\in E,}\ x \in D_{e}. \tag{D-55}$$

这里，$\varepsilon_{IRM}$表示该共同符号模型在所列各环境中的统一导数误差上界。若策略也满足

$$\parallel D\mu_{\det}(x) - Df_{c_{coef}}(x) \parallel \leq \varepsilon_{pol},\quad 其中\quad e\mathcal{\in E,}\ x \in D_{e}, \tag{D-56}$$

则逐环境应用命题D.12，得到

$$\parallel D\mu_{\det}(x) - Df_{e}(x) \parallel \leq \varepsilon_{IRM} + \varepsilon_{pol}. \tag{D-57}$$

进一步假设，对每个所考察环境，基准点满足$\parallel \mu_{\det}(x_{0}) - f_{e}(x_{0}) \parallel \leq \varepsilon_{val}$，且式(D-57)的导数误差界在连接该基准点与待评价点的整条线段上成立。上述基准误差上界对所有所考察环境统一成立时，由命题D.4得到

$$\parallel \mu_{\det}(x) - f_{e}(x) \parallel \leq \varepsilon_{val} + (\varepsilon_{IRM} + \varepsilon_{pol}) \parallel x - x_{0} \parallel . \tag{D-58}$$

考虑数学分解

$$f_{e}(x) = f_{sh}(x) + f_{s,e}(x), \tag{D-59}$$

其中$f_{sh}$是真实映射中在各环境共享的部分，$f_{s,e}$为环境相关部分；前文的$f_{c_{coef}}$仍表示由数据拟合的符号模型。以$\varepsilon_{sh}$表示符号模型相对共享真实部分的导数误差上界，假设

$$\parallel Df_{c_{coef}}(x) - Df_{sh}(x) \parallel \leq \varepsilon_{sh}, \tag{D-60}$$

并且在测试环境中 $\parallel Df_{s,test}(x) \parallel \leq \varepsilon_{s,test}$。则

$$\begin{aligned}
 \parallel Df_{c_{coef}}(x) - Df_{test}(x) \parallel & \leq \parallel Df_{c_{coef}}(x) - Df_{sh}(x) \parallel + \parallel Df_{s,test}(x) \parallel \\
 & \leq \varepsilon_{sh} + \varepsilon_{s,test}.
\end{aligned} \tag{D-61}$$

这一分解将符号模型对共享部分的估计误差与环境特有扰动分开。若再获得策略对该符号模型的导数拟合误差，还可继续用三角不等式组合。

**D.10 误差传播。****命题D.13。** 设已知某个基准点 $a$ 的导数误差满足

$$|{\mu_{\det}}'(a) - {f_{ref}}'(a)| \leq \varepsilon_{emp} + \Delta_{gen}, \tag{D-62}$$

其中，$\varepsilon_{emp}$和$\Delta_{gen}$分别表示给定的经验导数误差界及其推广误差界。右侧非负，并且函数值基准误差满足命题D.2的条件，策略导数与参考导数的正则性满足命题D.3的条件。则对 $b \geq a$，

$$\begin{aligned}
|\mu_{\det}(b) - f_{ref}(b)| \leq & \varepsilon_{val} + (\varepsilon_{emp} + \Delta_{gen})(b - a) \\
 & + \frac{L_{D\mu} + L_{Df}}{2}(b - a)^{2}.
\end{aligned} \tag{D-63}$$

**证明。** 将命题D.3和命题D.2中的 $\varepsilon_{der}$ 替换为给定非负和即可。证毕。

这是以基准点误差为前提的确定性传播界。若统计估计能够给出该点的误差控制，可将相应结论代入此式。

**D.11 适用范围。**本附录的导数误差结论适用于完整策略映射满足正则性条件的情形。基准误差、导数差异和外推距离共同决定函数误差的上界；动作误差与闭环累计回报之间还需要环境动力学和价值函数的敏感性条件。公交需求倍数与归一化状态空间中的距离具有不同含义，应用误差界时应先建立两者的对应关系。

固定置信算子的收缩性描述精确价值更新。置信、惩罚和价值同时变化时，需要进一步控制算子漂移及函数逼近误差。算术输出配置中的NAU线性层和NMU启发的二次函数需通过完整复合映射分析；不含NAU的软符号配置则使用第D.1节给出的完整动作及参考映射，二者的正则性不能互相代用。

符号模型与多环境误差分解均以给定的拟合误差条件为基础。稀疏系数的恢复、基准点误差的统计估计以及环境共享成分的识别分别属于估计问题。三角不等式比较的是可用误差上界，两个上界的大小不能单独确定两种估计方法的实际优劣。平均经验误差也不能直接代替指定点上的导数误差。

# 附录E MC-WM残差模型与条件排序的数学分析

本附录围绕模型误差和置信尺度两条主线展开。第E.1至E.5节分析固定策略价值、数据覆盖及合法残差更新条件，第E.3节另给出无折扣有限事件时域的条件误差界；第E.6节只讨论指定静态误差信号的排序关系。第E.9节直接对应公交改进中的归一化置信度递推更新，解释预测误差稳定时权重仍可能持续减小的原因。

**E.1 基本条件。**设$\mathcal{S}$、$\mathcal{A}$为有限非空集合。一个折扣决策过程由转移核$P$、奖励$R$和折扣因子$\gamma$构成，其中

$$P(s' \mid s,a) \geq 0,\quad\quad\sum_{s'\mathcal{\in S}}^{}P(s' \mid s,a) = 1, \tag{E-1}$$

$$|R(s,a)| \leq R_{\max},\quad\quad R_{\max} \geq 0,\quad\quad 0 \leq \gamma < 1. \tag{E-2}$$

策略$\pi$满足

$$\pi(a \mid s) \geq 0,\quad\quad\sum_{a}^{}\pi(a \mid s) = 1. \tag{E-3}$$

策略函数的非负性与归一化是概率解释所需条件。固定策略后，贝尔曼评价算子为

$$(T_{P}^{\pi}V)(s) = \sum_{a}^{}\pi(a \mid s)\left\lbrack R(s,a) + \gamma\sum_{s'}^{}P(s' \mid s,a)V(s') \right\rbrack. \tag{E-4}$$

价值函数空间为有限维实向量空间$\mathbb{R}^{\mathcal{|S|}}$，范数为$\parallel V \parallel_{\infty} = \max_{s}|V(s)|$。

给定源核$P_{sim}$及修正函数$\widehat{\delta}$，定义

$${\widehat{P}}_{R}(s' \mid s,a) = P_{sim}(s' \mid s,a) + \widehat{\delta}(s,a,s'). \tag{E-5}$$

假设对每个$(s,a)$有

$$P_{sim}(s' \mid s,a) + \widehat{\delta}(s,a,s') \geq 0,\quad\quad\sum_{s'}^{}\widehat{\delta}(s,a,s') = 0. \tag{E-6}$$

则

$$\sum_{s'}^{}{\widehat{P}}_{R}(s' \mid s,a) = \sum_{s'}^{}P_{sim}(s' \mid s,a) + \sum_{s'}^{}\widehat{\delta}(s,a,s') = 1. \tag{E-7}$$

由式(E-6)的非负性与式(E-7)的归一化，式(E-5)给出的组合核是合法随机核。残差本身可以取负值，要求非负的是组合后的概率。将该组合核与给定奖励及折扣因子结合，即得到新的有限决策过程。

本节的残差是转移概率的加性修正。正文实现中的状态增量残差则是连续状态预测器的加性修正。二者共享“保留基础模型并修正差异”的思想，但数学对象不同。

**E.2 收敛性。****引理E.1。** 若$p_{j} \geq 0$且$\sum_{j}^{}p_{j} = 1$，则任意有限实数序列$u_{j}$满足

$$\left| \sum_{j}^{}p_{j}u_{j} \right| \leq \sum_{j}^{}p_{j}|u_{j}| \leq \max_{j}|u_{j}|. \tag{E-8}$$

**证明。**非负概率加权满足三角不等式，且每项绝对值不超过最大值。证毕。

**定理E.1。** 对合法转移核和固定合法策略，

$$\parallel T_{P}^{\pi}V - T_{P}^{\pi}V^{(2)} \parallel_{\infty} \leq \gamma \parallel V - V^{(2)} \parallel_{\infty}. \tag{E-9}$$

**证明。** 相减后即时奖励抵消，对任意$s$，

$$\begin{aligned}
 & |(T_{P}^{\pi}V)(s) - (T_{P}^{\pi}V^{(2)})(s)| \\
 & = \gamma\left| \sum_{a}^{}\pi(a \mid s)\sum_{s'}^{}P(s' \mid s,a)\lbrack V(s') - V^{(2)}(s')\rbrack \right| \\
 & \leq \gamma\sum_{a}^{}\pi(a \mid s)\sum_{s'}^{}P(s' \mid s,a)|V(s') - V^{(2)}(s')| \\
 & \leq \gamma \parallel V - V^{(2)} \parallel_{\infty}.
\end{aligned} \tag{E-10}$$

取状态最大值得到结论。证毕。

由于E.1已经保证${\widehat{P}}_{R}$合法，将$P$替换为${\widehat{P}}_{R}$便得到残差模型的收缩性质。它不需要残差拟合误差为零，也不要求残差来自某种特定回归算法。

固定策略后的状态核仍非负且归一化，期望奖励仍满足式(E-2)。由有限维空间的完备性、定理E.1及收缩映射不动点定理，存在唯一价值固定点$V_{P}^{\pi}$，精确迭代$V_{n + 1} = T_{P}^{\pi}V_{n}$满足

$$\parallel V_{n} - V_{P}^{\pi} \parallel_{\infty} \leq \gamma^{n} \parallel V_{0} - V_{P}^{\pi} \parallel_{\infty}. \tag{E-11}$$

进一步，对固定点方程取绝对值并使用奖励界，得到

$$\parallel V_{P}^{\pi} \parallel_{\infty} \leq R_{\max} + \gamma \parallel V_{P}^{\pi} \parallel_{\infty}. \tag{E-12}$$

移项并除以$1 - \gamma > 0$，有

$$\parallel V_{P}^{\pi} \parallel_{\infty} \leq \frac{R_{\max}}{1 - \gamma}. \tag{E-13}$$

该上界由奖励有界性与折扣条件共同决定，对所有状态成立。

**E.3 误差关系。**对两个有限概率分布$p,q$，定义

$$TV(p,q) = \frac{1}{2}\sum_{j}^{}|p_{j} - q_{j}|. \tag{E-14}$$

**引理E.2。** 对任意有界函数$f$，

$$\left| \sum_{j}^{}(p_{j} - q_{j})f_{j} \right| \leq 2 \parallel f \parallel_{\infty}TV(p,q). \tag{E-15}$$

**证明。** 使用三角不等式，

$$\left| \sum_{j}^{}(p_{j} - q_{j})f_{j} \right| \leq \sum_{j}^{}|p_{j} - q_{j}|\,|f_{j}| \leq \parallel f \parallel_{\infty}\sum_{j}^{}|p_{j} - q_{j}|. \tag{E-16}$$

代入总变差定义即得。证毕。系数2不能在奖励可以正负取值的这一论证中直接省略。

设目标模型$P_{tgt}$与残差模型${\widehat{P}}_{R}$共享奖励$R$和折扣$\gamma$。定义

$$\varepsilon_{P} = \max_{s,a}TV\left( P_{tgt}( \cdot \mid s,a),{\widehat{P}}_{R}( \cdot \mid s,a) \right). \tag{E-17}$$

**定理E.2。** 对任意固定合法策略$\pi$，

$$\parallel V_{tgt}^{\pi} - {\widehat{V}}_{R}^{\pi} \parallel_{\infty} \leq \frac{2\gamma R_{\max}}{(1 - \gamma)^{2}}\varepsilon_{P}. \tag{E-18}$$

**证明。** 记价值差$\varepsilon_{V} = \parallel V_{tgt}^{\pi} - {\widehat{V}}_{R}^{\pi} \parallel_{\infty}$。由引理E.2及价值上界，

$$\begin{aligned}
 & \parallel T_{P_{tgt}}^{\pi}V_{tgt}^{\pi} - T_{{\widehat{P}}_{R}}^{\pi}V_{tgt}^{\pi} \parallel_{\infty} \\
 & \quad\quad \leq 2\gamma \parallel V_{tgt}^{\pi} \parallel_{\infty}\varepsilon_{P} \leq \frac{2\gamma R_{\max}}{1 - \gamma}\varepsilon_{P}.
\end{aligned} \tag{E-19}$$

在两个固定点之间加减$T_{{\widehat{P}}_{R}}^{\pi}V_{tgt}^{\pi}$，再结合式(E-9)与式(E-19)，得到

$$\begin{aligned}
\varepsilon_{V} & \leq \parallel T_{P_{tgt}}^{\pi}V_{tgt}^{\pi} - T_{{\widehat{P}}_{R}}^{\pi}V_{tgt}^{\pi} \parallel_{\infty} \\
 & \quad + \parallel T_{{\widehat{P}}_{R}}^{\pi}V_{tgt}^{\pi} - T_{{\widehat{P}}_{R}}^{\pi}{\widehat{V}}_{R}^{\pi} \parallel_{\infty} \\
 & \leq \frac{2\gamma R_{\max}}{1 - \gamma}\varepsilon_{P} + \gamma\varepsilon_{V}.
\end{aligned} \tag{E-20}$$

将式(E-20)中的$\gamma\varepsilon_{V}$移至左侧，再除以$1 - \gamma > 0$，得到式(E-18)。固定策略对动作的概率加权不放大式(E-17)的统一总变差界，因此该论证同样适用于状态动作转移核。证毕。

该定理说明：固定模型上的贝尔曼迭代误差可以随迭代消失，但模型相对目标环境的误差仍由$\varepsilon_{P}$控制。算子收缩和模型正确性不能相互替代。

若奖励也不同，设两模型的奖励绝对值均不超过$R_{\max}$，并令

$$\varepsilon_{r} = \max_{s,a}|R_{tgt}(s,a) - \widehat{R}(s,a)|. \tag{E-21}$$

对同一价值函数，两种模型的一步贝尔曼更新还会因奖励不同而产生至多$\varepsilon_{r}$的误差。沿用前述分解，有

$$\varepsilon_{V} \leq \varepsilon_{r} + \frac{2\gamma R_{\max}}{1 - \gamma}\varepsilon_{P} + \gamma\varepsilon_{V}, \tag{E-22}$$

从而

$$\varepsilon_{V} \leq \frac{\varepsilon_{r}}{1 - \gamma} + \frac{2\gamma R_{\max}}{(1 - \gamma)^{2}}\varepsilon_{P}. \tag{E-23}$$

式(E-23)将奖励误差与转移误差对固定策略价值的影响分别列出，对应正文第6.3.4节的误差分析。

**无折扣有限事件时域。** 对第5、6章的公交任务，将运营时刻、待决策事件及终止信息纳入状态，并将终止后补齐为零奖励吸收状态。本补充允许可测状态动作空间，假定策略、奖励和转移核均可测，总变差定义为可测集合上的概率差上确界。两概率测度之差的全变差质量等于这一上确界的两倍，故式(E-15)的有界函数估计仍适用于积分形式。比较模型共用状态、动作及终止定义；在对两模型均封闭的分析范围内，另假设每条路径至多有$K_{\max}$个决策事件，各模型每事件奖励绝对值不超过$R_{\max}$，期望即时奖励的统一误差不超过$\varepsilon_{r}$，后继事件、运行状态、事件间隔和终止信息的联合转移核总变差误差不超过$\varepsilon_{P}$。对同一固定策略，令剩余$n_{rem}$个事件的价值误差为

$$e_{n_{rem}} = \parallel V_{n_{rem}}^{\pi} - {\widehat{V}}_{n_{rem}}^{\pi} \parallel_{\infty},\quad\quad e_{0} = 0. \tag{E-23a}$$

在两个有限时域递推式之间加入并减去使用同一后继价值的中间项。即时奖励贡献至多$\varepsilon_{r}$，后继价值差贡献至多$e_{n_{rem}}$；剩余$n_{rem}$个事件的价值绝对值不超过$n_{rem}R_{\max}$，由式(E-15)得到联合转移误差项。因此

$$e_{n_{rem} + 1} \leq \varepsilon_{r} + e_{n_{rem}} + 2n_{rem}R_{\max}\varepsilon_{P}. \tag{E-23b}$$

从$e_{0} = 0$开始对$n_{rem} = 0,\ldots,K_{\max} - 1$逐项求和，使用$2\sum_{n_{rem} = 0}^{K_{\max} - 1}n_{rem} = K_{\max}(K_{\max} - 1)$，得到

$$\parallel V_{K_{\max}}^{\pi} - {\widehat{V}}_{K_{\max}}^{\pi} \parallel_{\infty} \leq K_{\max}\varepsilon_{r} + K_{\max}(K_{\max} - 1)R_{\max}\varepsilon_{P}. \tag{E-23c}$$

该推导不使用严格折扣，且把事件时间和结束条件包含在转移中。$K_{\max}$是决策事件数上界，不是24,000秒运营时长；同时到站车辆的序贯选择即使不推进物理时间，也占据决策事件。现有预测均方误差不是上述联合核的总变差，已有实验也未估计统一的$K_{\max}$、奖励界和转移误差界。因此式(E-23c)用于说明有限时域的误差传播条件，不作为已经验证的公交数值性能保证，也不将$\gamma = 1$代入式(E-23)。

回到本附录前述有限状态设定，还可以只在状态子集$\mathcal{C \subseteq S}$上比较模型。此时，奖励与转移误差界只需覆盖$s\mathcal{\in C}$及策略赋予正概率的动作，但两种模型在这些状态动作下都必须以概率1留在$\mathcal{C}$。贝尔曼算子因而可限制在仅记录$\mathcal{C}$内各状态价值的有限维空间上，前述证明继续成立，结论中的范数相应改为$\parallel \cdot \parallel_{\mathcal{\infty,C}}$。

这一局部结论要求分析范围对两种模型均封闭。仅列出目标策略在某些运行轨迹中访问过的状态，既不能保证近似模型也留在其中，也不足以控制其余状态的最大价值误差。

例如，取两个状态0、1，只有一个动作，初始状态为0，奖励为$R(0) = 0,R(1) = 1$。目标模型令两个状态各自吸收，近似模型令状态0吸收、状态1下一步转移到0。两模型在从0开始的全部访问轨迹上完全相同，访问范围内的转移误差为零。然而

$$V_{tgt}(1) = \frac{1}{1 - \gamma},\quad\quad\widehat{V}(1) = 1, \tag{E-24}$$

所以当$0 < \gamma < 1$时，全局价值误差至少为$\gamma/(1 - \gamma) > 0$。该例表明，仅控制已访问状态上的模型误差，不能得到覆盖全部状态的最大范数误差界；还需对其余状态建立相应的误差条件。

若一列近似模型在同一固定策略、同一奖励界及同一封闭分析范围上满足$\varepsilon_{P,n} \rightarrow 0$和$\varepsilon_{r,n} \rightarrow 0$，则上述界给出

$$\parallel V_{tgt}^{\pi} - {\widehat{V}}_{n}^{\pi} \parallel_{\infty} \rightarrow 0. \tag{E-25}$$

**证明。** 价值误差非负，并由式(E-23)中两个非负项之和控制。固定折扣条件下，两项的系数均为有限常数；当奖励误差和转移误差分别趋于零时，上界趋于零。由夹逼定理得到式(E-25)。证毕。

**E.4 数据覆盖。**模型在数据采集策略下准确，不等于对任意新策略准确。下面给出一个有限且可直接计算的例子。

设状态为$\{ 0,1\}$，动作为$\{ a_{0},a_{1}\}$，初始状态为0，奖励为$R(0,a) = 0$、$R(1,a) = 1$，$0 < \gamma < 1$。目标模型在状态0执行$a_{0}$时留在0，执行$a_{1}$时转移到1；状态1在任一动作下都吸收。近似模型在状态0执行任一动作都留在0，状态1也吸收。

数据采集策略始终选择$a_{0}$，因此只能观察$(0,a_{0},0)$。在这一数据支持集上，近似模型与目标模型完全一致，增加同一策略下的样本数量不会暴露$(0,a_{1})$上的差异。

若新策略在状态0选择$a_{1}$，则目标环境在第一步之后获得持续奖励1，而近似模型始终停留在零奖励状态。因此

$$V_{tgt}^{\pi'}(0) = \sum_{k = 1}^{\infty}\gamma^{k} = \frac{\gamma}{1 - \gamma},\quad\quad{\widehat{V}}^{\pi'}(0) = 0. \tag{E-26}$$

该例说明，模型在已有数据支持集上的零误差，可以与新策略下的正价值误差同时存在。数据覆盖是把预测误差用于控制分析的重要条件。

这一结论与公交离线学习中的覆盖问题一致：只观察某类停站行为，并不足以辨识所有其他停站行为的运行后果。它指出的是数据条件，而不是否定利用仿真补充经验的价值。

**E.5 更新条件。**在式(E-6)的概率约束下，将合法转移残差集合写为闭集，并直接分析保持该集合的更新映射：

$$\begin{aligned}
X_{\delta} & = \mathbb{R}^{|S|\,|A|\,|S|}, \\
\mathcal{C}_{\delta} & = \{\delta \in X_{\delta}:P_{sim} + \delta \geq 0,\ \sum_{s'}^{}\delta(s,a,s') = 0\}, \\
F & :\mathcal{C}_{\delta} \rightarrow \mathcal{C}_{\delta}.
\end{aligned} \tag{E-27}$$

其中，$X_{\delta}$是按各状态动作对及其后继状态排列残差分量所得的有限维实向量空间，$\mathcal{C}_{\delta}$是使组合转移概率合法的残差集合。非负概率与各行残差和为0构成闭约束；零残差属于该集合，因此$\mathcal{C}_{\delta}$非空，并在最大范数下完备。$F(\delta)$表示由当前残差所对应的策略、数据条件及下一次模型拟合共同确定的更新，假定$F$将合法残差仍映射为合法残差。范数仍为

$$\parallel \delta \parallel_{\infty} = \max_{s,a,s'}|\delta(s,a,s')|. \tag{E-28}$$

若直接假设

$$\parallel F(\delta_{1}) - F(\delta_{2}) \parallel_{\infty} \leq L \parallel \delta_{1} - \delta_{2} \parallel_{\infty},\quad\quad 0 \leq L < 1, \tag{E-29}$$

则$F$是收缩映射。这一前提与贝尔曼算子的收缩前提不同。前者需要描述策略更新、数据分布变化和回归优化共同形成的映射，后者只需要固定合法随机核与折扣。

**定理E.3。** 若$F$在上述合法残差闭集上满足收缩条件，则存在唯一$\delta^{*}$满足$F(\delta^{*}) = \delta^{*}$。迭代$\delta_{n + 1} = F(\delta_{n})$满足

$$\parallel \delta_{n} - \delta^{*} \parallel_{\infty} \leq L^{n} \parallel \delta_{0} - \delta^{*} \parallel_{\infty}. \tag{E-30}$$

**证明。**非空闭集$\mathcal{C}_{\delta}$完备，$F$保持该集合并满足式(E-29)，由收缩映射不动点定理得到唯一固定点；反复比较迭代与固定点即得式(E-30)。证毕。

这是以完整残差更新收缩为前提的条件结论，不是对实际MC-WM拟合过程已经满足该条件的证明。唯一固定点也不必等于真实转移差，识别真实残差仍需数据可辨识性及拟合一致性。

若残差更新由策略调整、访问分布变化及模型再拟合复合而成，可分别以 $L_{adj}$、$L_{occ}$ 和 $B$ 描述其敏感性。要应用收缩定理，仍需证明完整映射满足

$$\parallel F(\delta_{1}) - F(\delta_{2}) \parallel_{\infty} \leq L_{adj}L_{occ}B \parallel \delta_{1} - \delta_{2} \parallel_{\infty} \tag{E-31}$$

且 $L_{adj}L_{occ}B < 1$。这一乘积形式要求各组件之间具有可复合的Lipschitz关系。若实际策略优化或回归步骤不满足其中某项条件，则应直接分析完整更新映射，不能仅凭组件名称确定其收缩系数。

**E.6 置信排序。**本节研究一个独立的静态问题：一个按预测误差构造的正值分数，与一个二分类后验分数，何时会对同一批输入给出相同排序。以下两类分数由式(E-32)、式(E-33)指定，不分别指代实际MC-WM权重和H2O+条件转移比。

在比较域$\mathcal{X}$上，给定误差函数$e_{real}(x)$、$e_{resid}(x)$，正值分数$w_{rank}(x) > 0$及分类分数$0 < \varphi_{cls}(x) < 1$。下标real与resid在本节区分两个比较模型，不表示本节新增了真实公交运行数据。$\tau_{rank} > 0$为分数尺度，$c_{rank}$为加性常数，$\varepsilon_{rank}$为后续采用的误差波动界。

与此不同，第6章的MC-WM先计算$1/(1 + e/\tau)$，再递推后继置信形成样本权重；H2O+则由两个分类器估计条件转移比。静态排序结论不能直接替代这两种实际机制的分析。以下假设存在$\tau_{rank} > 0$和常数$c_{rank}$，使

$$\mathcal{l}_{w_{rank}}(x): = \log w_{rank}(x) = - \frac{e_{real}(x)}{\tau_{rank}} + c_{rank}, \tag{E-32}$$

$$\mathcal{l}_{\varphi_{cls}}(x): = \log\frac{\varphi_{cls}(x)}{1 - \varphi_{cls}(x)} = \frac{e_{resid}(x) - e_{real}(x)}{\tau_{rank}}. \tag{E-33}$$

因此

$$\mathcal{l}_{\varphi_{cls}}(x) - \mathcal{l}_{w_{rank}}(x) = \frac{e_{resid}(x)}{\tau_{rank}} - c_{rank}. \tag{E-34}$$

式(E-32)、式(E-33)是以下结论的指定前提。其一种概率解释为：对同一个$d$维观测$y$，两类条件密度分别是均值为$\mu_{real}$、$\mu_{resid}$且协方差均为$\sigma^{2}I_{d}$的高斯密度，类别先验相等，$\varphi_{cls}$表示“real”类的后验概率；取$e_{real} = \parallel y - \mu_{real} \parallel_{2}^{2}$、$e_{resid} = \parallel y - \mu_{resid} \parallel_{2}^{2}$及$\tau_{rank} = 2\sigma^{2}$，对数密度比即为式(E-33)。若误差定义为逐维均方值，尺度应改为$2\sigma^{2}/d$；协方差不同时还需保留行列式及相应二次型项。式(E-32)来自另行指定的指数信号，不由上述分类解释自动推出。这里$I_{d}$为$d$阶单位矩阵。

**引理E.3。** 对任意$x_{1},x_{2}$，

$$w_{rank}(x_{1}) < w_{rank}(x_{2})\quad \Leftrightarrow \quad e_{real}(x_{2}) < e_{real}(x_{1}). \tag{E-35}$$

**证明。** 本段简写下标1、2分别表示在两个比较点上的取值。正数上的对数严格递增，故${w_{rank}}_{1} < {w_{rank}}_{2}$等价于$\log{w_{rank}}_{1} < log{w_{rank}}_{2}$。代入第一条等式，消去$c_{rank}$，并乘以$\tau_{rank} > 0$，得到$- e_{real,1} < - e_{real,2}$，即所需结论。各步均可逆，得到等价关系。证毕。

**定理E.4。** 若存在常数$e_{const}$，使$e_{resid}(x) = e_{const}$对所有$x$成立，则

$$w_{rank}(x_{1}) < w_{rank}(x_{2})\quad \Leftrightarrow \quad\varphi_{cls}(x_{1}) < \varphi_{cls}(x_{2}). \tag{E-36}$$

**证明。** 由前提，$\mathcal{l}_{\varphi_{cls}}(x) = \mathcal{l}_{w_{rank}}(x) + e_{const}/\tau_{rank} - c_{rank}$，两种对数分数只相差常数。于是它们对任意点对的严格排序相同。

还需说明$\mathcal{l}_{\varphi_{cls}}$与$\varphi_{cls}$本身排序相同。对$0 < u,v < 1$，由于分母为正，

$$\frac{u}{1 - u} < \frac{v}{1 - v}\quad \Leftrightarrow \quad u(1 - v) < v(1 - u)\quad \Leftrightarrow \quad u < v. \tag{E-37}$$

再使用对数严格递增即可。因此$w_{rank}$与$\varphi_{cls}$排序一致。证毕。

该结论是排序等价，不是数值相等或概率校准等价。两个信号可以有不同尺度，同时保持相同排序。

**定理E.5。** 若存在$\overline{e}$和$\varepsilon_{rank} \geq 0$，使

$$|e_{resid}(x) - \overline{e}| \leq \varepsilon_{rank}\quad\text{对所有 }x\text{ 成立}, \tag{E-38}$$

则对满足

$$|logw_{rank}(x_{1}) - logw_{rank}(x_{2})| > \frac{2\varepsilon_{rank}}{\tau_{rank}} \tag{E-39}$$

的点对，$w_{rank}$与$\varphi_{cls}$的严格排序一致。

**证明。** 定义

$$\Delta e = e_{real}(x_{1}) - e_{real}(x_{2}),\quad\quad\Delta e_{resid} = e_{resid}(x_{1}) - e_{resid}(x_{2}). \tag{E-40}$$

由三角不等式有$|\Delta e_{resid}| \leq 2\varepsilon_{rank}$。由两种信号的定义，

$$\mathcal{l}_{w_{rank}}(x_{1}) - \mathcal{l}_{w_{rank}}(x_{2}) = - \Delta e/\tau_{rank}, \tag{E-41}$$

$$\mathcal{l}_{\varphi_{cls}}(x_{1}) - \mathcal{l}_{\varphi_{cls}}(x_{2}) = (\Delta e_{resid} - \Delta e)/\tau_{rank}. \tag{E-42}$$

间隔条件等价于$|\Delta e| > 2\varepsilon_{rank}$。若$\Delta e > 2\varepsilon_{rank}$，则$- \Delta e < 0$，且$\Delta e_{resid} - \Delta e \leq 2\varepsilon_{rank} - \Delta e < 0$，两种对数分数的差都为负。若$\Delta e < - 2\varepsilon_{rank}$，则$- \Delta e > 0$，且$\Delta e_{resid} - \Delta e \geq - 2\varepsilon_{rank} - \Delta e > 0$，两个差都为正。零情形被严格间隔条件排除。

因此两种对数分数的差具有同一符号。分别使用对数和对数几率的严格单调性，得到$w_{rank}$与$\varphi_{cls}$排序一致。证毕。

该定理没有对间隔不超过$2\varepsilon_{rank}/\tau_{rank}$的点对作出保证。误差变化较大时，两种分数可能给出不同排序。因此，部分输入上的排序一致，并不意味着两种分数在全部输入上都能相互替代。。

**E.7 理论解释。**合法残差核保持了固定策略评价的收缩结构，但所得固定点是否接近目标价值，还取决于模型与目标环境之间的误差。因而，源结构的保留、目标数据拟合和策略评价承担不同作用：源结构提供先验表达，目标数据修正分布差异，策略评价检验这些修正的控制效果。

第6章使用标准化一步预测误差比较模型类别，并通过一步模型样本参与策略更新。预测误差与转移核总变差不是同一度量；把前者代入后者的价值界，需要额外建立概率模型与误差度量之间的关系。本附录的有限模型分析提供误差分解，连续动力学中的实际适应效果由相应实验考察。

E.6研究的是指数型误差关系下的静态排序。第6章的置信评价还包含后继置信度递推更新，二者具有不同时间结构。静态排序结论用于说明分数一致性的条件，不能直接替代动态置信评价的性能分析。E.9进一步分析固定传播关系下归一化对数置信的尺度变化。

**E.8 适用范围。**价值误差界要求误差控制覆盖被评价的状态空间，或者限定在对两种模型均封闭的状态子集内。采集策略的访问集合未必覆盖新策略可能到达的状态；E.4的反例说明，仅在原有数据上降低预测误差不足以保证全局价值误差降低。

残差拟合的收敛分析以完整更新映射满足收缩条件为前提。策略、数据分布和模型参数共同变化时，各部分的正则性必须能够组合为该条件，单独一次拟合误差下降并不能给出固定点收敛结论。

置信排序分析要求正值信号、指定的对数关系及相应误差间隔。上述关系需要由所采用的概率模型或信号定义得到；对任意评分函数并不成立。因此，条件排序等价、预测风险降低和控制收益改善应分别论证。

**E.9 尺度分析。**第6.3.2节改变即时对数置信的系数。本节固定策略、局部置信函数及传播关系，分析这一改变对精确评价固定点的作用。令$x = (s,a)$，$c(x) \in \lbrack c_{\min}^{\Delta},1\rbrack$为局部置信，其中$c_{\min}^{\Delta} > 0$是给定下界；$K_{\Delta}^{\pi}(x' \mid x)$为策略$\pi$下包含非终止条件的置信传播核，满足非负性及$\sum_{x'}^{}K_{\Delta}^{\pi}(x' \mid x) \leq 1$。不足1的概率质量表示终止后不再传播置信。状态动作集合$\mathcal{S} \times \mathcal{A}$取为有限集，$Q^{\Delta}$表示其上的任意实值置信评价函数，定义原算子与归一化算子

$$\begin{aligned}
(T_{\Delta}^{\pi}Q^{\Delta})(x) & = logc(x) + \gamma_{\Delta}\sum_{x'}^{}K_{\Delta}^{\pi}(x' \mid x)Q^{\Delta}(x'), \\
({\widetilde{T}}_{\Delta}^{\pi}Q^{\Delta})(x) & = (1 - \gamma_{\Delta})logc(x) + \gamma_{\Delta}\sum_{x'}^{}K_{\Delta}^{\pi}(x' \mid x)Q^{\Delta}(x'),
\end{aligned} \tag{E-43}$$

其中$0 \leq \gamma_{\Delta} < 1$，$T_{\Delta}^{\pi}$和${\widetilde{T}}_{\Delta}^{\pi}$分别作用于原对数置信与归一化对数置信。

**命题E.1。** （置信尺度归一化）两算子均为无穷范数下的$\gamma_{\Delta}$收缩，具有唯一固定点$Q^{\Delta,*}$和${\widetilde{Q}}^{\Delta,*}$，且

$$\begin{aligned}
{\widetilde{Q}}^{\Delta,*} & = (1 - \gamma_{\Delta})Q^{\Delta,*}, \\
exp({\widetilde{Q}}^{\Delta,*}) & = \lbrack exp(Q^{\Delta,*})\rbrack^{1 - \gamma_{\Delta}}.
\end{aligned} \tag{E-44}$$

**证明。** 对任意有界置信评价函数$Q_{1}^{\Delta},Q_{2}^{\Delta}$，即时项在算子相减时抵消。利用传播核非负且行和不超过1，得到

$$\parallel {\widetilde{T}}_{\Delta}^{\pi}Q_{1}^{\Delta} - {\widetilde{T}}_{\Delta}^{\pi}Q_{2}^{\Delta} \parallel_{\infty} \leq \gamma_{\Delta} \parallel Q_{1}^{\Delta} - Q_{2}^{\Delta} \parallel_{\infty}. \tag{E-45}$$

原算子同理；行和小于1只会减弱后继项的传播，不影响这一收缩界。两个算子均作用于有限维完备空间，因此分别存在唯一固定点。将原固定点等式两侧乘以1−$\gamma_{\Delta}$，并利用有限概率加权对评价函数的线性性，得到

$$\begin{aligned}
 & (1 - \gamma_{\Delta})Q^{\Delta,*}(x) \\
 & = (1 - \gamma_{\Delta})logc(x) + \gamma_{\Delta}\sum_{x'}^{}K_{\Delta}^{\pi}(x' \mid x)(1 - \gamma_{\Delta})Q^{\Delta,*}(x').
\end{aligned} \tag{E-46}$$

因此$(1 - \gamma_{\Delta})Q^{\Delta,*}$也是归一化算子的固定点，由唯一性得到式(E-44)的第一式，再取指数得到第二式。证毕。

若无终止且局部置信恒定为$c_{const}$，传播核行和为1，常数函数$Q^{\Delta,*} = logc_{const}/(1 - \gamma_{\Delta})$和${\widetilde{Q}}^{\Delta,*} = logc_{const}$分别满足两固定点方程。原权重因而为$c_{const}^{1/(1 - \gamma_{\Delta})}$，归一化权重为$c_{const}$，得到正文的尺度解释。

例如无终止、局部评分恒为0.5且$\gamma_{\Delta} = 0.9$时，原精确权重为${0.5}^{10} \approx 0.00098$，归一化权重为0.5。该例解释的是更新强度的尺度差。第6章同时设置固定0.5样本权重的对照，用于区分权重整体增大与状态相关加权的作用；相应收益不能全部解释为状态排序更加准确。

函数$z \mapsto z^{1 - \gamma_{\Delta}}$在正实数上严格递增，故两种精确权重保持相同排序。归一化没有据此获得更准确的概率解释，也不保证控制回报提高；它改变的是该评分进入价值损失时的数值强度。实际训练中策略和预测误差随更新变化，两个神经网络也未必达到精确固定点，因此本命题用于解释第6.3.2节的改造方向，控制效果另由第6.4.2至6.4.3节的配对实验给出。

# 附录F 公交世界模型的补充实验

本附录补充共同离线初始化下的公交策略续训比较、最终控制器比较和源仿真标定偏差试验。第F.1与F.3节报告单次训练下的探索结果，第F.2节比较五次训练得到的完整控制器，各方法使用相同的源经验生成条件和驻站执行规则，但保留各自的训练更新安排。正文第6.4.1至第6.4.3节的多次训练比较构成主要公交实验依据；各批实验的数据复用及评价数量见第F.4节。

**F.1 共同离线初始化下的续训比较。**第一组比较采用同一个完成50,000次离线更新的策略网络，分别固定该策略网络参数而不再训练、仅以目标离线数据继续训练、以目标离线数据和源仿真混合训练，以及加入世界模型辅助训练。三种继续训练方案各增加2,000次策略更新和2,500次价值更新；两种使用源经验的方案共用41,039条源仿真转移。每种方法在需求倍率1.0和1.25下各评价两个共同场景组，共16个24,000秒SUMO回合。

表F-1显示，人均广义时间成本的主要下降已经出现在纯离线续训阶段。H2O+混合续训相对纯离线续训再减少1.24秒，世界模型辅助相对H2O+再减少3.31秒。后一个差值按四组场景分别计算百分比后平均为0.134%，其中两组改善、两组退化。

表F-1 共同离线初始化下的公交闭环结果

|         控制方法         | 人均广义时间成本（秒） |
|:------------------------:|:----------------------:|
| 仅离线预训练，不继续更新 |        2719.39         |
|        纯离线续训        |        2693.27         |
|       H2O+混合续训       |        2692.03         |
|       世界模型辅助       |        2688.72         |

图F-1 比较共同离线初始化之后的续训方式，展示乘客时间的变化。

同色点线连接同一场景组中四种控制器的结果，表示配对比较关系，不表示一条连续训练轨迹。人均广义时间成本越低越好。该组采用一个训练种子、两个共同场景组、SUMO 1.25.0与各组共用且保持不变的源经验数据。

该组试验首先检查世界模型能否与现有公交训练及控制流程共同运行。从仅离线预训练到世界模型辅助方案的总改善，也包含继续训练带来的影响；模型辅助的增量应相对于更新安排匹配的无世界模型方案评价。多次独立训练的比较见第6.4.1节。

![](appendix_figures_20261016/fig_F_1.png)

图F-1 共同离线初始化下不同续训方式的乘客服务结果

**F.2 最终控制器比较。**本节比较MC-WM、H2O+和纯源SAC的最终控制效果。三组使用相同的源仿真生成规则与需求轮换计划，均计划采集约25,000个源事件，并保留五次训练结束时的策略。MC-WM与H2O+在每个对应种子下共用完成50,000次IQL更新的离线初始化，随后继续训练20,000次；纯源SAC从随机网络参数开始，仅使用源仿真经验。MC-WM采用归一化置信度递推更新，模型损失系数为0.25。三种方法各自的完整更新安排保持不变，因此相同源交互量不代表总训练计算量相同。源仿真依次按0.75、1.0和1.25倍需求轮换；受采样量限制，本次三组均只运行到前两个源运行日，实际源事件数与计划值相差不超过1次。

所有策略均在目标SUMO环境中评价，纯源仅指训练数据来源。评价采用五个新运行日与0.75、1.0和1.25倍需求，三种方法各含五个训练种子，共225回合。每回合在SUMO 1.25.0中运行24,000秒，同一运行日和需求下共用乘客输入；训练期间不追加目标SUMO交互。三组均采用第5.3.4节的共同附加驻站执行规则，且不叠加额外的0.25动作缩放。

图F-2同时展示人均广义时间成本及方法间的配对差异。统计先在每个训练种子内等权平均十五个测试情形，再汇总五个种子。人均广义时间成本由评价时域内全部已出发乘客的两倍累计候车时间与累计车内时间之和除以其人数得到；未完成行程者已经发生的时间仍计入成本。

![](appendix_figures_20261016/fig_F_2.png)

图F-2 共同源采样与执行条件下最终控制器的公交服务表现

空心圆为各训练种子的均值，菱形为五个种子的总体均值，误差线为基于五个训练种子的95% t置信区间。图（b）先在同一种子内计算前一种方法减后一种方法的时间差，再汇总五个配对差；负值表示前者更优，区间跨零表示尚不能确定稳定优势。225回合不作为225个独立训练重复。

MC-WM、H2O+和纯源SAC的人均广义时间成本分别为2,360.49、2,383.51和2,400.74秒。MC-WM相对纯源SAC平均减少40.25秒（1.68%），五个训练种子均改善，配对95%区间为−65.35至−15.16秒。相对H2O+，MC-WM平均减少23.02秒（0.97%），三个种子改善、两个种子变差，区间为−69.23至23.19秒。H2O+相对纯源SAC平均减少17.23秒，四个种子改善，区间为−53.51至19.04秒。

MC-WM相对H2O+的候车时间减少13.04秒，乘车时间增加3.05秒，因而人均广义时间成本净减少23.02秒。在1.25倍需求下，MC-WM相对H2O+和纯源SAC的平均时间分别减少42.55秒和78.44秒，表明本组收益主要体现在候车负担较重的情形。局限在于训练重复仅有五个种子，区间未作多重比较调整；纯源SAC不含离线预训练开销，各方法的训练日程也不完全相同。因此，该比较支持MC-WM相对纯源的控制收益及相对H2O+的平均改善，尚不足以认定其稳定优于H2O+，也不能把全部差异归因于世界模型单一部件。

**F.3 单次训练下的标定偏差试验。**仿真标定误差涉及状态转移规律。本节使用6个完整目标场景组的离线数据，并固定16,384次计划源交互，分别改变道路平均速度、OD强度、新增信号路口数量及必要上下客服务时间，再设置联合偏差。速度偏差25%表示速度乘以0.75，OD和服务时间偏差25%表示相应量乘以1.25。经核对，新增路口沿用第6.4.1节的泊松计数规则：均值为新增路口数量的比例参数乘以线路既有信号路口数，再除以仿真路段数，沿用既有绿灯比例和信号周期产生延误。联合偏差50%同时使用0.5倍速度、1.5倍OD和必要服务时间，以及0.5的新增路口数量比例参数。必要服务时间不同于策略选择的附加驻站动作。目标SUMO保持原设置。本节保留归一化置信度改进前的单训练结果，训练预算、源采样和目标测试日均与第F.2节不同。

另设去除经验运行时间标定的条件，以原有速度、加速度和信号动力学替代经验路段运行时间采样。它考察经验运行时间分布的作用，仍保留原有路网、OD和上下客输入。OD偏差采用矩阵强度统一缩放；联合偏差50%的物理客流为1.5倍，其预计到达时间（Estimated Time of Arrival，ETA）观测采用已有1.25倍客流的速度模板。

每种偏差分别给出正常和1.25倍目标需求的结果。图正值表示世界模型降低人均广义时间成本；蓝色为改善，红色为退化。每格均为同一目标场景组内的配对比较。

表F-2 给出图F-3 的数值。包含已标定参照的16组需求与偏差组合中，世界模型在8组降低人均广义时间成本，降幅为0.068%至0.797%；其余8组增加0.088%至1.198%。必要服务时间增加25%和联合偏差25%在两档需求下均有时间改善；OD偏差、额外路口偏差及去除经验运行时间标定三类条件下，两档需求均未改善。

联合偏差由25%增至50%时，正常需求的时间收益由0.295%转为-0.392%；去除经验运行时间标定时，世界模型增加27.11秒。因而，收益并未随名义偏差单调扩大。必要服务时间误差主要改变上下客时长，OD和信号误差还影响队列及事件顺序，这为差异提供了可能解释，但该单种子比较未分离各转移分量的控制作用。

![](appendix_figures_20261016/fig_F_3.png)

图F-3 不同源转移偏差下世界模型辅助相对H2O+的服务变化（单次训练）

表F-2 世界模型辅助相对H2O+的时间变化（单次训练）

| 源转移条件 | 正常需求下人均广义时间成本降幅（%） | 1.25倍需求下人均广义时间成本降幅（%） |
|:--------------------------------:|:-----------------:|:-----------------:|
| 已标定 | 0.068 | 0.797 |
| 速度偏差25% | -0.115 | 0.349 |
| OD偏差25% | -0.677 | -0.141 |
| 新增路口数量比例参数为0.25 | -0.363 | -0.703 |
| 必要服务时间增加25% | 0.302 | 0.682 |
| 联合偏差25% | 0.295 | 0.555 |
| 联合偏差50% | -0.392 | 0.437 |
| 去除经验标定 | -1.198 | -0.088 |

**F.4 实验记录。**完整混合训练过程中的模型辅助比较分两批开展，均见第6.4.1节。第一批比较两种方法、三档源标定条件与三个训练种子，共18组训练、180个SUMO评价回合；第二批增加混合损失系数为0.75的无模型参照，重新执行全部三种方法，共27组训练、270个新评价回合。两批复用相同的三个离线初始化种子，评价运行日分开，不能合并成六个独立训练种子。新增无模型参照仅缩放混合经验损失，模型组仍使用0.75倍混合损失与0.25倍含置信权重的模型损失；九组模型训练的180个时序诊断点用于第6.4.1节的机制分析，不计作控制评价回合。

第6.3.3节及第6.4.2节针对累积置信尺度另行比较归一化置信度递推更新、固定权重及源目标修正。该批复用前两批中的两个离线初始化，保留原方法与无模型参照，共38次训练、380个新SUMO评价回合；随后在相同十个评价条件上补充两种既有规则，共20回合。规则补充不重新训练，也不增加训练种子数。归一化比较包含正常、联合50%及75%源失配，与此前正常、25%及50%的条件区分报告。

第6.4.3节另以20个新种子从头完成各50000次离线初始化，在正常标定、联合50%和75%源失配下比较无世界模型混合基线、两个固定模型损失系数的配置及按源仿真偏差调节模型损失系数。每个种子内四种方法共用该种子的初始化；各配置继续采集25,000个源事件，并完成20,000次训练更新，共240项策略续训。每个固定策略在两档需求、五个开发运行日和五个独立测试运行日上评价，共4800回合；两种规则另各评价20回合。统计先在每个训练种子内等权平均10个测试情形，再汇总20个种子。五个运行日分别在两档需求下使用，既不算十个独立运行日，也不作为额外训练重复。开发集只用于预先规定的最终策略选择，全部20个种子的测试结果均保留。

第6.4.3节的按源仿真偏差调节模型损失系数扩展比较复用上述前四个初始化及36项已训练对照，在三种源条件下新增两种配置、24项续训。五种配置共60个固定策略，在该分支另设的开发日和测试日上完成1200回合评价。该分支与20种子主比较使用不同的评价日，分别统计；复用四个初始化不增加独立初始化总数，分支也不复用主比较的规则测试成本。

策略网络和价值网络更新次数相同的适配比较按完整源运行日预先划分分类器拟合与留出集合：五日中的第二日留出，其余四日用于拟合。四个启用比率的混合配置均按这一划分重新训练，替换先前未正确形成源留出集的批次；纯源与纯离线组不受该划分影响，保留原结果。第5.4.5节使用其中同一批五个无模型混合策略，每个策略另生成三个独立源运行日，共15个诊断日。诊断期间冻结策略和分类器，不追加梯度更新；这些源轨迹用于检验分类与权重分布，不计入表6-2的SUMO服务评价数量，也不增加独立训练种子数。

第F.2节的最终控制器比较复用五个正式H2O+策略和五个纯源SAC策略，新增五次MC-WM继续训练。MC-WM复用H2O+对应种子的离线初始化；十五个固定策略在五个新运行日和三档需求下完成225个目标SUMO评价回合。该组使用独立生成的测试运行日，与前述20种子比较分别统计，复用策略不增加独立训练种子数。

第6.4.3节的车头时距奖励补充试验采用66001、66002两个初始化，在正常标定与联合75%源偏差下新增4次MC-WM训练，并复用对应4个H2O+策略。MC-WM的模型损失系数为0.125，混合经验损失系数为0.75。各策略在两档需求、每档五个既有评价日中，直接执行，新增MC-WM评价40回合，另复用H2O+评价40回合。该组与乘客时间目标下20次独立初始化的主要比较分别统计。
