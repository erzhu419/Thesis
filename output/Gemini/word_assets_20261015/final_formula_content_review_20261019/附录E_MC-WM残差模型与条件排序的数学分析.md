# 附录E MC-WM残差模型与条件排序的数学分析

[P4298] 附录E MC-WM残差模型与条件排序的数学分析

[P4299] 本附录围绕模型误差和置信尺度两条主线展开。第E.1至E.5节分析固定策略价值、数据覆盖及合法残差更新条件，第E.3节另给出无折扣有限事件时域的条件误差界；第E.6节只讨论指定静态误差信号的排序关系。第E.9节直接对应公交改进中的归一化置信度递推更新，解释预测误差稳定时权重仍可能持续减小的原因。

[P4300 M004527,M004528,M004529,M004530,M004531] E.1 基本条件。设$\mathcal{S}$、$\mathcal{A}$为有限非空集合。一个折扣决策过程由转移核$P$、奖励$R$和折扣因子$\gamma$构成，其中

[P4302 式(E-1) M004532] $P(s' \mid s,a) \geq 0,\quad\quad\sum_{s'\mathcal{\in S}}^{}P(s' \mid s,a) = 1,$

[P4303 式(E-1)] (E-1)

[P4305 式(E-2) M004533] $|R(s,a)| \leq R_{\max},\quad\quad R_{\max} \geq 0,\quad\quad 0 \leq \gamma < 1.$

[P4306 式(E-2)] (E-2)

[P4307 M004534] 策略$\pi$满足

[P4309 式(E-3) M004535] $\pi(a \mid s) \geq 0,\quad\quad\sum_{a}^{}\pi(a \mid s) = 1.$

[P4310 式(E-3)] (E-3)

[P4311] 策略函数的非负性与归一化是概率解释所需条件。固定策略后，贝尔曼评价算子为

[P4313 式(E-4) M004536] $(T_{P}^{\pi}V)(s) = \sum_{a}^{}\pi(a \mid s)\left\lbrack R(s,a) + \gamma\sum_{s'}^{}P(s' \mid s,a)V(s') \right\rbrack.$

[P4314 式(E-4)] (E-4)

[P4315 M004537,M004538] 价值函数空间为有限维实向量空间 $\mathbb{R}^{\mathcal{|S|}}$，范数为 $\parallel V \parallel_{\infty} = \max_{s}|V(s)|$。

[P4316 M004539,M004540] 给定源核$P_{sim}$及修正函数$\widehat{\delta}$，定义

[P4318 式(E-5) M004541] ${\widehat{P}}_{R}(s' \mid s,a) = P_{sim}(s' \mid s,a) + \widehat{\delta}(s,a,s').$

[P4319 式(E-5)] (E-5)

[P4320 M004542] 假设对每个$(s,a)$有

[P4322 式(E-6) M004543] $P_{sim}(s' \mid s,a) + \widehat{\delta}(s,a,s') \geq 0,\quad\quad\sum_{s'}^{}\widehat{\delta}(s,a,s') = 0.$

[P4323 式(E-6)] (E-6)

[P4324] 则

[P4326 式(E-7) M004544] $\sum_{s'}^{}{\widehat{P}}_{R}(s' \mid s,a) = \sum_{s'}^{}P_{sim}(s' \mid s,a) + \sum_{s'}^{}\widehat{\delta}(s,a,s') = 1.$

[P4327 式(E-7)] (E-7)

[P4328] 由式(E-6)的非负性与式(E-7)的归一化，式(E-5)给出的组合核是合法随机核。残差本身可以取负值，要求非负的是组合后的概率。将该组合核与给定奖励及折扣因子结合，即得到新的有限决策过程。

[P4329] 本节的残差是转移概率的加性修正。正文实现中的状态增量残差则是连续状态预测器的加性修正。二者共享“保留基础模型并修正差异”的思想，但数学对象不同。

[P4330] E.2 收敛性。

[P4331 M004545,M004546,M004547] 引理E.1。 若 $p_{j} \geq 0$ 且 $\sum_{j}^{}p_{j} = 1$，则任意有限实数序列 $u_{j}$ 满足

[P4333 式(E-8) M004548] $\left| \sum_{j}^{}p_{j}u_{j} \right| \leq \sum_{j}^{}p_{j}|u_{j}| \leq \max_{j}|u_{j}|.$

[P4334 式(E-8)] (E-8)

[P4335] 证明。非负概率加权满足三角不等式，且每项绝对值不超过最大值。证毕。

[P4336] 定理E.1。 对合法转移核和固定合法策略，

[P4338 式(E-9) M004549] $\parallel T_{P}^{\pi}V - T_{P}^{\pi}V^{(2)} \parallel_{\infty} \leq \gamma \parallel V - V^{(2)} \parallel_{\infty}.$

[P4339 式(E-9)] (E-9)

[P4340 M004550] 证明。 相减后即时奖励抵消，对任意$s$，

[P4342 式(E-10) M004551] $\begin{aligned}
 & |(T_{P}^{\pi}V)(s) - (T_{P}^{\pi}V^{(2)})(s)| \\
 & = \gamma\left| \sum_{a}^{}\pi(a \mid s)\sum_{s'}^{}P(s' \mid s,a)\lbrack V(s') - V^{(2)}(s')\rbrack \right| \\
 & \leq \gamma\sum_{a}^{}\pi(a \mid s)\sum_{s'}^{}P(s' \mid s,a)|V(s') - V^{(2)}(s')| \\
 & \leq \gamma \parallel V - V^{(2)} \parallel_{\infty}.
\end{aligned}$

[P4343 式(E-10)] (E-10)

[P4344] 取状态最大值得到结论。证毕。

[P4345 M004552,M004553,M004554] 由于E.1已经保证${\widehat{P}}_{R}$合法，将$P$替换为${\widehat{P}}_{R}$便得到残差模型的收缩性质。它不需要残差拟合误差为零，也不要求残差来自某种特定回归算法。

[P4346 M004555,M004556] 固定策略后的状态转移核仍保持非负与概率归一化，期望奖励仍满足式(E-2)。由有限维赋范空间的完备性、定理E.1及巴拿赫不动点定理，算子存在唯一价值不动点 $V_{P}^{\pi}$，精确迭代 $V_{n + 1} = T_{P}^{\pi}V_{n}$ 满足

[P4348 式(E-11) M004557] $\parallel V_{n} - V_{P}^{\pi} \parallel_{\infty} \leq \gamma^{n} \parallel V_{0} - V_{P}^{\pi} \parallel_{\infty}.$

[P4349 式(E-11)] (E-11)

[P4350] 进一步，对不动点方程取绝对值并代入奖励有界条件，可得

[P4352 式(E-12) M004558] $\parallel V_{P}^{\pi} \parallel_{\infty} \leq R_{\max} + \gamma \parallel V_{P}^{\pi} \parallel_{\infty}.$

[P4353 式(E-12)] (E-12)

[P4354 M004559] 移项并除以 $1 - \gamma > 0$，即得状态价值函数的一致有界估计：

[P4356 式(E-13) M004560] $\parallel V_{P}^{\pi} \parallel_{\infty} \leq \frac{R_{\max}}{1 - \gamma}.$

[P4357 式(E-13)] (E-13)

[P4358 M004561] E.3 误差关系。对两个有限离散概率分布 $p,q$，定义总变差距离（Total Variation Distance）为

[P4360 式(E-14) M004562] $TV(p,q) = \frac{1}{2}\sum_{j}^{}|p_{j} - q_{j}|.$

[P4361 式(E-14)] (E-14)

[P4362 M004563] 引理E.2。 对任意有界函数 $f$，有

[P4364 式(E-15) M004564] $\left| \sum_{j}^{}(p_{j} - q_{j})f_{j} \right| \leq 2 \parallel f \parallel_{\infty}\, TV(p,q).$

[P4365 式(E-15)] (E-15)

[P4366] 证明。 由三角不等式，

[P4368 式(E-16) M004565] $\left| \sum_{j}^{}(p_{j} - q_{j})f_{j} \right| \leq \sum_{j}^{}|p_{j} - q_{j}|\,|f_{j}| \leq \parallel f \parallel_{\infty}\sum_{j}^{}|p_{j} - q_{j}|.$

[P4369 式(E-16)] (E-16)

[P4370] 代入式(E-14)总变差定义即得。证毕。

[P4371 M004566,M004567,M004568,M004569] 设目标环境转移核 $P_{tgt}$ 与残差修正转移核 ${\widehat{P}}_{R}$ 共享奖励函数 $R$ 和折扣因子 $\gamma$，记一致总变差误差上界为

[P4373 式(E-17) M004570] $\varepsilon_{P} = \max_{s,a}TV\left( P_{tgt}( \cdot \mid s,a),\ {\widehat{P}}_{R}( \cdot \mid s,a) \right).$

[P4374 式(E-17)] (E-17)

[P4375 M004571] 定理E.2。 对任意固定合法策略 $\pi$，有

[P4377 式(E-18) M004572] $\parallel V_{tgt}^{\pi} - {\widehat{V}}_{R}^{\pi} \parallel_{\infty} \leq \frac{2\gamma R_{\max}}{(1 - \gamma)^{2}}\varepsilon_{P}.$

[P4378 式(E-18)] (E-18)

[P4379 M004573] 证明。 记价值不动点之差为 $\varepsilon_{V} = \parallel V_{tgt}^{\pi} - {\widehat{V}}_{R}^{\pi} \parallel_{\infty}$。由引理E.2及式(E-13)价值上界，有

[P4381 式(E-19) M004574] $\begin{array}{r}
 \parallel T_{P_{tgt}}^{\pi}V_{tgt}^{\pi} - T_{{\widehat{P}}_{R}}^{\pi}V_{tgt}^{\pi} \parallel_{\infty} \leq 2\gamma \parallel V_{tgt}^{\pi} \parallel_{\infty}\varepsilon_{P} \\
 \leq \frac{2\gamma R_{\max}}{1 - \gamma}\varepsilon_{P}.
\end{array}$

[P4382 式(E-19)] (E-19)

[P4383 M004575] 在两个不动点之间加减中间项 $T_{{\widehat{P}}_{R}}^{\pi}V_{tgt}^{\pi}$，再结合式(E-9)与式(E-19)，可得

[P4385 式(E-20) M004576] $\begin{array}{r}
\varepsilon_{V} \leq \parallel T_{P_{tgt}}^{\pi}V_{tgt}^{\pi} - T_{{\widehat{P}}_{R}}^{\pi}V_{tgt}^{\pi} \parallel_{\infty} + \parallel T_{{\widehat{P}}_{R}}^{\pi}V_{tgt}^{\pi} - T_{{\widehat{P}}_{R}}^{\pi}{\widehat{V}}_{R}^{\pi} \parallel_{\infty} \\
 \leq \frac{2\gamma R_{\max}}{1 - \gamma}\varepsilon_{P} + \gamma\varepsilon_{V}.
\end{array}$

[P4386 式(E-20)] (E-20)

[P4387 M004577,M004578] 将 $\gamma\varepsilon_{V}$ 移至左端并除以 $1 - \gamma > 0$，即得式(E-18)。证毕。

[P4388 M004579] 若两模型的奖励函数亦存在差异，设两模型奖励绝对值均不超过 $R_{\max}$，且单步期望奖励的一致误差上界为

[P4390 式(E-21) M004580] $\varepsilon_{r} = \max_{s,a}|R_{tgt}(s,a) - \widehat{R}(s,a)|,$

[P4391 式(E-21)] (E-21)

[P4392 M004581] 则一步贝尔曼更新额外引入至多 $\varepsilon_{r}$ 的即时奖励偏差：

[P4394 式(E-22) M004582] $\varepsilon_{V} \leq \varepsilon_{r} + \frac{2\gamma R_{\max}}{1 - \gamma}\varepsilon_{P} + \gamma\varepsilon_{V},$

[P4395 式(E-22)] (E-22)

[P4396] 移项化简即得正文第6.3.4节式(6-13)的完整误差上界：

[P4398 式(E-23) M004583] $\varepsilon_{V} \leq \frac{\varepsilon_{r}}{1 - \gamma} + \frac{2\gamma R_{\max}}{(1 - \gamma)^{2}}\varepsilon_{P}.$

[P4399 式(E-23)] (E-23)

[P4400 M004584,M004585,M004586,M004587,M004588,M004589,M004590] 无折扣有限事件时域下的误差界。 针对第5、6章采用有限时域无折扣（$\gamma = 1$）设置的公交事件决策任务，将运营时刻、待决策事件及终止标记纳入增广状态，并将终止后补齐为零奖励吸收态。设在对两模型均封闭的分析范围内，每条运行轨迹至多包含 $K_{\max}$ 个决策事件，单次事件奖励绝对值不超过 $R_{\max}$，期望即时奖励的一致误差不超过 $\varepsilon_{r}$，联合转移核的总变差误差不超过 $\varepsilon_{P}$。对同一固定策略 $\pi$，记剩余 $n_{rem}$ 个事件时的价值误差为

[P4402 式(E-23a) M004591] $e_{n_{rem}} = \parallel V_{n_{rem}}^{\pi} - {\widehat{V}}_{n_{rem}}^{\pi} \parallel_{\infty},\quad\quad e_{0} = 0.$

[P4403 式(E-23a)] (E-23a)

[P4404 M004592,M004593,M004594,M004595,M004596] 在两个有限时域递推式之间加减使用同一后继价值的中间项：即时奖励误差贡献至多 $\varepsilon_{r}$，后继价值误差贡献至多 $e_{n_{rem}}$；同时，剩余 $n_{rem}$ 个事件的累积价值绝对值不超过 $n_{rem}R_{\max}$，由式(E-15)可得转移核误差项至多为 $2n_{rem}R_{\max}\varepsilon_{P}$。因此递推不等式为

[P4406 式(E-23b) M004597] $e_{n_{rem} + 1} \leq \varepsilon_{r} + e_{n_{rem}} + 2n_{rem}R_{\max}\varepsilon_{P}.$

[P4407 式(E-23b)] (E-23b)

[P4408 M004598,M004599,M004600] 从 $e_{0} = 0$ 出发对 $n_{rem} = 0,\ldots,K_{\max} - 1$ 累加求和，利用 $\sum_{n_{rem} = 0}^{K_{\max} - 1}2n_{rem} = K_{\max}(K_{\max} - 1)$，即得有限事件步时域的价值评估误差上界：

[P4410 式(E-23c) M004601] $\parallel V_{K_{\max}}^{\pi} - {\widehat{V}}_{K_{\max}}^{\pi} \parallel_{\infty} \leq K_{\max}\varepsilon_{r} + K_{\max}(K_{\max} - 1)R_{\max}\varepsilon_{P}.$

[P4411 式(E-23c)] (E-23c)

[P4412 M004602,M004603,M004604,M004605,M004606,M004607] 回到有限状态折扣设定，若仅在状态子集 $\mathcal{C \subseteq S}$ 上比较两模型，当奖励与转移误差界覆盖所有 $s\mathcal{\in C}$ 及策略赋予正概率的动作、且两模型在这些状态动作下均以概率1保持在子集 $\mathcal{C}$ 内（即 $\mathcal{C}$ 对两模型后继转移封闭）时，贝尔曼算子可限制在子集价值空间 $\mathbb{R}^{\mathcal{|C|}}$ 上，上述定理结论对子集上确界范数 $\parallel \cdot \parallel_{\mathcal{\infty,C}}$ 同样成立。

[P4413 M004608,M004609,M004610,M004611,M004612,M004613,M004614,M004615,M004616,M004617] 反之，若缺乏全状态空间或封闭子集上的误差控制，仅在初始轨迹访问的局部状态上满足零误差，并不能控制其余状态上的价值偏差。例如设状态集为 $\{ 0,1\}$、单动作、初始状态为 $0$，奖励为 $R(0) = 0,R(1) = 1$：目标模型令状态 $0,1$ 均自吸收，近似模型令状态 $0$ 自吸收、状态 $1$ 一步转移至状态 $0$。从初态 $0$ 出发的轨迹仅访问状态 $0$，两模型在该轨迹上的转移误差为零；然而在状态 $1$ 上有

[P4415 式(E-24) M004618] $V_{tgt}(1) = \frac{1}{1 - \gamma},\quad\quad\widehat{V}(1) = 1,$

[P4416 式(E-24)] (E-24)

[P4417 M004619,M004620] 当 $0 < \gamma < 1$ 时，全局最大范数价值误差为 $\gamma/(1 - \gamma) > 0$。

[P4418 M004621,M004622] 若一列近似模型在同一固定策略、同一奖励上界及同一封闭状态子集上满足 $\varepsilon_{P,n} \rightarrow 0$ 和 $\varepsilon_{r,n} \rightarrow 0$，则由式(E-23)及夹逼定理立得一致收敛性：

[P4420 式(E-25) M004623] $\parallel V_{tgt}^{\pi} - {\widehat{V}}_{n}^{\pi} \parallel_{\infty} \rightarrow 0.$

[P4421 式(E-25)] (E-25)

[P4422 M004624,M004625,M004626,M004627,M004628,M004629,M004630,M004631,M004632,M004633,M004634,M004635,M004636] E.4 数据覆盖。模型在历史数据采集策略下预测准确，并不自动保证对任意新策略同样准确。考察如下双状态双动作反例：设状态集为 $\{ 0,1\}$，动作集为 $\{ a_{0},a_{1}\}$，初始状态为 $0$，奖励为 $R(0,a) = 0$、$R(1,a) = 1$，$0 < \gamma < 1$。目标模型在状态 $0$ 执行 $a_{0}$ 时留在 $0$、执行 $a_{1}$ 时转移至吸收态 $1$；近似模型则在状态 $0$ 执行任一动作均留在 $0$。

[P4423 M004637,M004638,M004639,M004640,M004641] 若历史数据采集策略始终选择动作 $a_{0}$，则数据集仅包含转移 $(0,a_{0},0)$，近似模型在历史数据支撑集上的预测误差恒为零；然而当新策略 $\pi'$ 在状态 $0$ 选择动作 $a_{1}$ 时，两模型下的状态价值分别为

[P4425 式(E-26) M004642] $V_{tgt}^{\pi'}(0) = \sum_{k = 1}^{\infty}\gamma^{k} = \frac{\gamma}{1 - \gamma},\quad\quad{\widehat{V}}^{\pi'}(0) = 0.$

[P4426 式(E-26)] (E-26)

[P4427] 该反例直观印证了第5章与第6章在固定离线数据之外引入源仿真在线探索以拓展动作覆盖范围的理论必要性。

[P4428 M004643] E.5 更新条件。在式(E-6)的概率约束下，合法转移残差集合构成有限维空间 $X_{\delta} = \mathbb{R}^{\mathcal{|S|}\,\mathcal{|A|}\,\mathcal{|S|}}$ 中的非空紧致凸闭集：

[P4430 式(E-27) M004644] $\begin{array}{r}
X_{\delta} = \mathbb{R}^{\mathcal{|S|}\,\mathcal{|A|}\,\mathcal{|S|}}, \\
\mathcal{C}_{\delta} = \left\{ \delta \in X_{\delta}:\ P_{sim} + \delta \geq 0,\ \sum_{s'}^{}\delta(s,a,s') = 0 \right\}, \\
F:\mathcal{C}_{\delta} \rightarrow \mathcal{C}_{\delta},
\end{array}$

[P4431 式(E-27)] (E-27)

[P4432] 赋予无穷范数

[P4434 式(E-28) M004645] $\parallel \delta \parallel_{\infty} = \max_{s,a,s'}|\delta(s,a,s')|.$

[P4435 式(E-28)] (E-28)

[P4436 M004646] 若由策略更新、访问分布变化与模型重拟合复合而成的完整残差更新映射 $F:\mathcal{C}_{\delta} \rightarrow \mathcal{C}_{\delta}$ 满足压缩条件

[P4438 式(E-29) M004647] $\parallel F(\delta_{1}) - F(\delta_{2}) \parallel_{\infty} \leq L \parallel \delta_{1} - \delta_{2} \parallel_{\infty},\quad\quad 0 \leq L < 1,$

[P4439 式(E-29)] (E-29)

[P4440] 则可得如下不动点存在唯一性定理：

[P4441 M004648,M004649,M004650,M004651,M004652] 定理E.3。 若映射 $F$ 在非空完备闭集 $\mathcal{C}_{\delta}$ 上满足式(E-29)，则存在唯一残差不动点 $\delta^{*} \in \mathcal{C}_{\delta}$ 满足 $F(\delta^{*}) = \delta^{*}$，且迭代序列 $\delta_{n + 1} = F(\delta_{n})$ 满足

[P4443 式(E-30) M004653] $\parallel \delta_{n} - \delta^{*} \parallel_{\infty} \leq L^{n} \parallel \delta_{0} - \delta^{*} \parallel_{\infty}.$

[P4444 式(E-30)] (E-30)

[P4445 M004654,M004655,M004656,M004657] 证明。 非空闭集 $\mathcal{C}_{\delta}$ 在无穷范数下完备，$F$ 映射 $\mathcal{C}_{\delta}$ 至自身且满足式(E-29)，由巴拿赫不动点定理立得唯一不动点 $\delta^{*}$ 及几何收敛界式(E-30)。证毕。

[P4446 M004658,M004659,M004660,M004661,M004662] 若复合更新映射 $F$ 由策略调整、访问分布映射与回归拟合三步串联而成，各自的Lipschitz常数分别为 $L_{adj}$、$L_{occ}$ 和 $B$，则当乘积满足 $L = L_{adj}L_{occ}B < 1$ 时：

[P4448 式(E-31) M004663] $\parallel F(\delta_{1}) - F(\delta_{2}) \parallel_{\infty} \leq L_{adj}L_{occ}B \parallel \delta_{1} - \delta_{2} \parallel_{\infty},$

[P4449 式(E-31)] (E-31)

[P4450] 即可满足定理E.3的压缩前提条件。

[P4451] E.6 置信排序。本节分析一个静态排序问题：按预测误差构造的正值置信评分与二元判别后验评分在何种条件下对同一批输入给出相同的严格排序。

[P4452 M004664,M004665,M004666,M004667,M004668,M004669,M004670,M004671,M004672,M004673] 在比较域 $\mathcal{X}$ 上，给定两个候选模型的平方误差函数 $e_{real}(x)$、$e_{resid}(x)$，正值评分函数 $w_{rank}(x) > 0$ 及判别概率评分 $0 < \varphi_{cls}(x) < 1$。记 $\tau_{rank} > 0$ 为温度尺度参数，$c_{rank}$ 为加性常数，$\varepsilon_{rank} \geq 0$ 为误差波动上界。假设存在 $\tau_{rank} > 0$ 和常数 $c_{rank}$，使得对数评分满足：

[P4454 式(E-32) M004674] $\mathcal{l}_{w_{rank}}(x): = \log w_{rank}(x) = - \frac{e_{real}(x)}{\tau_{rank}} + c_{rank},$

[P4455 式(E-32)] (E-32)

[P4457 式(E-33) M004675] $\mathcal{l}_{\varphi_{cls}}(x): = \log\frac{\varphi_{cls}(x)}{1 - \varphi_{cls}(x)} = \frac{e_{resid}(x) - e_{real}(x)}{\tau_{rank}}.$

[P4458 式(E-33)] (E-33)

[P4459] 由此可得两对数评分之差为：

[P4461 式(E-34) M004676] $\mathcal{l}_{\varphi_{cls}}(x) - \mathcal{l}_{w_{rank}}(x) = \frac{e_{resid}(x)}{\tau_{rank}} - c_{rank}.$

[P4462 式(E-34)] (E-34)

[P4463 M004677,M004678] 式(E-33)对应于等先验、同方差高斯条件密度假设（协方差为 $\sigma^{2}I_{d}$，尺度取 $\tau_{rank} = 2\sigma^{2}$）下的对数后验几率比。

[P4464 M004679] 引理E.3。 对任意 $x_{1},x_{2}\mathcal{\in X}$，

[P4466 式(E-35) M004680] $w_{rank}(x_{1}) < w_{rank}(x_{2})\quad \Leftrightarrow \quad e_{real}(x_{2}) < e_{real}(x_{1}).$

[P4467 式(E-35)] (E-35)

[P4468 M004681,M004682,M004683,M004684,M004685] 证明。 正实数上的对数函数严格递增，故 $w_{rank}(x_{1}) < w_{rank}(x_{2})$ 等价于 $\log w_{rank}(x_{1}) < logw_{rank}(x_{2})$。代入式(E-32)，消去常数 $c_{rank}$ 并同乘 $\tau_{rank} > 0$，即得 $- e_{real}(x_{1}) < - e_{real}(x_{2})$。各步均可逆。证毕。

[P4469 M004686,M004687,M004688] 定理E.4。 若存在常数 $e_{const}$，使 $e_{resid}(x) = e_{const}$ 对所有 $x\mathcal{\in X}$ 成立，则

[P4471 式(E-36) M004689] $w_{rank}(x_{1}) < w_{rank}(x_{2})\quad \Leftrightarrow \quad\varphi_{cls}(x_{1}) < \varphi_{cls}(x_{2}).$

[P4472 式(E-36)] (E-36)

[P4473 M004690,M004691,M004692,M004693] 证明。 由前提条件及式(E-34)可知，$\mathcal{l}_{\varphi_{cls}}(x) = \mathcal{l}_{w_{rank}}(x) + e_{const}/\tau_{rank} - c_{rank}$，两种对数评分仅相差一个常数平移项，故对任意点对的严格大小次序完全一致。同时，对 $0 < u,v < 1$，由于对数几率变换 $u \mapsto log\lbrack u/(1 - u)\rbrack$ 在 $(0,1)$ 上严格递增：

[P4475 式(E-37) M004694] $\frac{u}{1 - u} < \frac{v}{1 - v}\quad \Leftrightarrow \quad u(1 - v) < v(1 - u)\quad \Leftrightarrow \quad u < v,$

[P4476 式(E-37)] (E-37)

[P4477 M004695,M004696] 故 $w_{rank}$ 与 $\varphi_{cls}$ 的严格排序完全一致。证毕。

[P4478 M004697,M004698] 定理E.5。 若存在常数 $\overline{e}$ 和 $\varepsilon_{rank} \geq 0$，使得

[P4480 式(E-38) M004699] $|e_{resid}(x) - \overline{e}| \leq \varepsilon_{rank}\quad(\forall x\mathcal{\in X),}$

[P4481 式(E-38)] (E-38)

[P4482] 则对任意满足

[P4484 式(E-39) M004700] $|logw_{rank}(x_{1}) - logw_{rank}(x_{2})| > \frac{2\varepsilon_{rank}}{\tau_{rank}}$

[P4485 式(E-39)] (E-39)

[P4486 M004701,M004702,M004703] 的输入点对 $(x_{1},x_{2})$，$w_{rank}$ 与 $\varphi_{cls}$ 的严格排序保持一致。

[P4487] 证明。 定义误差差分为

[P4489 式(E-40) M004704] $\Delta e = e_{real}(x_{1}) - e_{real}(x_{2}),\quad\quad\Delta e_{resid} = e_{resid}(x_{1}) - e_{resid}(x_{2}).$

[P4490 式(E-40)] (E-40)

[P4491 M004705] 由式(E-38)及三角不等式有 $|\Delta e_{resid}| \leq 2\varepsilon_{rank}$。由式(E-32)与式(E-33)可得：

[P4493 式(E-41) M004706] $\mathcal{l}_{w_{rank}}(x_{1}) - \mathcal{l}_{w_{rank}}(x_{2}) = - \Delta e/\tau_{rank},$

[P4494 式(E-41)] (E-41)

[P4496 式(E-42) M004707] $\mathcal{l}_{\varphi_{cls}}(x_{1}) - \mathcal{l}_{\varphi_{cls}}(x_{2}) = (\Delta e_{resid} - \Delta e)/\tau_{rank}.$

[P4497 式(E-42)] (E-42)

[P4498 M004708,M004709,M004710,M004711,M004712,M004713,M004714,M004715,M004716] 式(E-39)的间隔条件等价于 $|\Delta e| > 2\varepsilon_{rank}$。若 $\Delta e > 2\varepsilon_{rank}$，则 $- \Delta e < 0$，且 $\Delta e_{resid} - \Delta e \leq 2\varepsilon_{rank} - \Delta e < 0$，两对数评分之差同为负；若 $\Delta e < - 2\varepsilon_{rank}$，则 $- \Delta e > 0$，且 $\Delta e_{resid} - \Delta e \geq - 2\varepsilon_{rank} - \Delta e > 0$，两对数评分之差同为正。结合对数与对数几率的严格单调性，即知 $w_{rank}$ 与 $\varphi_{cls}$ 严格排序一致。证毕。

[P4499 M004717,M004718] 对于差值不超过阈值 $2\varepsilon_{rank}/\tau_{rank}$ 的点对，当 $e_{resid}(x)$ 随状态显著波动时，两类评分可能给出不同的局部排序。

[P4500 M004719,M004720] E.7 理论解释。合法残差转移核保持了固定策略评价算子的收缩结构，而所得价值不动点与真实目标价值的接近程度则由两者的单步奖励误差 $\varepsilon_{r}$ 与转移总变差误差 $\varepsilon_{P}$ 共同决定（定理E.2）：源模型结构提供基础动力学先验，目标域离线数据修正跨域分布偏移，验证集误差准则则在残差修正与直接建模之间客观择优。

[P4501] E.8 适用范围。定理E.2的价值误差上界要求误差控制覆盖整个状态空间或对两模型后继转移均封闭的状态子集；E.4节的反例表明，仅在历史采集策略支撑集上降低预测误差不足以保证新策略下的全局价值精度，需结合源仿真在线探索扩展动作覆盖。

[P4502 M004721,M004722,M004723,M004724,M004725,M004726,M004727,M004728,M004729,M004730] E.9 对数置信度递推的尺度归一化分析。本节在固定策略、固定局部可靠性评分函数及状态转移核的条件下，严格证明正文第6.2.3节原始对数置信度递推（式(6-5)）与归一化对数置信度递推（式(6-7)）的不动点幂次尺度关系。记状态—动作对为 $x = (s,a)$，局部预测可靠性评分为 $c(x) \in \lbrack c_{\min}^{\Delta},1\rbrack$（其中下界 $c_{\min}^{\Delta} > 0$）；记 $K_{\Delta}^{\pi}(x' \mid x)$ 为策略 $\pi$ 下包含非终止条件 $(1 - d)$ 的一步转移核，满足非负性及次概率行和条件 $\sum_{x'}^{}K_{\Delta}^{\pi}(x' \mid x) \leq 1$（严格小于1的部分对应终止事件）。在有限状态动作集合 $\mathcal{S \times A}$ 上，定义原始对数置信度算子 $T_{\Delta}^{\pi}$ 与归一化对数置信度算子 ${\widetilde{T}}_{\Delta}^{\pi}$ 分别为：

[P4504 式(E-43) M004731] $\begin{array}{r}
(T_{\Delta}^{\pi}Q^{\Delta})(x) = logc(x) + \gamma_{\Delta}\sum_{x'}^{}K_{\Delta}^{\pi}(x' \mid x)Q^{\Delta}(x'), \\
({\widetilde{T}}_{\Delta}^{\pi}Q^{\Delta})(x) = (1 - \gamma_{\Delta})logc(x) + \gamma_{\Delta}\sum_{x'}^{}K_{\Delta}^{\pi}(x' \mid x)Q^{\Delta}(x'),
\end{array}$

[P4505 式(E-43)] (E-43)

[P4506 M004732] 式中，递推折扣系数满足 $0 \leq \gamma_{\Delta} < 1$。

[P4507 M004733,M004734,M004735,M004736,M004737] 命题E.1（对数置信度不动点尺度归一化定理）。 算子 $T_{\Delta}^{\pi}$ 与 ${\widetilde{T}}_{\Delta}^{\pi}$ 均为无穷范数下的 $\gamma_{\Delta}$-压缩映射，各自存在唯一不动点 $Q^{\Delta,*}$ 与 ${\widetilde{Q}}^{\Delta,*}$，且两不动点及其对应的指数样本权重满足如下精确恒等关系：

[P4509 式(E-44) M004738] $\begin{array}{r}
{\widetilde{Q}}^{\Delta,*}(x) = (1 - \gamma_{\Delta})\, Q^{\Delta,*}(x), \\
\exp\left( {\widetilde{Q}}^{\Delta,*}(x) \right) = \lbrack exp\left( Q^{\Delta,*}(x) \right)\rbrack^{1 - \gamma_{\Delta}}.
\end{array}$

[P4510 式(E-44)] (E-44)

[P4511 M004739,M004740] 证明。 对任意两个有界置信评价函数 $Q_{1}^{\Delta},Q_{2}^{\Delta}$，即时项在算子作差时精确抵消；利用传播核 $K_{\Delta}^{\pi}$ 非负且行和不超过1，有

[P4513 式(E-45) M004741] $\parallel {\widetilde{T}}_{\Delta}^{\pi}Q_{1}^{\Delta} - {\widetilde{T}}_{\Delta}^{\pi}Q_{2}^{\Delta} \parallel_{\infty} \leq \gamma_{\Delta} \parallel Q_{1}^{\Delta} - Q_{2}^{\Delta} \parallel_{\infty}.$

[P4514 式(E-45)] (E-45)

[P4515 M004742,M004743,M004744,M004745,M004746,M004747] 原始算子 $T_{\Delta}^{\pi}$ 同理。由于两算子均作用于有限维完备赋范空间，故各自存在唯一不动点 $Q^{\Delta,*}$ 与 ${\widetilde{Q}}^{\Delta,*}$。将原始算子的不动点方程 $Q^{\Delta,*}(x) = (T_{\Delta}^{\pi}Q^{\Delta,*})(x)$ 两端同乘常数 $(1 - \gamma_{\Delta})$，并利用线性性将 $(1 - \gamma_{\Delta})$ 移入求和号内，可得

[P4517 式(E-46) M004748] $\begin{array}{r}
(1 - \gamma_{\Delta})Q^{\Delta,*}(x) = (1 - \gamma_{\Delta})logc(x) \\
\quad + \gamma_{\Delta}\sum_{x'}^{}K_{\Delta}^{\pi}(x' \mid x)\,\left\lbrack (1 - \gamma_{\Delta})Q^{\Delta,*}(x') \right\rbrack \\
 = \left( {\widetilde{T}}_{\Delta}^{\pi}\left\lbrack (1 - \gamma_{\Delta})Q^{\Delta,*} \right\rbrack \right)(x).
\end{array}$

[P4518 式(E-46)] (E-46)

[P4519 M004749,M004750,M004751] 式(E-46)表明 $(1 - \gamma_{\Delta})Q^{\Delta,*}$ 恰好满足归一化算子 ${\widetilde{T}}_{\Delta}^{\pi}$ 的不动点方程；由不动点的唯一性立得 ${\widetilde{Q}}^{\Delta,*} = (1 - \gamma_{\Delta})Q^{\Delta,*}$，两端取指数即得式(E-44)第二式。证毕。

[P4520 M004752,M004753,M004754,M004755,M004756,M004757,M004758,M004759,M004760,M004761,M004762,M004763] 特别地，当过程非终止（$\sum_{x'}^{}K_{\Delta}^{\pi}(x' \mid x) = 1$）且局部可靠性评分恒为常数 $c_{const} \in (0,1)$ 时，常值函数 $Q^{\Delta,*} = \frac{\log c_{const}}{1 - \gamma_{\Delta}}$ 与 ${\widetilde{Q}}^{\Delta,*} = logc_{const}$ 分别为两算子的唯一不动点，取指数后原始不动点权重为 $c_{const}^{1/(1 - \gamma_{\Delta})}$，而归一化不动点权重恰为 $c_{const}$（即正文第6.3.4节式(6-15)）。例如当 $c_{const} = 0.5$ 且 $\gamma_{\Delta} = 0.9$ 时，原始递推不动点权重坍缩为 ${0.5}^{10} \approx 0.000977$，而归一化递推不动点权重保持为 $0.5$；同时，由于幂变换 $z \mapsto z^{1 - \gamma_{\Delta}}$ 在正实数域 $(0,\infty)$ 上严格单调递增，归一化递推在不改变各状态动作对之间相对可靠性排序的前提下，消除了折现深度引起的幂次衰减，恢复了模型辅助分支在动作价值更新中的有效梯度尺度。