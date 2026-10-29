# 附录D CS-BAPR策略响应与贝尔曼算子的数学证明

[P3964] 附录D CS-BAPR策略响应与贝尔曼算子的数学证明

[P3965] 本附录分别分析完整策略的动作误差、结构化函数及其复合的正则性，以及固定惩罚价值算子的性质。第D.1节先给出最终公交软符号配置的实现对应；第D.2至D.5节及第D.8至D.10节给出条件误差分析；第D.6至D.7节单独讨论价值目标与固定算子的关系。NAU输出结构的性质不能直接用于不含NAU的软符号策略。

[P3966] 首先讨论标量函数和赋范空间中的可微映射，再将分析扩展到有限状态动作空间中的价值算子。

[P3967 M004161,M004162,M004163] D.1 基本条件。设确定性策略映射与参考动作映射为 $\mu_{\det},f_{ref}\mathbb{:R \rightarrow R}$，考察输入集合为 $X_{eval}$。给定非负常数 $\varepsilon_{der}$，一阶导数一致性条件为

[P3969 式(D-1) M004164] $|{\mu_{\det}}'(x) - {f_{ref}}'(x)| \leq \varepsilon_{der},\quad\quad x \in X_{eval}.$

[P3970 式(D-1)] (D-1)

[P3971 M004165] 给定非负常数 $L_{D\mu}$，策略导数的Lipschitz连续条件为

[P3973 式(D-2) M004166] $|{\mu_{\det}}'(x) - {\mu_{\det}}'(y)| \leq L_{D\mu}|x - y|.$

[P3974 式(D-2)] (D-2)

[P3975] 上述两个条件分别刻画同一点处的雅可比导数一致性与不同点间的导数变化平滑性。

[P3976 M004167,M004168] 一般地，设 $X_{in},X_{out}$ 为实赋范空间，$D\mu_{\det}(x):X_{in} \rightarrow X_{out}$ 表示Fréchet导数，导数范数均取算子范数，则相应条件写为

[P3978 式(D-3) M004169] $\parallel D\mu_{\det}(x) - Df_{ref}(x) \parallel \leq \varepsilon_{der}$

[P3979 式(D-3)] (D-3)

[P3980] 以及

[P3982 式(D-4) M004170] $\parallel D\mu_{\det}(x) - D\mu_{\det}(y) \parallel \leq L_{D\mu} \parallel x - y \parallel .$

[P3983 式(D-4)] (D-4)

[P3984 M004171,M004172] 当 $X_{in} = \mathbb{R}^{n}$、$X_{out} = \mathbb{R}^{m}$ 时，Fréchet导数即为雅可比矩阵，所有距离与导数上界均相对于同一无量纲归一化坐标系定义。

[P3985 M004173,M004174,M004175,M004176,M004177,M004178] 公交软符号配置与正文第4.2.3节的符号对应关系。 正文第4.2.3节采用15维归一化状态 $x$（或记为 $s$）与归一化驻站动作 $\widetilde{a} \in \lbrack - 1,1\rbrack$。状态坐标从1编号，其中第6、7、9维分别表示采用同一尺度归一化的前向车头时距、后向车头时距和线路计划间隔。将当前状态 $x$ 的第6、7维同时替换为第9维计划间隔、其余13维保持不变，即得到正文式(4-13)中的计划均衡参考输入 $x_{ref}(x)$（本附录简记为映射 $\Pi_{ref}(x) = x_{ref}(x)$）。

[P3986 M004179,M004180,M004181,M004182,M004183,M004184,M004185] 设 $f_{\theta}$ 为正文第4.2.3节式(4-13)定义的策略神经网络分支，$\beta_{0}$ 为初值为0的可学习标量参数（在本附录中分别对应记为 $m_{\theta} \equiv f_{\theta}$ 与 $b_{\theta} \equiv \beta_{0}$）。完整确定性归一化动作 $\mu_{det,\theta}(x)$（即正文中的 $\widetilde{a}(x)$）及物理驻站秒数 $a_{hold}(x)$ 为：

[P3988 式(D-4a) M004186] $\begin{array}{r}
q_{\theta}(x) = \beta_{0} + f_{\theta}(x) - f_{\theta}\left( x_{ref}(x) \right), \\
\mu_{det,\theta}(x) = tanh\left( q_{\theta}(x) \right), \\
a_{hold}(x) = 30\,\left\lbrack 1 + \mu_{det,\theta}(x) \right\rbrack.
\end{array}$

[P3989 式(D-4a)] (D-4a)

[P3990 M004187,M004188] 由于 $x_{ref}(x_{ref}(x)) = x_{ref}(x)$，在计划均衡参考点处神经差分修正项精确为0，归一化基准输出为 $tanh(\beta_{0})$。

[P3991 M004189,M004190,M004191,M004192,M004193] 对于线路—方向组合 $l$ 经SINDy辨识得到的一步短期代价多项式 $\widehat{c}(x,\widetilde{a})$，在候选网格 $\mathcal{A}_{g} = \{ - 1, - 5/6,\ldots,1\}$ 上取玻尔兹曼温度 $\tau_{boltz} = 0.25$，构造正文式(4-14)的平滑符号参考动作映射 $f_{ref}(x) \equiv {\widetilde{a}}_{sym}(x)$：

[P3993 式(D-4b) M004194] $f_{ref}(x) = {clip}_{\lbrack - 0.999,0.999\rbrack}\left\lbrack \frac{\sum_{{\widetilde{a}}_{k} \in \mathcal{A}_{g}}^{}{\widetilde{a}}_{k}\exp\left( - \widehat{c}(x,{\widetilde{a}}_{k})/\tau_{boltz} \right)}{\sum_{{\widetilde{a}}_{k'} \in \mathcal{A}_{g}}^{}\exp\left( - \widehat{c}(x,{\widetilde{a}}_{k'})/\tau_{boltz} \right)} \right\rbrack.$

[P3994 式(D-4b)] (D-4b)

[P3995 M004195,M004196,M004197,M004198] 记 $J_{\mu}(x) = D\mu_{det,\theta}(x) \equiv D\pi(x)$、$J_{f}(x) = Df_{ref}(x) \equiv D\pi_{sym}(x)$，正文式(4-15)的逐状态归一化雅可比导数误差 $\mathcal{l}_{sym}(x)$ 及策略损失函数 $\mathcal{L}_{\pi}$ 为：

[P3997 式(D-4c) M004199] $\begin{array}{r}
\mathcal{l}_{sym}(x) = \frac{\parallel J_{\mu}(x) - J_{f}(x) \parallel_{F}^{2}}{1 + \parallel J_{f}(x) \parallel_{F}^{2}}, \\
\mathcal{L}_{\pi} = \mathcal{L}_{\pi,base} + \frac{0.01}{B}\sum_{i = 1}^{B}\mathcal{l}_{sym}(x_{i}),
\end{array}$

[P3998 式(D-4c)] (D-4c)

[P3999 M004200,M004201,M004202,M004203] 式中，$\parallel \cdot \parallel_{F}$ 为Frobenius范数，小批量大小 $B = 256$。相应地，正文式(4-16)在构造动作价值自举回归目标 $y_{i}$ 时，计算小批量后继状态上的批平均导数误差（取 $\lambda_{sym} = 0.01$）：

[P4001 式(D-4d) M004204] ${\bar{\mathcal{L}}}_{JC}' = \frac{1}{B}\sum_{i = 1}^{B}\mathcal{l}_{sym}(x_{i}')$

[P4002 式(D-4d)] (D-4d)

[P4004 式(D-4e) M004205] $y_{i} = \frac{r_{i}}{\sigma_{r}} + (1 - d_{i})\gamma\left\lbrack \min_{1 \leq k \leq 3}{\widehat{Q}}_{\phi_{k}'}(s_{i}',a_{i}') - \alpha\log\pi_{\theta}(a_{i}' \mid s_{i}') - \lambda_{sym}\, sg\left( {\bar{\mathcal{L}}}_{JC}' \right) \right\rbrack.$

[P4005 式(D-4e)] (D-4e)

[P4006] 将下文导数界应用于公交连续运行状态时，系在给定离散类别编码的条件下考察连续运行状态截面（且位于参考映射非饱和可微开集内）。

[P4007] D.2 标量误差。

[P4008 M004206,M004207,M004208,M004209,M004210] 命题D.1。 设区间端点满足 $a \leq b$，标量函数 $\mu_{\det},f_{ref}$ 在 $\lbrack a,b\rbrack$ 上连续、在 $(a,b)$ 上可微，且对任意 $x \in \lbrack a,b)$ 满足

[P4010 式(D-5) M004211] $|{\mu_{\det}}'(x) - {f_{ref}}'(x)| \leq \varepsilon_{der},\quad\quad\varepsilon_{der} \geq 0,$

[P4011 式(D-5)] (D-5)

[P4012 M004212] 则误差函数 $e_{map}(x) = \mu_{\det}(x) - f_{ref}(x)$ 满足

[P4014 式(D-6) M004213] $|\lbrack\mu_{\det}(b) - f_{ref}(b)\rbrack - \lbrack\mu_{\det}(a) - f_{ref}(a)\rbrack| \leq \varepsilon_{der}(b - a).$

[P4015 式(D-6)] (D-6)

[P4016 M004214,M004215,M004216,M004217,M004218] 证明。 令 $e_{map}(x) = \mu_{\det}(x) - f_{ref}(x)$。当 $a = b$ 时结论直接成立。当 $a < b$ 时，$e_{map}$ 在闭区间上连续、开区间上可微，由中值定理存在 $\xi \in (a,b)$，使

[P4018 式(D-7) M004219] $e_{map}(b) - e_{map}(a) = {e_{map}}'(\xi)(b - a).$

[P4019 式(D-7)] (D-7)

[P4020] 取绝对值并代入导数界即可。证毕。

[P4021 M004220] 若基准误差满足 $|e_{map}(a)| \leq \varepsilon_{val}$，则由三角不等式有

[P4023 式(D-8) M004221] $|e_{map}(b)| \leq \varepsilon_{val} + \varepsilon_{der}(b - a).$

[P4024 式(D-8)] (D-8)

[P4025] 该界表明，导数差异沿外推路径累积；基准点的函数误差与路径上的累积误差共同决定最终误差。

[P4026 M004222,M004223,M004224] 命题D.2。 设 $\varepsilon_{val},\varepsilon_{der},L \geq 0$，$\mu_{\det},f_{ref}$ 可微，$a \leq b$，并满足

[P4028 式(D-9) M004225] $|e_{map}(a)| \leq \varepsilon_{val},\quad\quad|{e_{map}}'(x)| \leq \varepsilon_{der} + L(x - a),\quad x \in \lbrack a,b).$

[P4029 式(D-9)] (D-9)

[P4030] 则

[P4032 式(D-10) M004226] $|e_{map}(b)| \leq \varepsilon_{val} + \varepsilon_{der}(b - a) + \frac{L}{2}(b - a)^{2}.$

[P4033 式(D-10)] (D-10)

[P4034] 证明。 定义边界函数

[P4036 式(D-11) M004227] $B_{err}(x) = \varepsilon_{val} + \varepsilon_{der}(x - a) + \frac{L}{2}(x - a)^{2}.$

[P4037 式(D-11)] (D-11)

[P4038 M004228,M004229,M004230,M004231,M004232,M004233,M004234,M004235,M004236] 其导数为 ${B_{err}}'(x) = \varepsilon_{der} + L(x - a)$。由 ${e_{map}}'(x) \leq {B_{err}}'(x)$ 可知 $e_{map} - B_{err}$ 在区间上非增，且 $e_{map}(a) - B_{err}(a) \leq 0$，所以 $e_{map}(b) \leq B_{err}(b)$。同理，由 $- {e_{map}}'(x) \leq {B_{err}}'(x)$ 可知 $- e_{map} - B_{err}$ 非增，得到 $- e_{map}(b) \leq B_{err}(b)$。合并两式即有 $|e_{map}(b)| \leq B_{err}(b)$。证毕。

[P4039 M004237] 该证明只需要导数比较，不需要额外假设 ${e_{map}}'$ 连续。

[P4040] 命题D.3。 若

[P4042 式(D-12) M004238] $\begin{aligned}
|{\mu_{\det}}'(a) - {f_{ref}}'(a)| & \leq \varepsilon_{der}, \\
|{\mu_{\det}}'(x) - {\mu_{\det}}'(a)| & \leq L_{D\mu}|x - a|, \\
|{f_{ref}}'(x) - {f_{ref}}'(a)| & \leq L_{Df}|x - a|,
\end{aligned}$

[P4043 式(D-12)] (D-12)

[P4044] 则

[P4046 式(D-13) M004239] $|{\mu_{\det}}'(x) - {f_{ref}}'(x)| \leq \varepsilon_{der} + (L_{D\mu} + L_{Df})|x - a|.$

[P4047 式(D-13)] (D-13)

[P4048] 证明。 在导数差中加减基准点的两个导数：

[P4050 式(D-14) M004240] $\begin{aligned}
{\mu_{\det}}'(x) - {f_{ref}}'(x) = & \lbrack{\mu_{\det}}'(x) - {\mu_{\det}}'(a)\rbrack + \lbrack{\mu_{\det}}'(a) - {f_{ref}}'(a)\rbrack \\
 & + \lbrack{f_{ref}}'(a) - {f_{ref}}'(x)\rbrack.
\end{aligned}$

[P4051 式(D-14)] (D-14)

[P4052] 取绝对值，逐项应用三角不等式和已知上界即可。证毕。

[P4053] 将命题D.3代入命题D.2，得到

[P4055 式(D-15) M004241] $|\mu_{\det}(b) - f_{ref}(b)| \leq \varepsilon_{val} + \varepsilon_{der}(b - a) + \frac{L_{D\mu} + L_{Df}}{2}(b - a)^{2}.$

[P4056 式(D-15)] (D-15)

[P4057 M004242] 当参考函数的导数为常数时，取 $L_{Df} = 0$ 即得到相应特殊情形。

[P4058] D.3 向量误差。

[P4059 M004243,M004244,M004245,M004246] 命题D.4。 设 $\mu_{\det},f_{ref}:X_{in} \rightarrow X_{out}$ 可微，$\varepsilon_{der} \geq 0$，并在连接 $x_{0}$ 与 $x$ 的线段上满足

[P4061 式(D-16) M004247] $\parallel D\mu_{\det}(z) - Df_{ref}(z) \parallel \leq \varepsilon_{der}.$

[P4062 式(D-16)] (D-16)

[P4063 M004248] 若 $\parallel \mu_{\det}(x_{0}) - f_{ref}(x_{0}) \parallel \leq \varepsilon_{val}$，则

[P4065 式(D-17) M004249] $\parallel \mu_{\det}(x) - f_{ref}(x) \parallel \leq \varepsilon_{val} + \varepsilon_{der} \parallel x - x_{0} \parallel .$

[P4066 式(D-17)] (D-17)

[P4067 M004250,M004251,M004252,M004253] 证明。 记 $e_{map} = \mu_{\det} - f_{ref}$，$\Delta x = x - x_{0}$，线段路径 $z(t) = x_{0} + t\Delta x$（$t \in \lbrack 0,1\rbrack$）。由赋范空间上的微分中值不等式，

[P4069 式(D-18) M004254] $\parallel e_{map}(x) - e_{map}(x_{0}) \parallel \leq \sup_{0 \leq t \leq 1} \parallel De_{map}(z(t)) \parallel \, \parallel \Delta x \parallel \leq \varepsilon_{der} \parallel \Delta x \parallel .$

[P4070 式(D-18)] (D-18)

[P4071 M004255,M004256] 再应用三角不等式即得结论（亦可对任意满足 $\parallel \mathcal{l}_{fun} \parallel \leq 1$ 的连续线性泛函 $\mathcal{l}_{fun}:X_{out}\mathbb{\rightarrow R}$ 应用标量中值定理并取对偶上确界证得）。证毕。

[P4072 M004257,M004258,M004259,M004260,M004261] 命题D.5。 设 $X_{out}$ 为实Banach空间，$e_{path}:\lbrack 0,1\rbrack \rightarrow X_{out}$ 连续，且在 $\lbrack 0,1)$ 的每一点存在右导数 $e_{path, +}'(t)$。给定非负常数 $\varepsilon_{val},\varepsilon_{path},L_{path} \geq 0$，若

[P4074 式(D-19) M004262] $\parallel e_{path}(0) \parallel \leq \varepsilon_{val},\quad\quad \parallel e_{path, +}'(t) \parallel \leq \varepsilon_{path} + L_{path}t,$

[P4075 式(D-19)] (D-19)

[P4076] 则

[P4078 式(D-20) M004263] $\parallel e_{path}(1) \parallel \leq \varepsilon_{val} + \varepsilon_{path} + \frac{L_{path}}{2}.$

[P4079 式(D-20)] (D-20)

[P4080 M004264,M004265,M004266] 证明。 令 $B_{err}(t) = \varepsilon_{val} + \varepsilon_{path}t + L_{path}t^{2}/2$。对任意满足 $\parallel \mathcal{l}_{fun} \parallel \leq 1$ 的连续线性泛函 $\mathcal{l}_{fun}$，有

[P4082 式(D-21) M004267] $(\mathcal{l}_{fun} \circ e_{path})_{+}'(t) = \mathcal{l}_{fun}\left( e_{path, +}'(t) \right) \leq {B_{err}}'(t).$

[P4083 式(D-21)] (D-21)

[P4084 M004268,M004269,M004270,M004271,M004272,M004273,M004274,M004275,M004276,M004277,M004278] 给定 $\eta > 0$，令 $v_{cmp}(t) = \mathcal{l}_{fun}(e_{path}(t)) - B_{err}(t) - \eta t$，则 $v_{cmp, +}'(t) \leq - \eta$，且 $v_{cmp}(0) \leq 0$。假设 $v_{cmp}(1) > 0$，连续性保证 $v_{cmp}$ 在紧区间 $\lbrack 0,1\rbrack$ 上取得最小值；由于 $v_{cmp}(1) > v_{cmp}(0)$，极小值点 $t_{*}$ 必严格小于1。在该点处右侧差商非负，故右导数非负，这与 $v_{cmp, +}'(t_{*}) \leq - \eta$ 矛盾。因此 $v_{cmp}(1) \leq 0$。

[P4085 M004279,M004280,M004281] 令 $\eta \downarrow 0$，得 $\mathcal{l}_{fun}(e_{path}(1)) \leq B_{err}(1)$。利用Hahn-Banach范数对偶表示，对全部 $\parallel \mathcal{l}_{fun} \parallel \leq 1$ 取上确界，即得式(D-20)。证毕。

[P4086] 连续可微路径满足上述右导数条件。

[P4087 M004282,M004283,M004284] 推论D.6。 设 $\mu_{\det},f_{ref}:X_{in} \rightarrow X_{out}$ 在包含线段 $\lbrack x_{0},x\rbrack$ 的开集上可微，$X_{out}$ 完备。若

[P4089 式(D-22) M004285] $\begin{aligned}
 \parallel \mu_{\det}(x_{0}) - f_{ref}(x_{0}) \parallel & \leq \varepsilon_{val}, \\
 \parallel D\mu_{\det}(x_{0}) - Df_{ref}(x_{0}) \parallel & \leq \varepsilon_{der}, \\
 \parallel D\mu_{\det}(z) - D\mu_{\det}(x_{0}) \parallel & \leq L_{D\mu} \parallel z - x_{0} \parallel , \\
 \parallel Df_{ref}(z) - Df_{ref}(x_{0}) \parallel & \leq L_{Df} \parallel z - x_{0} \parallel 
\end{aligned}$

[P4090 式(D-22)] (D-22)

[P4091] 在线段上成立，则

[P4093 式(D-23) M004286] $\begin{aligned}
 \parallel \mu_{\det}(x) - f_{ref}(x) \parallel \leq & \varepsilon_{val} + \varepsilon_{der} \parallel x - x_{0} \parallel \\
 & + \frac{L_{D\mu} + L_{Df}}{2} \parallel x - x_{0} \parallel^{2}.
\end{aligned}$

[P4094 式(D-23)] (D-23)

[P4095 M004287] 证明。 令 $\Delta x = x - x_{0}$，构造真实误差路径

[P4097 式(D-24) M004288] $e_{path}(t) = \mu_{\det}(x_{0} + t\Delta x) - f_{ref}(x_{0} + t\Delta x).$

[P4098 式(D-24)] (D-24)

[P4099] 由链式法则，

[P4101 式(D-25) M004289] ${e_{path}}'(t) = \lbrack D\mu_{\det}(x_{0} + t\Delta x) - Df_{ref}(x_{0} + t\Delta x)\rbrack\Delta x.$

[P4102 式(D-25)] (D-25)

[P4103 M004290,M004291] 在算子差中加减 $D\mu_{\det}(x_{0})$ 和 $Df_{ref}(x_{0})$，得到

[P4105 式(D-26) M004292] $\begin{aligned}
 \parallel {e_{path}}'(t) \parallel & \leq \lbrack\varepsilon_{der} + (L_{D\mu} + L_{Df})t \parallel \Delta x \parallel \rbrack \parallel \Delta x \parallel \\
 & = \varepsilon_{der} \parallel \Delta x \parallel + (L_{D\mu} + L_{Df}) \parallel \Delta x \parallel^{2}t.
\end{aligned}$

[P4106 式(D-26)] (D-26)

[P4107 M004293,M004294] 取 $\varepsilon_{path} = \varepsilon_{der} \parallel \Delta x \parallel$、$L_{path} = (L_{D\mu} + L_{Df}) \parallel \Delta x \parallel^{2}$，应用命题D.5即可。证毕。

[P4108] 由线段构造、链式法则和导数误差界，向量情形归结为上述路径命题。

[P4109] 若在整条路径上使用统一的导数误差上界，则还可得到

[P4111 式(D-27) M004295] $\parallel \mu_{\det}(x) - f_{ref}(x) \parallel \leq \varepsilon_{val} + \varepsilon_{der} \parallel \Delta x \parallel + (L_{D\mu} + L_{Df}) \parallel \Delta x \parallel^{2}.$

[P4112 式(D-27)] (D-27)

[P4113 M004296,M004297,M004298] 这是先把半径 $\parallel \Delta x \parallel$ 范围内的导数误差统一上界为 $\varepsilon_{der} + (L_{D\mu} + L_{Df}) \parallel \Delta x \parallel$，再应用命题D.4所得。它的二次项系数为1，路径比较所得系数为 $1/2$。两者使用误差沿路径变化的信息精细程度不同，因此应保留其区别。

[P4114 M004299,M004300] 取标量函数 $\mu_{\det}(x) = c_{quad}x^{2}$，参考函数为其在 $a$ 点的一阶展开

[P4116 式(D-28) M004301] $f_{ref}(x) = c_{quad}a^{2} + 2c_{quad}a(x - a).$

[P4117 式(D-28)] (D-28)

[P4118 M004302,M004303] 此时基准函数值和导数误差均为0，$L_{D\mu} = 2|c_{quad}|$，$L_{Df} = 0$，且

[P4120 式(D-29) M004304] $|\mu_{\det}(x) - f_{ref}(x)| = |c_{quad}||x - a|^{2} = \frac{L_{D\mu}}{2}|x - a|^{2}.$

[P4121 式(D-29)] (D-29)

[P4122 M004305] 因此，在这组条件下二次项中的 $1/2$ 不能普遍进一步减小。若参考函数本来就是策略自身的局部线性化，那么该界刻画的是策略偏离其线性化的程度，而非相对于未知最优策略的误差。

[P4123] D.4 输出结构。本节分析NAU线性输出与NMU启发的二次修正结构的数学性质。以NAU为例，其理想化线性映射由各输入分量的加权和构成，权重取−1、0或1：

[P4125 式(D-30) M004306] $F_{NAU}(x) = \sum_{i = 1}^{n}c_{NAU,i}x_{i},\quad\quad c_{NAU,i} \in \{ - 1,0,1\}.$

[P4126 式(D-30)] (D-30)

[P4127 M004307,M004308] 其中各坐标的NAU权重记为 $c_{NAU,i}$。固定其余坐标，对第$j$ 个坐标求导，有

[P4129 式(D-31) M004309] $\frac{\partial F_{NAU}}{\partial x_{j}} = c_{NAU,j}.$

[P4130 式(D-31)] (D-31)

[P4131] 因此，该线性层的雅可比为常数，雅可比Lipschitz常数可取0。该性质对任意固定实权重同样成立，不要求训练权重已经精确离散为三个值。固定任意其余坐标只增加常数项，因此每个坐标方向的一维截面具有相同性质。

[P4132 M004310,M004311] 对 $F_{quad}(x) = c_{quad}x^{2}$，有 ${F_{quad}}'(x) = 2c_{quad}x$，因此下一式给出该二次函数的导数变化界。二次系数及随后引入的缩放系数均为固定实数。

[P4134 式(D-32) M004312] $|{F_{quad}}'(x) - {F_{quad}}'(y)| = 2|c_{quad}||x - y|.$

[P4135 式(D-32)] (D-32)

[P4136 M004313,M004314,M004315,M004316,M004317] 导数Lipschitz常数可取 $2|c_{quad}|$。进一步，对 $G_{quad}(x) = \omega_{quad}c_{quad}x^{2}$，常数可取 $2|\omega_{quad}c_{quad}|$。线性缩放相应地缩放了导数的Lipschitz常数。这里$c_{quad}$为二次项系数，$\omega_{quad}$为乘性缩放系数。

[P4137 M004318] 上述结论精确适用于标量二次函数。更高次乘积的导数条件取决于次数和定义域。例如 $x^{3}$ 的导数在全实线上不是Lipschitz的，但在任意给定有界区间上具有有限Lipschitz常数。

[P4138 M004319,M004320] 命题D.7。 令 $r_{ReLU}(x) = max\{ 0,x\}$。不存在有限非负常数 $L$，使在全部非零点上有

[P4140 式(D-33) M004321] $|{r_{ReLU}}'(x) - {r_{ReLU}}'(y)| \leq L|x - y|.$

[P4141 式(D-33)] (D-33)

[P4142 M004322,M004323,M004324,M004325] 证明。 当 $L = 0$ 时，取 $x = - 1$、$y = 1$，左侧为1，右侧为0，矛盾。当 $L > 0$ 时，取

[P4144 式(D-34) M004326] $x = - \frac{1}{4L},\quad\quad y = \frac{1}{4L}.$

[P4145 式(D-34)] (D-34)

[P4146 M004327] 由于负半轴导数为0、正半轴导数为1，左侧等于1，而右侧等于 $1/2$，仍矛盾。证毕。

[P4147] 证明没有使用原点处的导数约定。

[P4148 M004328,M004329,M004330,M004331] 命题D.8。 若 ${f_{ref}}'$ 在原点左、右邻域内分别恒等于常数 $d_{left}$ 和 $d_{right}$，且 $d_{left} \neq d_{right}$，则不存在覆盖该邻域的有限导数Lipschitz常数。

[P4149 M004332,M004333,M004334,M004335,M004336,M004337,M004338] 证明。 假设存在常数 $L \geq 0$。取足够小的距离 $t > 0$，使 $- t$ 和 $t$ 位于给定左右邻域内，并满足 $2Lt < |d_{left} - d_{right}|$（当 $L = 0$ 时任取位于邻域内的 $t > 0$）。则

[P4151 式(D-35) M004339] $|d_{left} - d_{right}| = |{f_{ref}}'( - t) - {f_{ref}}'(t)| \leq 2Lt < |d_{left} - d_{right}|,$

[P4152 式(D-35)] (D-35)

[P4153] 矛盾。证毕。

[P4154] 导数跳跃与函数本身是否满足Lipschitz条件属于不同性质：ReLU本身是1-Lipschitz函数，有限权重的有限层ReLU网络仍具有有限的函数Lipschitz常数，其输出幅度至多随输入范数线性增长。

[P4155] D.5 复合结构。

[P4156 M004340,M004341,M004342] 命题D.9。 若标量函数 $g$ 可微，且 $|g'(x)| \leq B_{g}$ 对所有 $x$ 成立，则

[P4158 式(D-36) M004343] $|g(x) - g(y)| \leq B_{g}|x - y|.$

[P4159 式(D-36)] (D-36)

[P4160 M004344] 证明。 当 $x = y$ 时显然成立；否则在二者之间应用中值定理。证毕。

[P4161 M004345] 函数的Lipschitz性质在此由导数幅度上界保证。导数Lipschitz本身不蕴含函数在全实线上Lipschitz，例如 $g(x) = x^{2}$ 的导数Lipschitz常数为2，而导数幅度不受全局常数控制。

[P4162 M004346] 命题D.10。 设 $f_{out},g$ 可微，并满足

[P4164 式(D-37) M004347] $\begin{aligned}
|{f_{out}}'(u) - {f_{out}}'(v)| & \leq L_{out}|u - v|, \\
|g(x) - g(y)| & \leq K_{g}|x - y|, \\
|g'(x)| & \leq B_{g},\quad\quad|{f_{out}}'(u)| \leq B_{out}, \\
|g'(x) - g'(y)| & \leq L_{g}|x - y|.
\end{aligned}$

[P4165 式(D-37)] (D-37)

[P4166 M004348] 则 $f_{out} \circ g$ 的导数Lipschitz常数可以取

[P4168 式(D-38) M004349] $L_{f_{out} \circ g} = L_{out}K_{g}B_{g} + B_{out}L_{g}.$

[P4169 式(D-38)] (D-38)

[P4170 M004350,M004351] 证明。 链式法则给出 $(f_{out} \circ g)'(x) = {f_{out}}'(g(x))g'(x)$。加减 ${f_{out}}'(g(y))g'(x)$，有

[P4172 式(D-39) M004352] $\begin{aligned}
 & (f_{out} \circ g)'(x) - (f_{out} \circ g)'(y) \\
 = & \lbrack{f_{out}}'(g(x)) - {f_{out}}'(g(y))\rbrack g'(x) \\
 & + {f_{out}}'(g(y))\lbrack g'(x) - g'(y)\rbrack.
\end{aligned}$

[P4173 式(D-39)] (D-39)

[P4174] 取绝对值并分别估计，得到

[P4176 式(D-40) M004353] $\begin{aligned}
 & |(f_{out} \circ g)'(x) - (f_{out} \circ g)'(y)| \\
 \leq & L_{out}|g(x) - g(y)|B_{g} + B_{out}L_{g}|x - y| \\
 \leq & (L_{out}K_{g}B_{g} + B_{out}L_{g})|x - y|.
\end{aligned}$

[P4177 式(D-40)] (D-40)

[P4178 M004354,M004355] 证毕。当 $B_{out} = 1$ 时，该上界化为 $L_{out}K_{g}B_{g} + L_{g}$。

[P4179 M004356] 对向量映射，链式法则为 $D(f_{out} \circ g)(x) = Df_{out}(g(x))Dg(x)$。按上述方式加减中间项，并使用算子范数的次乘性，可得到相同常数形式。这是对标量论证的直接数学推广。

[P4180 M004357,M004358,M004359,M004360,M004361,M004362,M004363,M004364,M004365,M004366,M004367] 特征函数的Lipschitz常数 $K_{g}$ 不能替代其一阶导数的Lipschitz常数 $L_{g}$。例如取 $f_{out}(u) = u$、$g(x) = max\{ 0,x\}$，虽有 $L_{out} = 0$、$K_{g} = 1$，但 $f_{out} \circ g$ 的导数在原点两侧发生阶跃跳变；同理，若内部特征网络 $g$ 在原点处存在左右斜率不等的折角且 $g(0) = 0$，由于 $\tanh'(0) = 1$，外层叠加平滑变换 $tanh \circ g$ 在原点两侧仍保留不同的单侧导数。因此，唯当内部特征变换与外层输出映射同时满足可微与导数Lipschitz条件时，方可应用上述二阶泰勒误差界。

[P4181 M004368,M004369] D.6 固定符号惩罚下价值算子的基本性质。设状态、动作和模式集合 $\mathcal{S,A,Z}$ 均有限非空。令 $0 \leq \gamma < 1$，模式条件转移核满足

[P4183 式(D-41) M004370] $P_{z}(s' \mid s,a) \geq 0,\quad\quad\sum_{s'}^{}P_{z}(s' \mid s,a) = 1.$

[P4184 式(D-41)] (D-41)

[P4185 M004371,M004372,M004373,M004374] 给定固定奖励 $R_{z}$、认知不确定性惩罚 $\Gamma_{epi,z}$、符号导数偏差惩罚 $\Gamma_{sym,z}$ 以及非负系数 $\lambda_{epi},\lambda_{sym}$，记

[P4187 式(D-42) M004375] $V_{Q}(s) = \max_{a}Q(s,a),$

[P4188 式(D-42)] (D-42)

[P4190 式(D-43) M004376] $\begin{array}{r}
(\mathcal{T}_{z}Q)(s,a) = R_{z}(s,a) + \gamma\lbrack\sum_{s'}^{}P_{z}(s' \mid s,a)V_{Q}(s') \\
 - \lambda_{epi}\Gamma_{epi,z}(s,a) \\
 - \lambda_{sym}\Gamma_{sym,z}(s,a) - \kappa_{fix}\rbrack.
\end{array}$

[P4191 式(D-43)] (D-43)

[P4192 M004377,M004378] 给定固定模式信念 $\rho(z) \geq 0$、$\sum_{z}^{}\rho(z) = 1$，CS-BAPR的置信加权算子定义为

[P4194 式(D-44) M004379] $(\mathcal{T}_{CS}Q)(s,a) = \sum_{z}^{}\rho(z)(\mathcal{T}_{z}Q)(s,a).$

[P4195 式(D-44)] (D-44)

[P4196 M004380,M004381] 沿用附录C.2的混合随机转移核 ${\overline{P}}_{\rho}$，将与输入价值函数 $Q$ 无关的固定项合并记为

[P4198 式(D-45) M004382] $\begin{array}{r}
{\overline{R}}_{pen,CS}(s,a) = \sum_{z}^{}\rho(z)\lbrack R_{z}(s,a) - \gamma\lambda_{epi}\Gamma_{epi,z}(s,a) \\
 - \gamma\lambda_{sym}\Gamma_{sym,z}(s,a) - \gamma\kappa_{fix}\rbrack.
\end{array}$

[P4199 式(D-45)] (D-45)

[P4200 M004383,M004384] 命题D.11。 在D.6节的固定信念与固定惩罚条件下，$\mathcal{T}_{CS}$ 单调，满足折扣平移性质，且为无穷范数下的 $\gamma$-压缩映射：

[P4202 式(D-46) M004385] $\parallel \mathcal{T}_{CS}Q_{1} - \mathcal{T}_{CS}Q_{2} \parallel_{\infty} \leq \gamma \parallel Q_{1} - Q_{2} \parallel_{\infty}.$

[P4203 式(D-46)] (D-46)

[P4204 M004386,M004387,M004388] 证明。 按有限求和重排可得 $\mathcal{T}_{CS}Q = {\overline{R}}_{pen,CS} + \gamma{\overline{P}}_{\rho}V_{Q}$。由命题C.1知 ${\overline{P}}_{\rho}$ 为合法随机核，而单步冻结的符号惩罚并入 ${\overline{R}}_{pen,CS}$ 后在两输入相减时精确抵消，故直接满足定理A.1的条件，唯一不动点及精确迭代的几何误差界由推论A.1给出。证毕。

[P4205 M004389] 在式(D-4e)的小批量更新中，通过停止梯度算子 $sg({\bar{\mathcal{L}}}_{JC}')$ 阻断符号导数误差项对价值网络参数的耦合反馈，即可使单步更新内的符号惩罚项作为固定标量进入自举目标。

[P4206 M004390,M004391,M004392,M004393] D.7 符号惩罚耦合于价值输入时的反馈影响。给定非负似然证据 $L(z)$，当归一化常数 $Z_{norm} = \sum_{z}^{}\rho(z)L(z) > 0$ 时，更新后的信念 $\rho^{+}(z) = \rho(z)L(z)/Z_{norm}$ 由命题C.2保持非负归一化，在下一单步更新内固定 $\rho^{+}$ 仍满足命题D.11。

[P4207 M004394,M004395,M004396,M004397] 反之，若在单状态单动作模型中令符号惩罚随当前在训价值输入线性变化：$\Gamma(Q_{sc}) = C_{pen} - \mu_{fb}Q_{sc}$（其中 $\mu_{fb} \geq 0$），合并与 $Q_{sc}$ 无关的常数项 $R_{0}$ 后，更新算子变为

[P4209 式(D-47) M004398] $T_{symfb}(Q_{sc}) = R_{0} + \gamma(1 + \lambda_{sym}\mu_{fb})Q_{sc}.$

[P4210 式(D-47)] (D-47)

[P4211 M004399] 记有效距离因子为 $c_{symfb} = \gamma(1 + \lambda_{sym}\mu_{fb}) \geq 0$，则有

[P4213 式(D-48) M004400] $|T_{symfb}(Q_{sc,1}) - T_{symfb}(Q_{sc,2})| = c_{symfb}|Q_{sc,1} - Q_{sc,2}|.$

[P4214 式(D-48)] (D-48)

[P4215 M004401,M004402,M004403] 当 $0 < \gamma < 1$、$\lambda_{sym} > 0$ 时，保持严格收缩（$c_{symfb} < 1$）的临界反馈斜率阈值为

[P4217 式(D-49) M004404] $\mu_{fb,crit} = \frac{1/\gamma - 1}{\lambda_{sym}}.$

[P4218 式(D-49)] (D-49)

[P4219 M004405,M004406,M004407,M004408,M004409,M004410] 例如当 $\gamma = 0.99$、$\lambda_{sym} = 0.01$ 时，$\mu_{fb,crit} \approx 1.0101$；一旦反馈斜率 $\mu_{fb} \geq \mu_{fb,crit}$（取 $C_{pen} \geq \mu_{fb}$ 可同时保证 $\Gamma(0),\Gamma(1) \geq 0$），算子即丧失严格收缩性质。

[P4220 M004411,M004412,M004413,M004414,M004415] D.8 符号基函数展开的导数误差传递关系。本节在动作映射层面引入有限基函数展开模型，分析其与参考动作映射 $f_{ref}$ 及学习策略映射 $\mu_{\det}$ 之间的导数误差传递关系。设 $\varphi_{i}:X_{in} \rightarrow X_{out}$（$i = 1,\ldots,n$）为可微基函数，实系数向量为 $c_{coef}$，对应的线性组合动作模型记为

[P4222 式(D-50) M004416] $f_{c_{coef}}(x) = \sum_{i = 1}^{n}c_{coef,i}\varphi_{i}(x).$

[P4223 式(D-50)] (D-50)

[P4224] 有限和与标量乘法保持可微性，因而

[P4226 式(D-51) M004417] $Df_{c_{coef}}(x) = \sum_{i = 1}^{n}c_{coef,i}D\varphi_{i}(x).$

[P4227 式(D-51)] (D-51)

[P4228] 可微性只依赖基函数及有限线性组合。系数向量的稀疏性则需要通过估计时的约束或正则化获得。

[P4229 M004418,M004419] 命题D.12。 假设同一个系数向量 $c_{coef}$ 在区域 $X_{eval}$ 满足

[P4231 式(D-52) M004420] $\begin{aligned}
 \parallel Df_{ref}(x) - Df_{c_{coef}}(x) \parallel & \leq \varepsilon_{sym}, \\
 \parallel D\mu_{\det}(x) - Df_{c_{coef}}(x) \parallel & \leq \varepsilon_{pol}.
\end{aligned}$

[P4232 式(D-52)] (D-52)

[P4233 M004421,M004422] 其中，$\varepsilon_{sym}$为符号模型与参考映射的导数误差上界，$\varepsilon_{pol}$为策略与同一符号模型的导数误差上界。则

[P4235 式(D-53) M004423] $\parallel D\mu_{\det}(x) - Df_{ref}(x) \parallel \leq \varepsilon_{sym} + \varepsilon_{pol},\quad\quad x \in X_{eval}.$

[P4236 式(D-53)] (D-53)

[P4237 M004424] 证明。 加减 $Df_{c_{coef}}(x)$，取范数并使用三角不等式即可。证毕。

[P4238 M004425,M004426,M004427] 使用同一 $c_{coef}$ 是这一组合关系的实质条件。如果策略拟合的是另一模型 $f_{\widetilde{c_{coef}}}$，还需要额外控制 $\parallel Df_{c_{coef}} - Df_{\widetilde{c_{coef}}} \parallel$，不能直接省略这一误差。

[P4239 M004428,M004429] 若上述两个误差界覆盖从 $x_{0}$ 到 $x$ 的整条线段，并满足基准函数值误差界，则由命题D.4有

[P4241 式(D-54) M004430] $\parallel \mu_{\det}(x) - f_{ref}(x) \parallel \leq \varepsilon_{val} + (\varepsilon_{sym} + \varepsilon_{pol}) \parallel x - x_{0} \parallel .$

[P4242 式(D-54)] (D-54)

[P4243] 若两个误差条件在全空间成立，上式对任意外推路径成立。符号模型在训练区域的误差界，与其在整条外推路径的误差界，是不同的信息。

[P4244 M004431,M004432,M004433,M004434,M004435] D.9 环境一致性。记所考察的环境集合为 $\mathcal{E}$，每个环境 $e\mathcal{\in E}$ 对应参考映射 $f_{e}$ 和状态区域 $D_{e}$。对各环境使用同一系数向量 $c_{coef}$ 时，其导数一致性条件定义为

[P4246 式(D-55) M004436] $\parallel Df_{e}(x) - Df_{c_{coef}}(x) \parallel \leq \varepsilon_{IRM},\quad\quad e\mathcal{\in E,}\ x \in D_{e},$

[P4247 式(D-55)] (D-55)

[P4248 M004437] 式中，$\varepsilon_{IRM}$ 表示该共同符号模型在所列各环境中的统一导数误差上界。若策略映射同时满足

[P4250 式(D-56) M004438] $\parallel D\mu_{\det}(x) - Df_{c_{coef}}(x) \parallel \leq \varepsilon_{pol},\quad\quad e\mathcal{\in E,}\ x \in D_{e},$

[P4251 式(D-56)] (D-56)

[P4252] 则逐环境应用命题D.12，得到

[P4254 式(D-57) M004439] $\parallel D\mu_{\det}(x) - Df_{e}(x) \parallel \leq \varepsilon_{IRM} + \varepsilon_{pol}.$

[P4255 式(D-57)] (D-57)

[P4256 M004440] 进一步假设，对每个所考察环境，基准点满足$\parallel \mu_{\det}(x_{0}) - f_{e}(x_{0}) \parallel \leq \varepsilon_{val}$，且式(D-57)的导数误差界在连接该基准点与待评价点的整条线段上成立。上述基准误差上界对所有所考察环境统一成立时，由命题D.4得到

[P4258 式(D-58) M004441] $\parallel \mu_{\det}(x) - f_{e}(x) \parallel \leq \varepsilon_{val} + (\varepsilon_{IRM} + \varepsilon_{pol}) \parallel x - x_{0} \parallel .$

[P4259 式(D-58)] (D-58)

[P4260] 考虑数学分解

[P4262 式(D-59) M004442] $f_{e}(x) = f_{sh}(x) + f_{s,e}(x),$

[P4263 式(D-59)] (D-59)

[P4264 M004443,M004444,M004445,M004446] 其中$f_{sh}$是真实映射中在各环境共享的部分，$f_{s,e}$为环境相关部分；前文的$f_{c_{coef}}$仍表示由数据拟合的符号模型。以$\varepsilon_{sh}$表示符号模型相对共享真实部分的导数误差上界，假设

[P4266 式(D-60) M004447] $\parallel Df_{c_{coef}}(x) - Df_{sh}(x) \parallel \leq \varepsilon_{sh},$

[P4267 式(D-60)] (D-60)

[P4268 M004448] 并且在测试环境中 $\parallel Df_{s,test}(x) \parallel \leq \varepsilon_{s,test}$。则

[P4270 式(D-61) M004449] $\begin{aligned}
 \parallel Df_{c_{coef}}(x) - Df_{test}(x) \parallel & \leq \parallel Df_{c_{coef}}(x) - Df_{sh}(x) \parallel + \parallel Df_{s,test}(x) \parallel \\
 & \leq \varepsilon_{sh} + \varepsilon_{s,test}.
\end{aligned}$

[P4271 式(D-61)] (D-61)

[P4272] 这一分解将符号模型对共享部分的估计误差与环境特有扰动分开。若再获得策略对该符号模型的导数拟合误差，还可继续用三角不等式组合。

[P4273] D.10 误差传播。

[P4274 M004450] 命题D.13。 设已知某个基准点 $a$ 的导数误差满足

[P4276 式(D-62) M004451] $|{\mu_{\det}}'(a) - {f_{ref}}'(a)| \leq \varepsilon_{emp} + \Delta_{gen},$

[P4277 式(D-62)] (D-62)

[P4278 M004452,M004453,M004454] 其中，$\varepsilon_{emp}$和$\Delta_{gen}$分别表示给定的经验导数误差界及其推广误差界。右侧非负，并且函数值基准误差满足命题D.2的条件，策略导数与参考导数的正则性满足命题D.3的条件。则对 $b \geq a$，

[P4280 式(D-63) M004455] $\begin{aligned}
|\mu_{\det}(b) - f_{ref}(b)| \leq & \varepsilon_{val} + (\varepsilon_{emp} + \Delta_{gen})(b - a) \\
 & + \frac{L_{D\mu} + L_{Df}}{2}(b - a)^{2}.
\end{aligned}$

[P4281 式(D-63)] (D-63)

[P4282 M004456] 证明。 将命题D.3和命题D.2中的 $\varepsilon_{der}$ 替换为给定非负和即可。证毕。

[P4283] 这是以基准点误差为前提的确定性传播界。若统计估计能够给出该点的误差控制，可将相应结论代入此式。

[P4284] D.11 适用范围。本附录的导数误差结论适用于完整策略映射满足正则性条件的情形。基准误差、导数差异和外推距离共同决定函数误差的上界；动作误差与闭环累计回报之间还需要环境动力学和价值函数的敏感性条件。公交需求倍数与归一化状态空间中的距离具有不同含义，应用误差界时应先建立两者的对应关系。

[P4285] 固定置信算子的收缩性描述精确价值更新。置信、惩罚和价值同时变化时，需要进一步控制算子漂移及函数逼近误差。算术输出配置中的NAU线性层和NMU启发的二次函数需通过完整复合映射分析；不含NAU的软符号配置则使用第D.1节给出的完整动作及参考映射，二者的正则性不能互相代用。

[P4286] 符号模型与多环境误差分解均以给定的拟合误差条件为基础。稀疏系数的恢复、基准点误差的统计估计以及环境共享成分的识别分别属于估计问题。三角不等式比较的是可用误差上界，两个上界的大小不能单独确定两种估计方法的实际优劣。平均经验误差也不能直接代替指定点上的导数误差。