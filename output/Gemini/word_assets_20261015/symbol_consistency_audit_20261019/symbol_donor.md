$$\mathcal L_{od},l$$

OD对($o$,$d$)的可服务线路集合及线路索引（式5-5）

$$v,i,j$$

车辆索引$v$；同车事件序号$i$（2.7.1节）；班次索引$i$与站点序号$j$（3.2.2节）

$$T_{\mathrm{hor}}$$

规定评价与控制时域的终点

$$t^{d},t^{s},y_{i,j}^{a},y_{i,j}^{d}$$

本站计划离站时刻、完成必要上下客的时刻；班次$i$在站点$j$的实际到站与离站时刻（第2、3章）

$$h_f,h_b,\widehat h_b$$

前向、后向车头时距及决策时刻可得的后向车头时距估计值

$$h_f^*,h_b^*,h_s$$

前向与后向计划间隔；单线路等间隔场景的共同计划间隔$h_s$

$$a,a_{\max}$$

以秒计的附加驻站时长及其上限

$$\widetilde a$$

策略网络内部的归一化附加驻站动作，取值范围[-1,1]

$$d_{i,j}^{svc},d_{i,j}^{run},d_{i,j}^{exit},d^{svc}(h)$$

第$i$个班次在第$j$个站点的必要上下客服务、驶往下一站的道路行驶及出站驶离等待耗时（第3.2.2节）；第4.2.1节以$d^{svc}(h)$表示随车头时距变化的必要服务耗时

$$\lambda,\eta^{board}$$

局部乘客到达率与单位乘客上车服务时间（第3、4章；OD分时到达率另依式5-5定义）

$$N^{w}(t),N^{b}(t)$$

物理时刻$t$全系统候车与车内乘客人数

$$C_{\mathrm{veh}},n$$

车辆容量；局部乘客成本解析中已在车内的乘客人数$n$（3.2.6节）

$$W,T_p^w,T_p^b$$

随机到达乘客的候车时间$W$；乘客$p$在规定评价时域内累计的候车与车内时间

$$N_{\mathrm{dep}},\mathcal P_{\mathrm{dep}}$$

规定评价时域内进入系统的独立出行乘客数及其集合

$$H,\mathrm{CV}_H$$

随机车头时距及其变异系数

$$\widehat Q_{\min}$$

两个目标动作价值网络输出的较小估计（5.2节）

$$\xi,\bar\xi$$

第6章对数置信评价网络及其目标网络的参数；第2章的同形$\xi$另为CVaR成本阈值

$$\mathcal L_Q,\mathcal L_\pi,\mathcal L_V$$

动作价值、策略与状态价值的训练损失；各章按相应目标定义

$$\gamma,\Gamma_k,d_t,d_k,d_i,d$$

回报折扣因子、事件延续因子与回合终止标记；终止标记索引随对应事件定义

$$p^\circ,\mathcal P_{s,a},\mathcal U_{s,a}$$

固定状态动作对的名义后继分布及允许后继分布集合；主文用$\mathcal P$，附录$A$另用$\mathcal U$

$$\sigma_{\mathcal P_{s,a}}(u),\sigma_{\mathcal U}(f),\Omega_{\mathcal U}(V)$$

允许分布集合相对于名义分布的支持函数及相应价值正则项；集合记号按章节区分

$$W_1,d_{\mathcal S},L_V,\varepsilon_{\mathrm{ipm}},\rho_{\mathrm{ipm}},\operatorname{Lip}(V)$$

一阶Wasserstein距离、状态距离、价值Lipschitz上界及分布距离半径；第3章相应记作$\rho_{ipm}$与Lip($V$)

$$C_{\mathrm{traj}},\mathbb P_0,Z$$

轨迹累计成本与固定策略下名义轨迹分布；第2.3节的累计成本（及分布RL语境中的随机回报）记为$Z$

$$\beta_{\mathrm{tail}},\eta_{\mathrm{risk}},\zeta_{\mathrm{risk}},\mathcal Z_{\beta_{\mathrm{tail}}},\nu,\xi,\zeta$$

CVaR尾部概率质量、成本阈值、轨迹重加权密度及风险包络；第2.3节前三者分别用$\nu$、$\xi$、$\zeta$

$$\mathcal D,\mathcal D_{\mathrm{off}},\mathcal D_{\mathrm{sim}}$$

通用经验回放数据、固定目标域离线数据与源仿真交互数据

$$\boldsymbol{x},\boldsymbol{o},E_j$$

拼接后的网络输入、连续运行观测与第$j$类离散变量的嵌入矩阵（式3-4至3-6）

$$K$$

动作价值网络集成的成员数，各章按具体方法设置

$$\beta_{\mathrm{bc}},\beta_{\mathrm{ood}},\beta_{\mathrm{lcb}}$$

行为克隆辅助项、当前在线价值集成分歧惩罚与策略下置信界保守评价的系数

$$\mathcal L_{bc}$$

行为克隆辅助项的未加权动作平方误差（式4-18）

$$\mathcal Z,z,b(z),\rho(z),G_z$$

候选运行模式集合、模式元素、依据历史的模式信念$b(z)$、冻结备份的模式权重$\rho(z)$及模式条件执行增益

$$\chi_j^{(r)},\chi_j^{(q)},\chi_j^{(w)},\chi_j,{\widetilde\chi}_j,\mu_j,v_j$$

BAPR的奖励偏差、集成分歧增量和权重范数漂移三项诊断量，其最大值与指数平滑值，以及诊断窗口奖励均值和方差（第4.2.3节）

$$\mathcal l,\rho_j(\mathcal l),L_j(\mathcal l),\lambda_w,\beta_{eff},p_{reset}$$

BAPR历史运行长度、运行长度权重分布、长度似然、反馈证据权重、动态保守系数与变化点先验重置率（第4.2.3及4.3.1节）

$$H_{k},\Psi(z^{\prime}\mid z)$$

模式推断使用的状态动作历史（第4.2.2节、附录$B$.4和$B$.6）与隐模式转移概率（式4-6）

$$\mathcal T,\mathcal B,V_Q,V^Q,\Lambda(s,a)$$

贝尔曼更新映射；$\mathcal B$表示有界函数空间或相应局部反例算子；动作价值聚合；单步更新内冻结的合成保守惩罚Λ

$$c_{pen},c_{pen}(s,a),\mathcal{V}_{\pi}Q$$

第4章及附录$C$中的反馈惩罚放大系数$c_{pen}$；附录$A$中的固定有效惩罚函数$c_{pen}(s,a)$；附录$A$固定策略下的含熵动作价值平均$\mathcal V_\pi Q$

$${\widehat y}_{res}(x),{\widehat y}_{dir}(x)$$

残差世界模型与直接目标世界模型的预测输出（第6.2.3节）

$$R_m,m^{\star}$$

候选模型$m$在独立模型选择验证集上的综合预测误差及所选模型类别（第6.3.1节）

$$\widehat\Delta(x),{\widehat f}_{dir}(x)$$

利用目标域离线数据学习的残差修正网络，以及不依赖源预测的直接目标预测网络（第6.2.3节）

$$\mathcal L_{wm},\sigma_{wm},\mathcal I_{cal}$$

公交混合事件世界模型的六类综合预测损失、置信网络输入的世界模型逐维预测标准差，以及独立模型选择验证集（第6.2.2–6.3.2节）

$$\tau_e,c_i,Q_{\xi}^{\Delta},Q_{\bar\xi}^{\Delta},w_i^{\Delta},e_i$$

第6章的局部预测误差尺度、局部可靠性评分、参数化对数置信评价网络及其目标网络、模型样本置信权重和综合预测误差

$$c,c_{const}$$

置信尺度分析中假设恒定的局部可靠性评分：第6.3.4节记$c$，附录$E$.9记$c_{const}$

$$q_i^{norm},Q_{norm}^{\Delta},w_{norm}^{\star},{\widetilde Q}^{\Delta,*}$$

第6章归一化对数置信递推目标、恒定局部评分情形的归一化置信不动点及其指数权重；附录$E$.9一般情形的归一化置信不动点

$$y_{mix},{\widehat y}_i,{\widehat y}_{mix}$$

双域真实转移的混合自举目标（第5–6章）、一步模型样本的标量价值目标，以及目标世界模型直接替换源样本目标时的标量目标（第6章）

$$\mathcal l_H,\tau_{IQL},\tau_e$$

Huber损失及IQL期望分位水平；IQL在第2章记$\tau_e$、第5章记$\tau_{IQL}$，第6章$\tau_e$另为预测误差尺度

$$\|\cdot\|_\infty,\|\cdot\|_2,\|\cdot\|_F$$

最大范数、欧氏范数与矩阵Frobenius范数

$$r_k,R(s,a)$$

一次事件区间的实际奖励与给定状态动作下的期望即时奖励；二者通过条件期望联系

$$\tau,\tau_h$$

$\tau$为乘客积分时刻（第5章）或目标网络软更新系数（第3章）；$\tau_h$为式3-11奖励衰减尺度，后续实验简写为$\tau$

$$f_{sh},\varepsilon_{sh}$$

跨环境共享真实映射与其相对于拟合符号模型的导数误差上界（附录$D$）

$$\mathbb I\{\cdot\},\mathbf1\{\cdot\},\operatorname{clip},\odot$$

条件示性函数的两种实际书写、区间截断和逐元素乘积

$$\phi_k,\phi_k^\prime,\widehat Q_{\phi_k^\prime}$$

第$k$个当前动作价值网络参数、对应目标网络参数及目标动作价值函数

$$m_{dem}$$

相对于基准客流到达强度的需求倍率（第4章）

$$\delta_{mis}$$

第6.4.1节源仿真四类标定参数联合偏差的幅度比例

$$m,m_{grp},M$$

状态中的离散类别变量个数$m$（第3.2.3节）；同一物理时刻到站车组内需依次决策的车辆数$m_{grp}$（第5.2.2节）；同一状态比较的候选驻站动作数$M$（第3.3.3节）

$$c_j,n_j,h,d_j,n,d$$

类别取值、类别数、隐藏层宽度与嵌入维数；单类别分析以$n$、$d$分别记类别数与嵌入维数（第3章）

$$N,N_v$$

共享经验回放中的转移总数及车辆$v$贡献的转移数（3.2.2节）

$$\delta,\rho,V$$

线性后继分布扰动、扰动范数约束半径及后继状态价值向量（2.3节）

$$\varepsilon_{\mathrm{dist}},\varepsilon,\varepsilon_{\mathrm{act}},\varepsilon_{\mathrm{path}}$$

最大范数距离；逐次逼近误差的统一上界$\varepsilon$（3-36）；动作相关剩余误差上界$\varepsilon_{act}$（3-40）；误差路径导数界的常数项

$$G,H,\mathcal X_{feat}$$

第4.3.2节复合策略表示$H\circ G$中的特征网络、输出映射及中间特征空间（此处$G$不同于乘客广义时间成本$G$）

$$f_{c_{\mathrm{coef}}},c_{\mathrm{coef}},c_{\mathrm{coef},i}$$

拟合符号模型、其系数向量及第$i$个系数分量

$$V_{n,k}^{O,\mathrm{path}},Q_{n,k}^{O,\mathrm{path}},\Delta J,\Delta V$$

固定外生序列下的参照路径价值及动作价值、路径回报差与期望价值差；$n$、$k$为对应路径推导的步数和时刻索引

$$\mathcal L,\pi_{off},\pi_{on},\pi_{cf}$$

第5.2.1节的学习映射，以及仅用固定离线数据、追加目标环境交互、追加源仿真交互三种条件下得到的策略

$$\mu_\theta(s)$$

DDPG中参数化的确定性策略（式2-9）

$$X_{\delta},L_{adj},L_{occ}$$

由状态动作对及后继状态的概率残差组成的实向量空间，以及策略调整映射和访问分布映射的Lipschitz常数（附录$E$.5）

$$\varphi_{cls}$$

静态分类信号（附录$E$.6）

$$\lambda_{ale},\lambda_{epi},\kappa_k,\kappa_k^{\prime},\kappa_{ale}$$

范数项与集成方差项系数；第3章$\kappa_k$为目标网络权重范数，第4章$\kappa_k$、$\kappa_k^\prime$分别为当前与目标网络范数；附录$A$的$\kappa_{ale}$为加权固定正则量

$$B,K_{\max},n_{rem}$$

训练小批量大小、有限事件数上界及剩余事件数；附录$E$.5中$B$另为回归拟合映射的Lipschitz常数

$$w_{sim},c_{NAU,i}$$

第5章条件转移比率形成的源域样本损失权重，以及附录$D$的NAU线性权重

$$V_{soft},B_{mix}$$

当前策略在后继状态上的软状态价值，以及与冻结离线状态价值组合后的混合后继价值（第5.2.3节）

$$A(s,a),w_{IQL}(s,a)$$

IQL样本内优势残差及其形成的数据内动作优势权重（第2章及第5.2.3节）

$$I_f,I_b$$

前向与后向同线路邻车存在的示性标记（第3.2.4节）

$$\mathcal l_{sym},\mathcal L_{JC},\bar{\mathcal L}_{JC}^{\prime}$$

逐状态归一化雅可比导数误差、当前训练批次均值及后继状态批次均值（式4-15、4-16及$D$-4$c$至$D$-4$e$）

$$\beta_0,x_{ref}(x),f_{\theta}(x)$$

CS-BAPR的可学习全局基准动作潜变量、计划均衡参考输入，以及参考状态归零差分中的神经网络分支（第4.2.3节）

$$\lambda_{jc},\lambda_{sym},\tau_{boltz},\sigma_r$$

CS-BAPR策略雅可比正则系数、价值目标中的符号误差惩罚系数、软最小化参考动作温度和奖励标准差缩放因子（第4.2.3–4.3.1节）

$$\zeta,\zeta(u)$$

第2章CVaR的非负重加权密度；4.3.3节连接基准点与评价点的线段（式4-23），两者作用域不同

$$\Delta d$$

第4.3.3节状态外推偏移向量$\mathrm{\Delta}d$=$x$−$x_0$，用于4-23/4-24的动作误差展开及上界