"""Resolve the remaining scoped names and add local definitions."""

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2] / "manuscript"
paths = sorted((ROOT / "chapters").glob("*.md")) + sorted(p for p in (ROOT / "appendices").glob("*.md") if p.name[0] in "ABCDE")
docs = {p.name.split("_")[0]: p.read_text() for p in paths}
MATH = re.compile(r"\$\$.*?\$\$|(?<!\$)\$(?!\$).*?(?<!\$)\$(?!\$)", re.S)


def change(key, old, new):
    assert old in docs[key], (key, old)
    docs[key] = docs[key].replace(old, new)


def math_change(key, mapping, start=None, end=None):
    text = docs[key]
    a = text.index(start) if start else 0
    b = text.index(end, a+1) if end else len(text)
    pattern = re.compile("|".join(re.escape(k) for k in sorted(mapping, key=len, reverse=True)))
    docs[key] = text[:a] + MATH.sub(lambda m: pattern.sub(lambda x: mapping[x[0]], m[0]), text[a:b]) + text[b:]


def add(key, heading, sentence):
    change(key, heading, heading+"\n\n"+sentence)


math_change("05", {r"{\xi_{\min}}_": "u_"})
math_change("05", {r"\mathrm da": r"\mathrm d a^{\mathrm n}"}, "### 5.4.3", "### 5.4.4")
math_change("05", {r"\tau": r"\tau_{\mathrm{IQL}}", "L_": r"\ell_", "u": r"e_{\mathrm{IQL}}"}, "离线价值拟合的非对称平方损失", "源阶段先进行")
math_change("A", {r"{\lambda_{\mathrm{fb}}}_{\mathrm{epi}}": r"\lambda_{\mathrm{epi}}", r"{\lambda_{\mathrm{fb}}}_{\mathrm{ale}}": r"\lambda_{\mathrm{ale}}"})
math_change("C", {r"\overline {\rho^{\mathrm{joint}}}_\rho": r"\overline R_{\mathrm{pen},\rho}"})
math_change("04", {r"\mathcal H": r"\mathcal Z", "(h)": "(z)", r"\ell_h": r"\ell_z"}, "早期BAPR采用", "### 4.2.2")
for key in ("C", "D"):
    math_change(key, {r"\mathcal H": r"\mathcal Z"})
math_change("03", {"$w$": "$w_{\mathrm{hold}}$", "w\\exp": r"w_{\mathrm{hold}}\exp", r"w\geq": r"w_{\mathrm{hold}}\geq"}, "### 3.2.2", "### 3.2.3")
math_change("06", {r"\mathrm{TV}": r"\operatorname{TV}"})
math_change("E", {"r(s,a)": "R(s,a)", "$r$": "$R$"}, "## E.1", "## E.5")

# Numeric feature position and reward headway difference are separate objects.
add("02", "### 2.4.4 间隔规律性与乘客时间的条件联系", r"变异系数（Coefficient of Variation，CV）是标准差与均值之比。下式的$H$为随机服务间隔，$W$为随机乘客等待时间，$\mathrm{CV}_H$为该间隔分布的变异系数。")
add("03", "### 3.2.1 车头时距对称与计划对齐", r"本节的$h_0$为计划目标间隔，$\varepsilon_h>0$为时距比值分母的稳定常数，$\omega_h$为前后时距相对差，$r_{\mathrm E}$表示用于对照的指数型奖励。")
math_change("03", {r"\varepsilon": r"\varepsilon_h", r"\omega": r"\omega_h"}, "设计划目标", "### 3.2.2") if "设计划目标" in docs["03"] else None
add("03", "### 3.3.3 固定目标偏移与评论家更新", r"$W_l^{(\phi'_k)}$为第$k$个目标评价器第$l$层的权重矩阵，$\phi'_k$表示目标网络参数；$\|\cdot\|_1$对矩阵各元素取绝对值后求和。$\lambda_{\mathrm{ale}}$与$\lambda_{\mathrm{epi}}$分别控制参数范数偏移和集成分歧惩罚，$\sigma_{\mathrm{ens}}$表示价值集成标准差，与前文$\Gamma_{\mathrm{epi}}$满足平方关系。")
add("04", "### 4.3.3 分布外响应的条件误差界", r"$x$为待评价输入，$x_0$为基准输入，$f$为参考控制映射；$\delta$为基准函数误差上界，$\varepsilon$为基准导数误差上界，$L_\pi,L_f$为两个导数映射的Lipschitz常数，$\Delta x=x-x_0$为输入偏移。")
add("05", "### 5.4.4 离线初始化与源仿真续训", r"$Q_{\min}$为双评价器较小值，$V_\nu$为状态价值网络，$A(s,a)$为相对优势；$B_{\mathrm{SAC}}$为熵正则化后继价值，$B_{\mathrm{mix}}$为其与离线状态价值的加权组合。")
add("06", "### 6.3.1 支持集固定的稀疏残差", r"$\widehat u_{\mathrm S}$表示源模型预测的状态增量，区别于无下标的物理停站$u$。$\psi$与$\zeta$分别是状态神经余项和奖励修正网络的参数，$\Theta$表示候选特征库，$\widetilde\Theta$为其尺度化形式，$\mathcal L_{\mathrm{sel}}$为选择集上的拟合损失。")
add("06", "### 6.3.3 对数置信评价与一步模型样本", r"$\xi$表示对数置信评价器参数，$\bar\xi$表示其目标网络参数；$\gamma_\Delta$为置信传播折扣，$d_i^{\mathrm{term}}$为终止标记。$\chi_i$是确定性有效性条件的联合指示量，$w_i^\Delta$是截断前的置信权重，最终权重$w_i$还包含有效性筛选。")

change("04", "CS-BAPR的历史全称为Causal-Symbolic Bayesian Adaptive Policy Regularization，即因果符号贝叶斯自适应策略正则化，本章公交实现的组件范围见第4.4.1节。", "本章公交实现的组件范围见第4.4.1节。")
change("04", "隐参数MDP以", "HiP-MDP以")
change("04", "这一机制侧重动作坐标与环境执行之间的关系，附录B中，", "这一机制侧重动作坐标与环境执行之间的关系。附录B中，")
change("06", "DAgger从", "数据集聚合（Dataset Aggregation，DAgger）从")
change("05", "RORL则", "鲁棒离线强化学习（Robust Offline Reinforcement Learning，RORL）则")
change("01", "SimOpt利用", "仿真优化方法（Simulation Optimization，SimOpt）利用")
change("01", "QMIX把", "单调价值混合方法QMIX把") if "QMIX把" in docs["01"] else None
change("06", "总变差定义", "总变差（Total Variation，TV）定义")
change("B", "### B.4.4 有限正概率核的KL链式分解", "### B.4.4 有限正概率核的相对熵链式分解")
change("B", "定义有限分布的相对熵", "定义有限分布的Kullback-Leibler散度（Kullback-Leibler Divergence，KL散度），亦称相对熵")

# A norm penalty is not an estimate of irreducible transition variance.
add("A", "## A.1 定义与假设", r"本附录沿用第3.3节的$\alpha_{\mathrm{ent}}$、$\lambda_{\mathrm{epi}}$与$\lambda_{\mathrm{ale}}$。$R_{\mathrm{pen}}$为包含固定惩罚后的即时项，$c_{\mathrm{shift}}$为常数平移，$\mathcal V$为一般动作聚合映射；它们均不是新增的随机环境量。")
add("C", "## C.1 模式条件贝尔曼算子", r"$\mathcal Z$为候选模式集合，$z$为其元素；$\rho$为单模式信念，$\rho^{\mathrm{joint}}$为联合模式信念，$\rho^{(2)}$为第二模式变量的边缘分布。$\overline R_{\mathrm{pen},\rho}$表示按信念加权的固定奖励与惩罚项。")
add("E", "## E.6 置信信号的条件排序等价", r"本节的$w_{\mathrm{rank}}$是用于静态排序的正值信号，$\tau_{\mathrm{rank}}>0$为排序尺度，$c_{\mathrm{rank}}$为加性常数；$\epsilon_{\mathrm{rank}}$表示拟合误差在比较域内的波动界。其定义与正文含未来备份的动态置信权重分别给出。")

for path in paths:
    path.write_text(docs[path.name.split("_")[0]])
