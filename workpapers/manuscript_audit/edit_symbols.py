"""Mechanical notation harmonization after semantic review of each section."""

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2] / "manuscript"
paths = sorted((ROOT / "chapters").glob("*.md")) + sorted(p for p in (ROOT / "appendices").glob("*.md") if p.name[0] in "ABCDE")
docs = {p.name.split("_")[0]: p.read_text() for p in paths}
MATH = re.compile(r"\$\$.*?\$\$|(?<!\$)\$(?!\$).*?(?<!\$)\$(?!\$)", re.S)
ATOM = re.compile(r"\\(?:begin|end|mathrm|text|operatorname|mathsf|mathcal|mathbf|mathbb|boldsymbol)\{[^{}]*\}|\\mathrm\s+[A-Za-z]+|\\[A-Za-z]+|[A-Za-z]")


def rewrite(key, mapping=None, tokens=None, start=None, end=None):
    text = docs[key]
    a = text.index(start) if start else 0
    b = text.index(end, a + 1) if end else len(text)
    def transform(match):
        value = match[0]
        if mapping:
            pattern = re.compile("|".join(re.escape(k) for k in sorted(mapping, key=len, reverse=True)))
            value = pattern.sub(lambda m: mapping[m[0]], value)
        if tokens:
            value = ATOM.sub(lambda m: "{" + tokens[m[0]] + "}" if m[0] in tokens else m[0], value)
        return value
    docs[key] = text[:a] + MATH.sub(transform, text[a:b]) + text[b:]


def add(key, heading, sentence):
    assert heading in docs[key], heading
    docs[key] = docs[key].replace(heading, heading+"\n\n"+sentence, 1)


def replace(key, old, new):
    assert old in docs[key], (key, old)
    docs[key] = docs[key].replace(old, new)


# Headway direction and target superscripts are identical across bus chapters.
headways = {
    r"H_k^{\mathrm b,*}": r"h_k^{\mathrm b,*}",
    r"h_{\mathrm{front}}": r"h^{\mathrm f}", r"h_{\mathrm{back}}": r"h^{\mathrm b}",
    r"h_{\mathrm{front}}^*": r"h^{\mathrm f,*}", r"h_{\mathrm{back}}^*": r"h^{\mathrm b,*}",
    r"h_f^*": r"h^{\mathrm f,*}", r"h_b^*": r"h^{\mathrm b,*}",
    r"H_f^*": r"h^{\mathrm f,*}", r"H_b^*": r"h^{\mathrm b,*}",
    r"h_f": r"h^{\mathrm f}", r"h_b": r"h^{\mathrm b}",
    r"H_f": r"h^{\mathrm f}", r"H_b": r"h^{\mathrm b}",
    r"H_{\max}": r"u_{\max}",
}
for key in ("02", "03", "04", "05"):
    rewrite(key, headways)
for key in ("03", "04", "05"):
    def entropy(m):
        return re.sub(r"\\alpha(?![A-Za-z_])", lambda _: r"{\alpha_{\mathrm{ent}}}", m[0])
    docs[key] = MATH.sub(entropy, docs[key])

rewrite("02", {r"b_{j,i}": r"d_{j,i}^{\mathrm{svc}}", r"b_{j-1,i}": r"d_{j-1,i}^{\mathrm{svc}}",
               r"r_{j,i}": r"d_{j,i}^{\mathrm{run}}", r"r_{j-1,i}": r"d_{j-1,i}^{\mathrm{run}}",
               r"e_{j,i}": r"d_{j,i}^{\mathrm{exit}}", r"e_{j-1,i}": r"d_{j-1,i}^{\mathrm{exit}}",
               r"H_{j,i+1}": r"h_{j,i+1}^{\mathrm f}", r"H_{j,i}": r"h_{j,i}^{\mathrm f}",
               r"\eta_i": r"\eta_i^{\mathrm{board}}", r"d_j^{\mathrm{plan}}": r"t_j^{\mathrm{plan}}",
               r"d_j^{\mathrm{act}}": r"t_j^{\mathrm{act}}"})
rewrite("02", tokens={"K": r"K_{\mathrm{evt}}"}, start="### 2.3.1", end="### 2.3.2")
rewrite("02", tokens={"z": r"{z^{\mathrm{emb}}}"}, start="### 2.3.3", end="### 2.3.4")
rewrite("02", tokens={r"\tau": "u", "a": r"{a^{\mathrm n}}"}, start="### 2.3.4", end="### 2.3.5")
rewrite("02", tokens={"V": r"T^{\mathrm{ride}}"}, start="### 2.4.2", end="### 2.4.3")
rewrite("02", {r"z_i": r"d_i^{\mathrm{term}}"}, start="### 2.5.2", end="### 2.5.3")
rewrite("02", {r"d_b": r"\Delta G_b"}, start="### 2.6.2", end="### 2.6.3")
add("02", "### 2.3.1 异步控制与半马尔可夫表述", r"下文以$s$、$a$和$r$分别表示决策状态、控制动作和一次转移实际获得的奖励；$\pi_\theta(a\mid s)$为参数$\theta$下的条件动作分布，$P$为环境转移核，$R(s,a)$为期望即时奖励。$\mathbb E$表示相应概率分布下的期望。时间$t$以秒计，$k$计数决策事件；两者不互换。")
add("02", "### 2.3.4 动作映射与实际执行", "以$u$统一表示以秒计的附加停站，$a^{\mathrm n}$表示归一化策略输出。一般决策模型中的$a$表示动作变量，其具体编码由本节映射确定。")

# Category position is not the trip index; an embedding is not a latent mode.
rewrite("03", tokens={"j": "i", "q": r"{q_{\mathrm{per}}}", "d": r"{d_{\mathrm{dir}}}"}, start="### 3.1.1", end="### 3.1.2")
rewrite("03", {r"e_j(j)": r"e_i(i)", r"e_q(q)": r"e_q(q_{\mathrm{per}})", r"e_d(d)": r"e_d(d_{\mathrm{dir}})"}, start="### 3.1.2", end="### 3.1.3")
rewrite("03", tokens={"z": r"{z^{\mathrm{emb}}}", "u": r"{j_{\mathrm{cat}}}"}, start="### 3.1.2", end="### 3.1.3")
rewrite("03", tokens={"a": "u"}, start="### 3.1.3", end="### 3.1.4")
rewrite("03", tokens={"a": "u"}, start="### 3.2.2", end="### 3.2.3")
rewrite("03", tokens={"a": "u", "w": r"{w_{\mathrm{wait}}}", "n": r"{n_{\mathrm{on}}}", "d": r"{\Delta h}"}, start="### 3.2.4", end="## 3.3")
rewrite("03", tokens={"q": r"Q_s", "p": r"\pi_s"}, start="### 3.2.3", end="### 3.2.4")
rewrite("03", tokens={"c": r"c_{\mathrm{scale}}"}, start="这里的“固定”", end="### 3.3.4")
rewrite("03", tokens={r"\lambda": r"\lambda_{\mathrm{cons}}"}, start="### 3.3.6", end="## 3.4")
rewrite("03", tokens={"w": r"{w_{\mathrm{wait}}}"}, start="### 3.4.4", end="### 3.4.5")
rewrite("03", tokens={"c": r"c_{\mathrm{shift}}"}, start="### 3.6.1", end="### 3.6.2")
rewrite("03", tokens={"a": "u", "w": r"w_{\mathrm{hold}}"}, start="### 3.7.3", end="### 3.7.4")
add("03", "### 3.2.3 熵正则化策略学习", r"记$\gamma\in[0,1)$为回报折扣因子，$\alpha_{\mathrm{ent}}>0$为熵温度，$\mathcal H(\pi)$为策略熵。局部变分推导中，$Q_s(a)$为固定状态下给定的动作价值曲线，$\pi_s$为待优化的动作密度；积分中的动作变量只在该积分内取值。")

# The same terminal flag and entropy temperature are used in all learners.
rewrite("04", tokens={"d": r"d^{\mathrm{term}}"}, start="### 4.3.1", end="### 4.3.2")
rewrite("04", tokens={"q": r"{q_{\mathrm{load}}}"}, start="（6）", end="### 4.3.2") if "（6）" in docs["04"] else None
rewrite("04", tokens={"J": r"J_{\mathrm{jac}}"}, start="### 4.3.2", end="### 4.3.3")
rewrite("04", tokens={"d": r"\Delta x"}, start="### 4.3.3", end="## 4.4")
rewrite("04", tokens={"b": r"\eta^{\mathrm{board}}", "d": r"{d^{\mathrm{svc}}}", "m": r"m_{\mathrm{dem}}"}, start="### 4.4.5", end="## 4.5")
rewrite("04", tokens={r"\tau": "t"}, start="### 4.5.2", end="### 4.5.3")
rewrite("04", tokens={r"\lambda": r"\lambda_{\mathrm{con}}", "c": r"{c_{\mathrm{con}}}"}, start="### 4.5.3", end="### 4.5.4")
rewrite("04", tokens={"C": r"\mathrm{CV}_h", "N": r"N_{\mathrm{fleet}}", "T": r"T^{\mathrm{trip}}", "B": r"T^{\mathrm{ride}}", "q": r"{\Delta t^{\mathrm{plan}}}", r"\tau": "t"}, start="### 4.5.4", end="### 4.5.5")
rewrite("04", {r"\overline a_i": r"\overline u_i"}, start="### 4.5.4", end="### 4.5.5")
rewrite("04", tokens={"a": "u", "H": r"h^*"}, start="### 4.6.1", end="### 4.6.2")
rewrite("04", tokens={"H": "h", "p": r"{t^{\mathrm{plan}}}", "u": r"{t^{\mathrm{act}}}"}, start="### 4.6.5", end="### 4.6.6")
rewrite("04", tokens={"C": r"\mathrm{CV}_h", "N": r"N_{\mathrm{fleet}}", "R": r"F_{\mathrm{excess}}"}, start="### 4.6.7", end="### 4.6.8")
rewrite("04", tokens={"V": r"V_{\mathrm{hours}}", "K": r"L_{\mathrm{fleet}}", r"\ell": r"l^{\mathrm{route}}"}, start="### 4.6.8", end="## 4.7")
rewrite("04", tokens={"Q": r"N_{\mathrm w}"}, start="### 4.7.2", end="### 4.7.3")
rewrite("04", {"G_r": r"G_{\mathrm{ref}}", r"\pi_r": r"\pi_{\mathrm{ref}}"}, start="### 4.7.3", end="### 4.7.4")
add("04", "### 4.3.2 结构化策略输出", r"$g_\theta$表示状态特征网络，$f_{\mathrm{NAU}}$和$f_{\mathrm{NMU}}$为两类算术输出，$\omega\in[0,1]$为混合权重。导数分析中以$h$表示输出映射、$D$表示Fréchet导数，$J_{\mathrm{jac}}$为复合映射的雅可比；这些映射记号与带方向标记的车头时距$h^{\mathrm f},h^{\mathrm b}$区分。")

rewrite("05", {r"\mu_Q": r"\overline Q", r"h_{\mathrm D}": r"u_{\mathrm D}", r"h_{\mathrm A}": r"u_{\mathrm A}"})
rewrite("05", tokens={r"\phi": r"\theta", "a": r"{a^{\mathrm n}}"}, start="### 5.4.3", end="### 5.4.4")
rewrite("05", {r"q_{\min}": r"Q_{\min}"})
rewrite("05", {r"h_\phi": r"u_\theta", r"h_{\mathrm{hb}}": r"u_{\mathrm{hb}}", r"h_{16}": r"u_{\mathrm{scaled}}", r"h_{\mathrm{par}}": r"u_{\mathrm{par}}", r"h_{\mathrm{safe}}": r"u_{\mathrm{safe}}"})
rewrite("05", tokens={"q": r"\xi_{\mathrm{time}}", "c": r"C_{\mathrm{veh}}"}, start="## 5.5", end="## 5.6")
rewrite("05", tokens={"u": r"\xi_{\min}", "v": r"\xi_{\max}"}, start="令 $n_{\mathrm{on}}$", end="后续完成保障" if "后续完成保障" in docs["05"][docs["05"].index("令 $n_{\mathrm{on}}$"):] else "## 5.6")
rewrite("05", {r"\Delta R_b": r"\Delta R_{\mathrm{cmp},b}", r"\bar R_b": r"\bar R_{\mathrm{cmp},b}"})
rewrite("05", tokens={"W": r"N_{\mathrm{win}}"}, start="### 5.6.3", end="### 5.6.4")
rewrite("05", tokens={"V": r"T^{\mathrm{ride}}", "w": r"{w_{\mathrm{wait}}}"}, start="### 5.6.7", end="### 5.6.8")
add("05", "### 5.4.3 零停站选择与正停站时长的混合策略", r"本节的$a^{\mathrm n}$沿用第2.3.4节的归一化动作定义，$\theta$为策略参数。$\delta_{-1}$是在$-1$处的Dirac概率质量，$f_\theta$是正停站分支的动作密度，$\varphi(\cdot;\mu,\sigma^2)$表示均值为$\mu$、方差为$\sigma^2$的高斯密度。")

# MC-WM reward, support mask, terminal flag and robust loss had reused belief symbols.
rewrite("06", {r"\rho_\ell": r"R_\ell", r"\widehat\rho_{\mathrm S}": r"\widehat R_{\mathrm S}",
               r"\pi_\phi": r"\pi_\theta", r"\kappa_t": r"d_t^{\mathrm{term}}", r"\kappa_i": r"d_i^{\mathrm{term}}",
               r"Q_{\Delta}": r"Q^\Delta", r"Q_{\eta_j}": r"Q_{\phi_j}"})
rewrite("06", tokens={"z": r"{y^{\mathrm{mdl}}}"}, start="### 6.2.3", end="## 6.3")
rewrite("06", tokens={r"\mu": r"m_{\mathrm{sup}}", "B": r"B_{\mathrm{coef}}"}, start="### 6.3.1", end="### 6.3.2")
rewrite("06", tokens={"b": r"B_{\mathrm{batch}}"}, start="### 6.3.4", end="## 6.4")
rewrite("06", tokens={"D": r"\epsilon_V", "W": r"V^{(2)}"}, start="### 6.4.1", end="### 6.4.3")
rewrite("06", tokens={r"\tau": r"\tau_e", r"\rho": r"\ell_{\mathrm{Hub}}"}, start="### 6.7.1", end="### 6.7.2")
add("06", "### 6.2.1 目标环境中的策略优化", r"$\mathcal S$、$\mathcal A$分别为状态与动作空间，$p_0$为初始状态分布；下标$\mathrm S$和$\mathrm T$分别表示源与目标环境。$R_\ell$为环境$\ell$的期望单步奖励。")
add("06", "### 6.3.4 违例代价与策略更新", r"下式中$B_{\mathrm{batch}}=|\mathcal I|$为批量大小，$\mathcal I$为样本索引集合，$\phi_j$为第$j$个动作价值网络参数，$y_i^Q$为对应的贝尔曼回归目标；$\operatorname{Std}_j$表示跨评价器的标准差。")

# Align appendix notation with the chapters and distinguish separate mathematical objects.
rewrite("A", tokens={r"\tau": r"{\alpha_{\mathrm{ent}}}", "b": r"R_{\mathrm{pen}}", "c": r"c_{\mathrm{shift}}", r"\lambda": r"\lambda_{\mathrm{fb}}"})
rewrite("B", {r"G_r": r"G_{\mathrm{ref}}", r"g_{r,": r"g_{\mathrm{ref},", r"\pi_r": r"\pi_{\mathrm{ref}}"})
rewrite("B", tokens={"q": r"{a^{\mathrm{ref}}}"}, start="### B.1.2", end="### B.4.4")
rewrite("B", tokens={"d": r"n_a"})
rewrite("B", tokens={"q": r"{a^{\mathrm{ref}}}"}, start="## B.5")
# C as a superscript labels the compensated controller; C( . ) is clipping.
docs["B"] = MATH.sub(lambda m: re.sub(r"(?<![\\_^])C(?=[($])", lambda _: r"\operatorname{clip}", m[0]), docs["B"])
rewrite("B", tokens={"c": r"\operatorname{clip}_1"}, start="## B.1", end="## B.3")
rewrite("B", tokens={"H": r"N_{\mathrm{hor}}"}, start="### B.4.4", end="## B.5")
replace("B", "参考模式为 $r$", r"参考模式记为$\mathrm{ref}$")
rewrite("C", tokens={"b": r"\rho^{\mathrm{joint}}", r"\mu": r"\rho^{(2)}", "h": "z", "z": r"z^{(2)}"})
rewrite("C", {r"\overline \rho^{\mathrm{joint}}_\rho": r"\overline R_{\mathrm{pen},\rho}"})
rewrite("C", tokens={"c": r"c_{\mathrm{shift}}"}, start="## C.3", end="## C.4")
rewrite("C", tokens={r"\lambda": r"\lambda_{\mathrm{fb}}", "q": r"Q_{\mathrm{sc}}", r"\Delta": r"\Delta R"}, start="## C.7", end="## C.8")
rewrite("C", tokens={"r": r"{O_{\mathrm{belief}}}", "L": r"L_{\mathrm{evid}}"}, start="## C.8")
rewrite("D", tokens={"h": "z"}, start="## D.6", end="## D.8")
rewrite("D", {r"\overline b_{\mathrm{CS}}": r"\overline R_{\mathrm{pen},\mathrm{CS}}"}, start="## D.6", end="## D.8")
rewrite("E", tokens={"W": r"V^{(2)}"}, start="## E.2", end="## E.3")
rewrite("E", tokens={"e": r"\epsilon_V"}, start="### E.3.2", end="### E.3.4")
rewrite("E", tokens={r"\tau": r"\tau_{\mathrm{rank}}", "q": r"w_{\mathrm{rank}}", "c": r"c_{\mathrm{rank}}", "C": r"e_{\mathrm{const}}", r"\delta": r"\epsilon_{\mathrm{rank}}"}, start="## E.6")
rewrite("E", {r"\Delta r": r"\Delta e_{\mathrm{resid}}"}, start="## E.6")

for path in paths:
    path.write_text(docs[path.name.split("_")[0]])
