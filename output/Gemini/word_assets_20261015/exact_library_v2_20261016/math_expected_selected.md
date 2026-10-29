EXPECT_SLOT_0143

$$
\| Q \|_{\infty} = \max_{s,a}|Q(s,a)|.
$$

EXPECT_SLOT_0144

$$
\kappa_{\mathrm{ale}} = \lambda_{\mathrm{ale}}\sum_{l} \| W_{l} \|_{1}.
$$

EXPECT_SLOT_0145

$$
\begin{aligned}
V_{Q}(s) &= \max_{a}Q(s,a), \\
V_{Q}^{\alpha_{\mathrm{ent}}}(s) &= \alpha_{\mathrm{ent}}\log\sum_{a}\exp\left( \frac{Q(s,a)}{\alpha_{\mathrm{ent}}} \right).
\end{aligned}
$$

EXPECT_SLOT_0146

$$
R_{\mathrm{pen}}(s,a) = R(s,a) - \gamma c_{\mathrm{pen}}(s,a).
$$

EXPECT_SLOT_0147

$$
\begin{aligned}
(\mathcal{T}Q)(s,a) &= R_{\mathrm{pen}}(s,a) + \gamma\sum_{s'}P(s' \mid s,a)V_{Q}(s'), \\
(\mathcal{T}_{\alpha_{\mathrm{ent}}}Q)(s,a) &= R_{\mathrm{pen}}(s,a) + \gamma\sum_{s'}P(s' \mid s,a)V_{Q}^{\alpha_{\mathrm{ent}}}(s').
\end{aligned}
$$

EXPECT_SLOT_0148

$$
Q_{1} \leq Q_{2}\  \Rightarrow \ \mathcal{V}Q_{1} \leq \mathcal{V}Q_{2},\qquad \mathcal{V}(Q + c_{\mathrm{shift}}) = \mathcal{V}Q + c_{\mathrm{shift}},
$$

EXPECT_SLOT_0149

$$
\| \mathcal{V}Q_{1} - \mathcal{V}Q_{2} \|_{\infty} \leq \| Q_{1} - Q_{2} \|_{\infty}.
$$

EXPECT_SLOT_0150

$$
\mathcal{V}Q_{2} - \varepsilon_{\mathrm{dist}} \leq \mathcal{V}Q_{1} \leq \mathcal{V}Q_{2} + \varepsilon_{\mathrm{dist}}.
$$

EXPECT_SLOT_0151

$$
(\mathcal{B}Q)(s,a) = R_{\mathrm{pen}}(s,a) + \gamma\sum_{s'}P(s' \mid s,a)(\mathcal{V}Q)(s').
$$

EXPECT_SLOT_0152

$$
\begin{aligned}
Q_{1} \leq Q_{2} &\  \Rightarrow \ \mathcal{B}Q_{1} \leq \mathcal{B}Q_{2}, \\
\mathcal{B}(Q + c_{\mathrm{shift}}) &= \mathcal{B}Q + \gamma c_{\mathrm{shift}}, \\
\| \mathcal{B}Q_{1} - \mathcal{B}Q_{2} \|_{\infty} &\leq \gamma \| Q_{1} - Q_{2} \|_{\infty}.
\end{aligned}
$$

EXPECT_SLOT_0153

$$
\mathcal{B}(Q + c_{\mathrm{shift}}) - \mathcal{B}Q = \gamma c_{\mathrm{shift}}\sum_{s'}P(s' \mid s,a) = \gamma c_{\mathrm{shift}}.
$$

EXPECT_SLOT_0154

$$
\begin{aligned}
|(\mathcal{B}Q_{1} - \mathcal{B}Q_{2})(s,a)| &\leq \gamma\sum_{s'}P(s' \mid s,a)|(\mathcal{V}Q_{1} - \mathcal{V}Q_{2})(s')| \\
 &\leq \gamma \| Q_{1} - Q_{2} \|_{\infty}.
\end{aligned}
$$

EXPECT_SLOT_0155

$$
\begin{aligned}
V_{Q + c_{\mathrm{shift}}}^{\alpha_{\mathrm{ent}}}(s) &= \alpha_{\mathrm{ent}}\log\left\lbrack \exp(c_{\mathrm{shift}}/\alpha_{\mathrm{ent}})\sum_{a}\exp(Q(s,a)/\alpha_{\mathrm{ent}}) \right\rbrack \\
 &= c_{\mathrm{shift}} + V_{Q}^{\alpha_{\mathrm{ent}}}(s).
\end{aligned}
$$

EXPECT_SLOT_0156

$$
\mathcal{T}_{\alpha_{\mathrm{ent}}}(Q + c_{\mathrm{shift}}) = \mathcal{T}_{\alpha_{\mathrm{ent}}}Q + \gamma c_{\mathrm{shift}},\qquad \| \mathcal{T}_{\alpha_{\mathrm{ent}}}Q_{1} - \mathcal{T}_{\alpha_{\mathrm{ent}}}Q_{2} \|_{\infty} \leq \gamma \| Q_{1} - Q_{2} \|_{\infty}.
$$

EXPECT_SLOT_0162

$$
\begin{aligned}
\langle p^{\circ},V\rangle - \Omega_{\mathcal{U}}(V) &= - \sigma_{\mathcal{U}}( - V) = \inf_{p \in \mathcal{U}}\langle p,V\rangle, \\
\Omega_{\mathcal{U}}(V) &= \langle p^{\circ},V\rangle - \inf_{p \in \mathcal{U}}\langle p,V\rangle \geq 0.
\end{aligned}
$$

EXPECT_SLOT_0163

$$
\inf_{p \in \mathcal{U}}\mathbb{E}_{p}[ V] \geq \mathbb{E}_{p^{\circ}}[ V] - \varepsilon_{\mathrm{ipm}}L_{V}.
$$

EXPECT_SLOT_0165

$$
\mathrm{CVaR}_{\beta_{\mathrm{tail}}}(C_{\mathrm{traj}}) = \inf_{\eta_{\mathrm{risk}} \in \mathbb{R}}\left\{ \eta_{\mathrm{risk}} + \frac{1}{\beta_{\mathrm{tail}}}\mathbb{E}_{\mathbb{P}_{0}}[(C_{\mathrm{traj}} - \eta_{\mathrm{risk}})_{+}] \right\}.
$$

EXPECT_SLOT_0169

$$
\mathcal{A} = [-1,1]^{n_{a}}.
$$

EXPECT_SLOT_0170

$$
\mathrm{clip}(a)_{j} = \mathrm{clip}_{1}(a_{j}),\quad \text{其中}\quad \mathrm{clip}_{1}(x) = \max\{ - 1, \min\{ x,1\}\}.
$$

EXPECT_SLOT_0171

$$
u_{\mathrm{exec}} = \mathrm{clip}(v_{\mathrm{nom}} + \varepsilon).
$$

EXPECT_SLOT_0172

$$
a^{\mathrm{ref}}(x_{\mathrm{phys}}) = \mu_{\mathrm{ref}}\big(o(x_{\mathrm{phys}})\big) \in \mathcal{A}.
$$

EXPECT_SLOT_0173

$$
G_{z}a_{z}(x_{\mathrm{phys}}) = G_{\mathrm{ref}}a^{\mathrm{ref}}(x_{\mathrm{phys}}).
$$

EXPECT_SLOT_0174

$$
\mathrm{clip}_{1}(x) = x\quad \Leftrightarrow \quad - 1 \leq x \leq 1.
$$

EXPECT_SLOT_0175

$$
\mathrm{clip}(a) = a\quad \Leftrightarrow \quad a \in \mathcal{A}.
$$

EXPECT_SLOT_0176

$$
a_{\mathrm{inv}} = G_{\mathrm{act}}^{- 1}v_{\mathrm{nom}}^{*},\qquad a_{\mathrm{inv},j} = v_{\mathrm{nom},j}^{*}/g_{j}.
$$

EXPECT_SLOT_0177

$$
G_{\mathrm{act}}\,\mathrm{clip}(a_{\mathrm{inv}}) = v_{\mathrm{nom}}^{*}\quad \Leftrightarrow \quad a_{\mathrm{inv}} \in \mathcal{A}.
$$

EXPECT_SLOT_0192

$$
D_{\mathrm{KL}}(p \parallel q) = \sum_{h_{\mathrm{obs}}}p(h_{\mathrm{obs}})\log\frac{p(h_{\mathrm{obs}})}{q(h_{\mathrm{obs}})}.
$$

EXPECT_SLOT_0211

$$
V_{0,k}^{\pi}(x_{aug}) = 0,
$$

EXPECT_SLOT_0212

$$
V_{n + 1,k}^{\pi}(x_{aug}) = \sum_{e}^{}p(e)\left\lbrack r_{e}(x_{aug},{v_{nom}}^{\pi}(k,x_{aug})) + V_{n,k + 1}^{\pi}(F_{dyn,e}(x_{aug},{v_{nom}}^{\pi}(k,x_{aug}))) \right\rbrack.
$$

EXPECT_SLOT_0219

$$
B_{ret,0,k}(x_{aug}) = 0,
$$

EXPECT_SLOT_0220

$$
B_{ret,n + 1,k}(x_{aug}) = b_{Q,loc}(k,x_{aug}) + \sum_{e}^{}p(e)B_{ret,n,k + 1}(F_{dyn,e}(x_{aug},{v_{nom}}^{C}(k,x_{aug}))).
$$

EXPECT_SLOT_0231

$$
(\mathcal{T}_{z}Q)(s,a) = R_{z}(s,a) + \gamma\left\lbrack \sum_{s'}P_{z}(s' \mid s,a)V_{Q}(s') - \lambda_{\mathrm{epi}}\Gamma_{z}(s,a) - \kappa_{\mathrm{fix}} \right\rbrack.
$$

EXPECT_SLOT_0232

$$
\rho(z) \geq 0,\qquad \sum_{z \in \mathcal{Z}}\rho(z) = 1.
$$

EXPECT_SLOT_0233

$$
(\mathcal{T}_{\rho}Q)(s,a) = \sum_{z \in \mathcal{Z}}\rho(z)(\mathcal{T}_{z}Q)(s,a).
$$

EXPECT_SLOT_0237

$$
\begin{aligned}
\mathcal{T}_{\rho}\left( Q + c_{shift} \right) = \mathcal{T}_{\rho}Q + \gamma c_{shift},\quad\quad \\
\parallel \mathcal{T}_{\rho}Q_{1} - \mathcal{T}_{\rho}Q_{2} \parallel_{\infty} \leq \gamma \parallel Q_{1} - Q_{2} \parallel_{\infty}.
\end{aligned}
$$

EXPECT_SLOT_0256

$$
L_{\mathrm{evid}}^{2n} \geq \frac{\Omega_{\mathrm{prior}}}{\delta_{\mathrm{post}}},
$$

EXPECT_SLOT_0257

$$
\frac{1}{\Omega_{\mathrm{prior}}}L_{\mathrm{evid}}^{2n} \geq \frac{1}{\Omega_{\mathrm{prior}}}\frac{\Omega_{\mathrm{prior}}}{\delta_{\mathrm{post}}} = \frac{1}{\delta_{\mathrm{post}}}.
$$

EXPECT_SLOT_0258

$$
\rho_{n}(\mathrm{new}) = \frac{O_{\mathrm{belief},n}}{1 + O_{\mathrm{belief},n}} \geq \frac{1}{1 + \delta_{\mathrm{post}}} \geq 1 - \delta_{\mathrm{post}},
$$

EXPECT_SLOT_0259

$$
n \geq \max\left\{ 0,\ \left\lceil \frac{\log(\Omega_{\mathrm{prior}}/\delta_{\mathrm{post}})}{2\log L_{\mathrm{evid}}} \right\rceil \right\}.
$$

EXPECT_SLOT_0260

$$
|{\mu_{\mathrm{det}}}'(x) - {f_{\mathrm{ref}}}'(x)| \leq \varepsilon_{\mathrm{der}},\qquad x \in X_{\mathrm{eval}}.
$$

EXPECT_SLOT_0261

$$
|{\mu_{\mathrm{det}}}'(x) - {\mu_{\mathrm{det}}}'(y)| \leq L_{D\mu}|x - y|.
$$

EXPECT_SLOT_0262

$$
\| D\mu_{\mathrm{det}}(x) - Df_{\mathrm{ref}}(x) \| \leq \varepsilon_{\mathrm{der}}
$$

EXPECT_SLOT_0263

$$
\| D\mu_{\mathrm{det}}(x) - D\mu_{\mathrm{det}}(y) \| \leq L_{D\mu} \| x - y \| .
$$

EXPECT_SLOT_0264

$$
\begin{aligned}
q_{\theta}(x) &= \beta_{0} + f_{\theta}(x) - f_{\theta}\big(x_{\mathrm{ref}}(x)\big), \\
\mu_{\mathrm{det},\theta}(x) &= \tanh\big(q_{\theta}(x)\big), \\
a_{\mathrm{hold}}(x) &= 30\,\big[ 1 + \mu_{\mathrm{det},\theta}(x)\big].
\end{aligned}
$$

EXPECT_SLOT_0265

$$
f_{\mathrm{ref}}(x) = \mathrm{clip}_{[-0.999,0.999]}\left\lbrack \frac{\sum_{\tilde{a}_k \in \mathcal{A}_{g}} \tilde{a}_k \exp\big( - \hat{c}(x,\tilde{a}_k)/\tau_{\mathrm{boltz}}\big)}{\sum_{\tilde{a}_{k'} \in \mathcal{A}_{g}}\exp\big( - \hat{c}(x,\tilde{a}_{k'})/\tau_{\mathrm{boltz}}\big)} \right\rbrack.
$$

EXPECT_SLOT_0266

$$
\begin{aligned}
\ell_{\mathrm{sym}}(x) &= \frac{\| J_{\mu}(x) - J_{f}(x) \|_{F}^{2}}{1 + \| J_{f}(x) \|_{F}^{2}}, \\
\mathcal{L}_{\pi} &= \mathcal{L}_{\pi,\mathrm{base}} + \frac{0.01}{B}\sum_{i = 1}^{B}\ell_{\mathrm{sym}}(x_{i}),
\end{aligned}
$$

EXPECT_SLOT_0267

$$
y_{i} = \frac{r_{i}}{\sigma_{r}} + (1 - d_{i})\gamma\Big\lbrack \min_{1 \leq k \leq 3}\hat{Q}_{\phi'_{k}}(s_{i}',a_{i}') - \alpha\log\pi_{\theta}(a_{i}' \mid s_{i}') - \lambda_{\mathrm{sym}}\,\mathrm{sg}\big(\bar{\mathcal{L}}_{\mathrm{JC}}'\big)\Big\rbrack.
$$

EXPECT_SLOT_0268

$$
|{\mu_{\mathrm{det}}}'(x) - {f_{\mathrm{ref}}}'(x)| \leq \varepsilon_{\mathrm{der}},\qquad \varepsilon_{\mathrm{der}} \geq 0,
$$

EXPECT_SLOT_0280

$$
\| D\mu_{\mathrm{det}}(z) - Df_{\mathrm{ref}}(z) \| \leq \varepsilon_{\mathrm{der}}.
$$

EXPECT_SLOT_0281

$$
\| \mu_{\mathrm{det}}(x) - f_{\mathrm{ref}}(x) \| \leq \varepsilon_{\mathrm{val}} + \varepsilon_{\mathrm{der}} \| x - x_{0} \| .
$$

EXPECT_SLOT_0282

$$
\| e_{\mathrm{map}}(x) - e_{\mathrm{map}}(x_{0}) \| \leq \sup_{0 \leq t \leq 1} \| De_{\mathrm{map}}(z(t)) \| \, \| \Delta x \| \leq \varepsilon_{\mathrm{der}} \| \Delta x \| .
$$

EXPECT_SLOT_0283

$$
\| e_{\mathrm{path}}(0) \| \leq \varepsilon_{\mathrm{val}},\qquad \| e_{\mathrm{path}, +}'(t) \| \leq \varepsilon_{\mathrm{path}} + L_{\mathrm{path}}t,
$$

EXPECT_SLOT_0284

$$
\| e_{\mathrm{path}}(1) \| \leq \varepsilon_{\mathrm{val}} + \varepsilon_{\mathrm{path}} + \frac{L_{\mathrm{path}}}{2}.
$$

EXPECT_SLOT_0285

$$
(\ell_{\mathrm{fun}} \circ e_{\mathrm{path}})_{+}'(t) = \ell_{\mathrm{fun}}\big(e_{\mathrm{path}, +}'(t)\big) \leq {B_{\mathrm{err}}}'(t).
$$

EXPECT_SLOT_0299

$$
|d_{\mathrm{left}} - d_{\mathrm{right}}| = |{f_{\mathrm{ref}}}'( - t) - {f_{\mathrm{ref}}}'(t)| \leq 2Lt < |d_{\mathrm{left}} - d_{\mathrm{right}}|,
$$

EXPECT_SLOT_0306

$$
V_{Q}(s) = \max_{a}Q(s,a),
$$

EXPECT_SLOT_0307

$$
\begin{aligned}
(\mathcal{T}_{z}Q)(s,a) = R_{z}(s,a) + \gamma\lbrack & \sum_{s'}^{}P_{z}(s' \mid s,a)V_{Q}(s') \\
 & - \lambda_{epi}\Gamma_{epi,z}(s,a) \\
 & - \lambda_{sym}\Gamma_{sym,z}(s,a) - \kappa_{fix}\rbrack.
\end{aligned}
$$

EXPECT_SLOT_0319

$$
\| Df_{e}(x) - Df_{c_{\mathrm{coef}}}(x) \| \leq \varepsilon_{\mathrm{IRM}},\qquad e \in \mathcal{E},\ x \in D_{e},
$$

EXPECT_SLOT_0320

$$
\| D\mu_{\mathrm{det}}(x) - Df_{c_{\mathrm{coef}}}(x) \| \leq \varepsilon_{\mathrm{pol}},\qquad e \in \mathcal{E},\ x \in D_{e},
$$

EXPECT_SLOT_0328

$$
P(s' \mid s,a) \geq 0,\qquad \sum_{s' \in \mathcal{S}}P(s' \mid s,a) = 1,
$$

EXPECT_SLOT_0329

$$
|R(s,a)| \leq R_{\max},\quad\quad R_{\max} \geq 0,\quad\quad 0 \leq \gamma < 1.
$$

EXPECT_SLOT_0362

$$
\ell_{w_{\mathrm{rank}}}(x) := \log w_{\mathrm{rank}}(x) = - \frac{e_{\mathrm{real}}(x)}{\tau_{\mathrm{rank}}} + c_{\mathrm{rank}},
$$

EXPECT_SLOT_0363

$$
\ell_{\varphi_{\mathrm{cls}}}(x) := \log\frac{\varphi_{\mathrm{cls}}(x)}{1 - \varphi_{\mathrm{cls}}(x)} = \frac{e_{\mathrm{resid}}(x) - e_{\mathrm{real}}(x)}{\tau_{\mathrm{rank}}}.
$$

EXPECT_SLOT_0364

$$
\ell_{\varphi_{\mathrm{cls}}}(x) - \ell_{w_{\mathrm{rank}}}(x) = \frac{e_{\mathrm{resid}}(x)}{\tau_{\mathrm{rank}}} - c_{\mathrm{rank}}.
$$

EXPECT_SLOT_0365

$$
w_{\mathrm{rank}}(x_{1}) < w_{\mathrm{rank}}(x_{2})\quad \Leftrightarrow \quad e_{\mathrm{real}}(x_{2}) < e_{\mathrm{real}}(x_{1}).
$$

EXPECT_SLOT_0366

$$
w_{\mathrm{rank}}(x_{1}) < w_{\mathrm{rank}}(x_{2})\quad \Leftrightarrow \quad\varphi_{\mathrm{cls}}(x_{1}) < \varphi_{\mathrm{cls}}(x_{2}).
$$

EXPECT_SLOT_0367

$$
\frac{u}{1 - u} < \frac{v}{1 - v}\quad \Leftrightarrow \quad u(1 - v) < v(1 - u)\quad \Leftrightarrow \quad u < v,
$$

EXPECT_SLOT_0368

$$
|e_{\mathrm{resid}}(x) - \overline{e}| \leq \varepsilon_{\mathrm{rank}}\quad (\forall x \in \mathcal{X}),
$$

EXPECT_SLOT_0369

$$
|\log w_{\mathrm{rank}}(x_{1}) - \log w_{\mathrm{rank}}(x_{2})| > \frac{2\varepsilon_{\mathrm{rank}}}{\tau_{\mathrm{rank}}}
$$

EXPECT_SLOT_0370

$$
\Delta e = e_{\mathrm{real}}(x_{1}) - e_{\mathrm{real}}(x_{2}),\qquad \Delta e_{\mathrm{resid}} = e_{\mathrm{resid}}(x_{1}) - e_{\mathrm{resid}}(x_{2}).
$$

EXPECT_SLOT_0371

$$
\ell_{w_{\mathrm{rank}}}(x_{1}) - \ell_{w_{\mathrm{rank}}}(x_{2}) = - \Delta e/\tau_{\mathrm{rank}},
$$

EXPECT_SLOT_0372

$$
\ell_{\varphi_{\mathrm{cls}}}(x_{1}) - \ell_{\varphi_{\mathrm{cls}}}(x_{2}) = (\Delta e_{\mathrm{resid}} - \Delta e)/\tau_{\mathrm{rank}}.
$$
