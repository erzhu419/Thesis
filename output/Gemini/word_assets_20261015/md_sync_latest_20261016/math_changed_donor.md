EXPECT_SLOT_0157

$$
\| Q_{n} - Q^{*} \|_{\infty} \leq \gamma^{n} \| Q_{0} - Q^{*} \|_{\infty}. 
$$

EXPECT_SLOT_0158

$$
\| Q_{m} - Q_{n} \|_{\infty} \leq \frac{\gamma^{n}}{1 - \gamma} \| Q_{1} - Q_{0} \|_{\infty}. 
$$

EXPECT_SLOT_0159

$$
\| Q_{n} - Q^{*} \|_{\infty} \leq \frac{\gamma^{n}}{1 - \gamma} \| Q_{1} - Q_{0} \|_{\infty}. 
$$

EXPECT_SLOT_0160

$$
\mathcal{T}_{\mathrm{fb}}(Q_{\mathrm{sc}}) = (\gamma + \lambda_{\mathrm{fb}})Q_{\mathrm{sc}},\qquad \gamma,\lambda_{\mathrm{fb}} \geq 0. 
$$

EXPECT_SLOT_0161

$$
|\mathcal{T}_{\mathrm{fb}}(Q_{\mathrm{sc},1}) - \mathcal{T}_{\mathrm{fb}}(Q_{\mathrm{sc},2})| = (\gamma + \lambda_{\mathrm{fb}})|Q_{\mathrm{sc},1} - Q_{\mathrm{sc},2}|. 
$$

EXPECT_SLOT_0164

$$
\left| \mathbb{E}_{p}[ V] - \mathbb{E}_{p^{\circ}}[ V] \right| = \left| \mathbb{E}_{(x,y) \sim \chi}[ V(x) - V(y)] \right| \leq L_{V}\mathbb{E}_{\chi}[ d_{\mathcal{S}}(x,y)]. 
$$

EXPECT_SLOT_0166

$$
\mathcal{Z}_{\beta_{\mathrm{tail}}} = \left\{ \zeta_{\mathrm{risk}}:\ 0 \leq \zeta_{\mathrm{risk}} \leq \frac{1}{\beta_{\mathrm{tail}}},\quad\mathbb{E}_{\mathbb{P}_{0}}[\zeta_{\mathrm{risk}}] = 1 \right\}. 
$$

EXPECT_SLOT_0167

$$
\mathrm{CVaR}_{\beta_{\mathrm{tail}}}(C_{\mathrm{traj}}) = \sup_{\zeta_{\mathrm{risk}} \in \mathcal{Z}_{\beta_{\mathrm{tail}}}}\mathbb{E}_{\mathbb{P}_{0}}[\zeta_{\mathrm{risk}}C_{\mathrm{traj}}]. 
$$

EXPECT_SLOT_0168

$$
\begin{aligned}
\mathbb{E}_{\mathbb{P}_{0}}[\zeta_{\mathrm{risk}}C_{\mathrm{traj}}] &= \eta_{\mathrm{risk}} + \mathbb{E}_{\mathbb{P}_{0}}[\zeta_{\mathrm{risk}}(C_{\mathrm{traj}} - \eta_{\mathrm{risk}})] \\
 &\leq \eta_{\mathrm{risk}} + \frac{1}{\beta_{\mathrm{tail}}}\mathbb{E}_{\mathbb{P}_{0}}[(C_{\mathrm{traj}} - \eta_{\mathrm{risk}})_{+}].
\end{aligned} 
$$

EXPECT_SLOT_0178

$$
g_{j}\left( \mathrm{clip}_{1}(a_{\mathrm{inv},j}) - a_{\mathrm{inv},j} \right) = 0 
$$

EXPECT_SLOT_0179

$$
G_{z}^{- 1}G_{\mathrm{ref}}a^{\mathrm{ref}} \in [-1,1]^{n_{a}}. 
$$

EXPECT_SLOT_0180

$$
G_{z}^{2} = I_{n_{a}},\qquad G_{z}^{- 1} = G_{z}. 
$$

EXPECT_SLOT_0181

$$
a_{z} = G_{z}G_{\mathrm{ref}}a^{\mathrm{ref}} 
$$

EXPECT_SLOT_0182

$$
G_{z}a_{z} = G_{\mathrm{ref}}a^{\mathrm{ref}}. 
$$

EXPECT_SLOT_0183

$$
g_{z,j}a_{z,j} = g_{z,j}^{2}g_{\mathrm{ref},j}a^{\mathrm{ref}}_{j} = g_{\mathrm{ref},j}a^{\mathrm{ref}}_{j}. 
$$

EXPECT_SLOT_0184

$$
\mathrm{clip}(v_{\mathrm{nom},1} + \varepsilon) = \mathrm{clip}(v_{\mathrm{nom},2} + \varepsilon). 
$$

EXPECT_SLOT_0185

$$
u_{\mathrm{exec},k} = \mathrm{clip}\big(v_{\mathrm{nom}}(k,x_{\mathrm{phys},k}) + \varepsilon_{k}\big),\qquad x_{\mathrm{phys},k + 1} = F_{\mathrm{phys}}(x_{\mathrm{phys},k},u_{\mathrm{exec},k},e_{k}). 
$$

EXPECT_SLOT_0186

$$
J_{n,k}^{v_{\mathrm{nom}}}(x_{\mathrm{phys},k}) = \sum_{i = 0}^{n - 1}r(x_{\mathrm{phys},k + i},u_{\mathrm{exec},k + i},e_{k + i}),\qquad J_{0,k}^{v_{\mathrm{nom}}} = 0. 
$$

EXPECT_SLOT_0187

$$
G_{z_{k}}\left( G_{z_{k}}G_{\mathrm{ref}}a^{\mathrm{ref}}(x_{\mathrm{phys}}) \right) = G_{\mathrm{ref}}a^{\mathrm{ref}}(x_{\mathrm{phys}}), 
$$

EXPECT_SLOT_0188

$$
\rho_{k} = U(\cdots U(U(\rho_{0},\xi_{0}),\xi_{1})\cdots,\xi_{k - 1}),\qquad a_{k} = S(\rho_{k},s_{k}). 
$$

EXPECT_SLOT_0189

$$
G_{z_{k}}a(k,x_{\mathrm{phys}}) = G_{z_{k}'}a(k,x_{\mathrm{phys}}), 
$$

EXPECT_SLOT_0190

$$
\begin{aligned}
p_{\mathrm{correct}} &= \frac{1}{2}\sum_{\omega}p(\omega)\mathbf{1}\{ D_{\mathrm{cls}}(Y_{1}(\omega)) = 1\} \\
 &\quad + \frac{1}{2}\sum_{\omega}p(\omega)\mathbf{1}\{ D_{\mathrm{cls}}(Y_{2}(\omega)) = 2\}.
\end{aligned} 
$$

EXPECT_SLOT_0191

$$
p_{\mathrm{correct}} = \frac{1}{2}\sum_{\omega}p(\omega) = \frac{1}{2}. 
$$

EXPECT_SLOT_0193

$$
D_{\mathrm{KL}}(pP_{\mathrm{obs},1} \parallel qP_{\mathrm{obs},2}) = D_{\mathrm{KL}}(p \parallel q) + \sum_{h_{\mathrm{obs}}}p(h_{\mathrm{obs}})\,D_{\mathrm{KL}}\!\left( P_{\mathrm{obs},1}( \cdot \mid h_{\mathrm{obs}}) \parallel P_{\mathrm{obs},2}( \cdot \mid h_{\mathrm{obs}}) \right). 
$$

EXPECT_SLOT_0194

$$
\log\frac{p(h_{\mathrm{obs}})P_{\mathrm{obs},1}(y \mid h_{\mathrm{obs}})}{q(h_{\mathrm{obs}})P_{\mathrm{obs},2}(y \mid h_{\mathrm{obs}})} = \log\frac{p(h_{\mathrm{obs}})}{q(h_{\mathrm{obs}})} + \log\frac{P_{\mathrm{obs},1}(y \mid h_{\mathrm{obs}})}{P_{\mathrm{obs},2}(y \mid h_{\mathrm{obs}})}. 
$$

EXPECT_SLOT_0195

$$
p(h_{\mathrm{obs}})\log\frac{p(h_{\mathrm{obs}})}{q(h_{\mathrm{obs}})}\sum_{y}P_{\mathrm{obs},1}(y \mid h_{\mathrm{obs}}) = p(h_{\mathrm{obs}})\log\frac{p(h_{\mathrm{obs}})}{q(h_{\mathrm{obs}})}, 
$$

EXPECT_SLOT_0196

$$
p(h_{\mathrm{obs}})\sum_{y}P_{\mathrm{obs},1}(y \mid h_{\mathrm{obs}})\log\frac{P_{\mathrm{obs},1}(y \mid h_{\mathrm{obs}})}{P_{\mathrm{obs},2}(y \mid h_{\mathrm{obs}})}. 
$$

EXPECT_SLOT_0197

$$
D_{\mathrm{KL}}(\mathsf{P}_{z}^{\pi_{\mathrm{beh}}} \parallel \mathsf{P}_{z'}^{\pi_{\mathrm{beh}}}) = \sum_{k = 0}^{N_{\mathrm{hor}} - 1}\mathbb{E}_{\mathsf{P}_{z}^{\pi_{\mathrm{beh}}}}\left\lbrack D_{\mathrm{KL}}\big(P_{z}( \cdot \mid H_{k},a_{k}) \parallel P_{z'}( \cdot \mid H_{k},a_{k})\big) \right\rbrack, 
$$

EXPECT_SLOT_0198

$$
v_{\mathrm{nom}}^{C} = G_{z}G_{\widehat{z}}G_{\mathrm{ref}}a^{\mathrm{ref}},\qquad v_{\mathrm{nom}}^{O} = G_{\mathrm{ref}}a^{\mathrm{ref}}. 
$$

EXPECT_SLOT_0199

$$
\Delta v_{\mathrm{nom},j} = (g_{z,j}g_{\widehat{z},j} - 1)g_{\mathrm{ref},j}a^{\mathrm{ref}}_{j}. 
$$

EXPECT_SLOT_0200

$$
(\Delta v_{\mathrm{nom},j})^{2} = \begin{cases}
0, & g_{z,j} = g_{\widehat{z},j}, \\
4(a^{\mathrm{ref}}_{j})^{2}, & g_{z,j} \neq g_{\widehat{z},j},
\end{cases} 
$$

EXPECT_SLOT_0201

$$
\| \Delta v_{\mathrm{nom}} \|_{2}^{2} = 4\sum_{j = 1}^{n_{a}}(a^{\mathrm{ref}}_{j})^{2}\mathbf{1}\{ g_{z,j} \neq g_{\widehat{z},j}\}, 
$$

EXPECT_SLOT_0202

$$
\| \Delta v_{\mathrm{nom}} \|_{2} = 2\sqrt{\sum_{j = 1}^{n_{a}}(a^{\mathrm{ref}}_{j})^{2}\mathbf{1}\{ g_{z,j} \neq g_{\widehat{z},j}\}}. 
$$

EXPECT_SLOT_0203

$$
\begin{aligned}
Q_{n,k}^{O,\mathrm{path}}(x_{\mathrm{aug}},v_{\mathrm{nom}}) &= r\big(x_{\mathrm{phys}},\mathrm{clip}(v_{\mathrm{nom}} + \varepsilon_{k}),e_{k}\big) \\
 &\quad + V_{n,k + 1}^{O,\mathrm{path}}\!\left( F_{\mathrm{aug}}(x_{\mathrm{aug}},v_{\mathrm{nom}},e_{k}) \right).
\end{aligned} 
$$

EXPECT_SLOT_0204

$$
\begin{aligned}
\Delta_{n,k}^{O}(x_{\mathrm{aug}}) &= Q_{n,k}^{O,\mathrm{path}}\big(x_{\mathrm{aug}},v_{\mathrm{nom}}^{C}(k,x_{\mathrm{aug}})\big) \\
 &\quad - Q_{n,k}^{O,\mathrm{path}}\big(x_{\mathrm{aug}},v_{\mathrm{nom}}^{O}(k,x_{\mathrm{aug}})\big).
\end{aligned} 
$$

EXPECT_SLOT_0205

$$
J_{N,k}^{C}(x_{\mathrm{aug},k}) - J_{N,k}^{O}(x_{\mathrm{aug},k}) = \sum_{i = 0}^{N - 1}\Delta_{N - 1 - i,k + i}^{O}(x_{\mathrm{aug},k + i}^{C}). 
$$

EXPECT_SLOT_0206

$$
x_{\mathrm{aug},C}' = F_{\mathrm{aug}}\!\left( x_{\mathrm{aug}},v_{\mathrm{nom}}^{C}(k,x_{\mathrm{aug}}),e_{k} \right). 
$$

EXPECT_SLOT_0207

$$
\begin{aligned}
{\Delta J}_{n + 1,k}(x_{\mathrm{aug}}) &= Q_{n,k}^{O,\mathrm{path}}\big(x_{\mathrm{aug}},v_{\mathrm{nom}}^{C}(k,x_{\mathrm{aug}})\big) - Q_{n,k}^{O,\mathrm{path}}\big(x_{\mathrm{aug}},v_{\mathrm{nom}}^{O}(k,x_{\mathrm{aug}})\big) \\
 &\quad + {\Delta J}_{n,k + 1}(x_{\mathrm{aug},C}').
\end{aligned} 
$$

EXPECT_SLOT_0208

$$
\big|Q_{n,k}^{O,\mathrm{path}}\big(x_{\mathrm{aug}},v_{\mathrm{nom}}^{C}(k,x_{\mathrm{aug}})\big) - Q_{n,k}^{O,\mathrm{path}}\big(x_{\mathrm{aug}},v_{\mathrm{nom}}^{O}(k,x_{\mathrm{aug}})\big)\big| \leq b_{Q,\mathrm{loc}}(k,x_{\mathrm{aug}}), 
$$

EXPECT_SLOT_0209

$$
\big|J_{N,k}^{C}(x_{\mathrm{aug},k}) - J_{N,k}^{O}(x_{\mathrm{aug},k})\big| \leq \sum_{i = 0}^{N - 1}b_{Q,\mathrm{loc}}(k + i,x_{\mathrm{aug},k + i}^{C}). 
$$

EXPECT_SLOT_0210

$$
\begin{aligned}
F_{\mathrm{dyn},e}(x_{\mathrm{aug}},v_{\mathrm{nom}}) &= F_{\mathrm{aug}}(x_{\mathrm{aug}},v_{\mathrm{nom}},e), \\
r_{e}(x_{\mathrm{aug}},v_{\mathrm{nom}}) &= r\big(x_{\mathrm{phys}},\mathrm{clip}(v_{\mathrm{nom}} + \varepsilon(e)),e\big).
\end{aligned} 
$$

EXPECT_SLOT_0211

$$
V_{0,k}^{\pi}(x_{\mathrm{aug}}) = 0, 
$$

EXPECT_SLOT_0212

$$
V_{n + 1,k}^{\pi}(x_{\mathrm{aug}}) = \sum_{e}p(e)\left\lbrack r_{e}\big(x_{\mathrm{aug}},v_{\mathrm{nom}}^{\pi}(k,x_{\mathrm{aug}})\big) + V_{n,k + 1}^{\pi}\!\big(F_{\mathrm{dyn},e}(x_{\mathrm{aug}},v_{\mathrm{nom}}^{\pi}(k,x_{\mathrm{aug}}))\big) \right\rbrack. 
$$

EXPECT_SLOT_0213

$$
Q_{n,k}^{O}(x_{\mathrm{aug}},v_{\mathrm{nom}}) = \sum_{e}p(e)\left\lbrack r_{e}(x_{\mathrm{aug}},v_{\mathrm{nom}}) + V_{n,k + 1}^{O}\!\big(F_{\mathrm{dyn},e}(x_{\mathrm{aug}},v_{\mathrm{nom}})\big) \right\rbrack. 
$$

EXPECT_SLOT_0238

$$
\mathcal{T}_{\rho^{\mathrm{joint}}}Q = \sum_{z,z^{(2)}}\rho^{\mathrm{joint}}(z,z^{(2)})\mathcal{T}_{z,z^{(2)}}Q 
$$

EXPECT_SLOT_0239

$$
Z_{\mathrm{norm}} = \sum_{z}\rho(z)L(z) > 0, 
$$

EXPECT_SLOT_0240

$$
\rho'(z) = \frac{\rho(z)L(z)}{Z_{\mathrm{norm}}}. 
$$

EXPECT_SLOT_0241

$$
\sum_{z}\rho'(z) = \frac{\sum_{z}\rho(z)L(z)}{Z_{\mathrm{norm}}} = \frac{Z_{\mathrm{norm}}}{Z_{\mathrm{norm}}} = 1. 
$$

EXPECT_SLOT_0242

$$
\rho(z) = \sum_{z^{(2)}}\rho^{\mathrm{joint}}(z,z^{(2)}),\qquad \rho^{(2)}(z^{(2)}) = \sum_{z}\rho^{\mathrm{joint}}(z,z^{(2)}). 
$$

EXPECT_SLOT_0243

$$
\sum_{z}\rho(z) = \sum_{z}\sum_{z^{(2)}}\rho^{\mathrm{joint}}(z,z^{(2)}) = 1, 
$$

EXPECT_SLOT_0244

$$
\sum_{z^{(2)}}\rho^{(2)}(z^{(2)}) = \sum_{z^{(2)}}\sum_{z}\rho^{\mathrm{joint}}(z,z^{(2)}) = \sum_{z}\sum_{z^{(2)}}\rho^{\mathrm{joint}}(z,z^{(2)}) = 1. 
$$

EXPECT_SLOT_0245

$$
B_{1} = \begin{pmatrix}
1/2 & 0 \\
0 & 1/2
\end{pmatrix},\qquad B_{2} = \begin{pmatrix}
0 & 1/2 \\
1/2 & 0
\end{pmatrix}. 
$$

EXPECT_SLOT_0246

$$
\sum_{z,z^{(2)}}\rho^{\mathrm{joint}}(z,z^{(2)})\mathcal{T}_{z,z^{(2)}}Q = \sum_{z}\left( \sum_{z^{(2)}}\rho^{\mathrm{joint}}(z,z^{(2)}) \right)\mathcal{T}_{z}Q = \mathcal{T}_{\rho}Q. 
$$

EXPECT_SLOT_0247

$$
\beta_{\mathrm{eff}}(\lambda_{w}) = \beta_{\mathrm{base}} - c_{\mathrm{pen}}\lambda_{w}, 
$$

EXPECT_SLOT_0248

$$
\beta_{\mathrm{eff}}(\lambda_{w,2}) \leq \beta_{\mathrm{eff}}(\lambda_{w,1}). 
$$

EXPECT_SLOT_0249

$$
\beta_{\mathrm{eff}}(\lambda_{w,2}) - \beta_{\mathrm{eff}}(\lambda_{w,1}) = - (\lambda_{w,2} - \lambda_{w,1})c_{\mathrm{pen}} \leq 0 
$$

EXPECT_SLOT_0250

$$
Q_{\mathrm{shared}}(z,s,a) = Q_{z}(s,a). 
$$

EXPECT_SLOT_0251

$$
\begin{aligned}
\mathcal{F}(Q_{\mathrm{sc}}) &= \lambda_{\mathrm{fb}}Q_{\mathrm{sc}}(\gamma Q_{\mathrm{sc}} + R_{1}) + (1 - \lambda_{\mathrm{fb}}Q_{\mathrm{sc}})(\gamma Q_{\mathrm{sc}} + R_{2}) \\
 &= (\gamma + \lambda_{\mathrm{fb}}\Delta R)Q_{\mathrm{sc}} + R_{2},\qquad \Delta R = R_{1} - R_{2}.
\end{aligned} 
$$

EXPECT_SLOT_0252

$$
\mathcal{F}(Q_{\mathrm{sc},1}) - \mathcal{F}(Q_{\mathrm{sc},2}) = (\gamma + \lambda_{\mathrm{fb}}\Delta R)(Q_{\mathrm{sc},1} - Q_{\mathrm{sc},2}). 
$$

EXPECT_SLOT_0253

$$
\big|\mathcal{F}(Q_{\mathrm{sc},1}) - \mathcal{F}(Q_{\mathrm{sc},2})\big| = (\gamma + \lambda_{\mathrm{fb}}\Delta R)\big|Q_{\mathrm{sc},1} - Q_{\mathrm{sc},2}\big|. 
$$

EXPECT_SLOT_0254

$$
\lambda_{\mathrm{fb}}\Delta R = 1 - \gamma, 
$$

EXPECT_SLOT_0305

$$
P_{z}(s' \mid s,a) \geq 0,\qquad \sum_{s'}P_{z}(s' \mid s,a) = 1. 
$$

EXPECT_SLOT_0307

$$
\begin{aligned}
(\mathcal{T}_{z}Q)(s,a) = R_{z}(s,a) + \gamma\Big\lbrack & \sum_{s'}P_{z}(s' \mid s,a)V_{Q}(s') \\
 & - \lambda_{\mathrm{epi}}\Gamma_{\mathrm{epi},z}(s,a) \\
 & - \lambda_{\mathrm{sym}}\Gamma_{\mathrm{sym},z}(s,a) - \kappa_{\mathrm{fix}}\Big\rbrack.
\end{aligned} 
$$

EXPECT_SLOT_0308

$$
(\mathcal{T}_{\mathrm{CS}}Q)(s,a) = \sum_{z}\rho(z)(\mathcal{T}_{z}Q)(s,a). 
$$

EXPECT_SLOT_0309

$$
\begin{aligned}
{\overline{R}}_{\mathrm{pen,CS}}(s,a) = \sum_{z}\rho(z)\Big\lbrack & R_{z}(s,a) - \gamma\lambda_{\mathrm{epi}}\Gamma_{\mathrm{epi},z}(s,a) \\
 & - \gamma\lambda_{\mathrm{sym}}\Gamma_{\mathrm{sym},z}(s,a) - \gamma\kappa_{\mathrm{fix}}\Big\rbrack.
\end{aligned} 
$$

EXPECT_SLOT_0310

$$
\| \mathcal{T}_{\mathrm{CS}}Q_{1} - \mathcal{T}_{\mathrm{CS}}Q_{2} \|_{\infty} \leq \gamma \| Q_{1} - Q_{2} \|_{\infty}. 
$$

EXPECT_SLOT_0311

$$
T_{\mathrm{symfb}}(Q_{\mathrm{sc}}) = R_{0} + \gamma(1 + \lambda_{\mathrm{sym}}\mu_{\mathrm{fb}})Q_{\mathrm{sc}}. 
$$

EXPECT_SLOT_0312

$$
|T_{\mathrm{symfb}}(Q_{\mathrm{sc},1}) - T_{\mathrm{symfb}}(Q_{\mathrm{sc},2})| = c_{\mathrm{symfb}}|Q_{\mathrm{sc},1} - Q_{\mathrm{sc},2}|. 
$$

EXPECT_SLOT_0313

$$
\mu_{\mathrm{fb,crit}} = \frac{1/\gamma - 1}{\lambda_{\mathrm{sym}}}. 
$$

EXPECT_SLOT_0314

$$
f_{c_{\mathrm{coef}}}(x) = \sum_{i = 1}^{n}c_{\mathrm{coef},i}\varphi_{i}(x). 
$$

EXPECT_SLOT_0338

$$
\| V_{n} - V_{P}^{\pi} \|_{\infty} \leq \gamma^{n} \| V_{0} - V_{P}^{\pi} \|_{\infty}. 
$$

EXPECT_SLOT_0339

$$
\| V_{P}^{\pi} \|_{\infty} \leq R_{\max} + \gamma \| V_{P}^{\pi} \|_{\infty}. 
$$

EXPECT_SLOT_0340

$$
\| V_{P}^{\pi} \|_{\infty} \leq \frac{R_{\max}}{1 - \gamma}. 
$$

EXPECT_SLOT_0341

$$
\mathrm{TV}(p,q) = \frac{1}{2}\sum_{j}|p_{j} - q_{j}|. 
$$

EXPECT_SLOT_0342

$$
\left| \sum_{j}(p_{j} - q_{j})f_{j} \right| \leq 2 \| f \|_{\infty}\,\mathrm{TV}(p,q). 
$$

EXPECT_SLOT_0343

$$
\left| \sum_{j}(p_{j} - q_{j})f_{j} \right| \leq \sum_{j}|p_{j} - q_{j}|\,|f_{j}| \leq \| f \|_{\infty}\sum_{j}|p_{j} - q_{j}|. 
$$

EXPECT_SLOT_0344

$$
\varepsilon_{P} = \max_{s,a}\mathrm{TV}\!\left( P_{\mathrm{tgt}}( \cdot \mid s,a),\ {\widehat{P}}_{R}( \cdot \mid s,a) \right). 
$$

EXPECT_SLOT_0345

$$
\| V_{\mathrm{tgt}}^{\pi} - {\widehat{V}}_{R}^{\pi} \|_{\infty} \leq \frac{2\gamma R_{\max}}{(1 - \gamma)^{2}}\varepsilon_{P}. 
$$

EXPECT_SLOT_0346

$$
\begin{aligned}
\| T_{P_{\mathrm{tgt}}}^{\pi}V_{\mathrm{tgt}}^{\pi} - T_{{\widehat{P}}_{R}}^{\pi}V_{\mathrm{tgt}}^{\pi} \|_{\infty} &\leq 2\gamma \| V_{\mathrm{tgt}}^{\pi} \|_{\infty}\varepsilon_{P} \\
 &\leq \frac{2\gamma R_{\max}}{1 - \gamma}\varepsilon_{P}.
\end{aligned} 
$$

EXPECT_SLOT_0347

$$
\begin{aligned}
\varepsilon_{V} &\leq \| T_{P_{\mathrm{tgt}}}^{\pi}V_{\mathrm{tgt}}^{\pi} - T_{{\widehat{P}}_{R}}^{\pi}V_{\mathrm{tgt}}^{\pi} \|_{\infty} + \| T_{{\widehat{P}}_{R}}^{\pi}V_{\mathrm{tgt}}^{\pi} - T_{{\widehat{P}}_{R}}^{\pi}{\widehat{V}}_{R}^{\pi} \|_{\infty} \\
 &\leq \frac{2\gamma R_{\max}}{1 - \gamma}\varepsilon_{P} + \gamma\varepsilon_{V}.
\end{aligned} 
$$

EXPECT_SLOT_0348

$$
\varepsilon_{r} = \max_{s,a}|R_{\mathrm{tgt}}(s,a) - \widehat{R}(s,a)|, 
$$

EXPECT_SLOT_0351

$$
e_{n_{\mathrm{rem}}} = \| V_{n_{\mathrm{rem}}}^{\pi} - {\widehat{V}}_{n_{\mathrm{rem}}}^{\pi} \|_{\infty},\qquad e_{0} = 0. 
$$

EXPECT_SLOT_0352

$$
e_{n_{\mathrm{rem}} + 1} \leq \varepsilon_{r} + e_{n_{\mathrm{rem}}} + 2n_{\mathrm{rem}}R_{\max}\varepsilon_{P}. 
$$

EXPECT_SLOT_0353

$$
\| V_{K_{\max}}^{\pi} - {\widehat{V}}_{K_{\max}}^{\pi} \|_{\infty} \leq K_{\max}\varepsilon_{r} + K_{\max}(K_{\max} - 1)R_{\max}\varepsilon_{P}. 
$$

EXPECT_SLOT_0354

$$
V_{\mathrm{tgt}}(1) = \frac{1}{1 - \gamma},\qquad \widehat{V}(1) = 1, 
$$

EXPECT_SLOT_0355

$$
\| V_{\mathrm{tgt}}^{\pi} - {\widehat{V}}_{n}^{\pi} \|_{\infty} \rightarrow 0. 
$$

EXPECT_SLOT_0356

$$
V_{\mathrm{tgt}}^{\pi'}(0) = \sum_{k = 1}^{\infty}\gamma^{k} = \frac{\gamma}{1 - \gamma},\qquad {\widehat{V}}^{\pi'}(0) = 0. 
$$

EXPECT_SLOT_0357

$$
\begin{aligned}
X_{\delta} &= \mathbb{R}^{|\mathcal{S}|\,|\mathcal{A}|\,|\mathcal{S}|}, \\
\mathcal{C}_{\delta} &= \Big\{\delta \in X_{\delta}:\ P_{\mathrm{sim}} + \delta \geq 0,\ \sum_{s'}\delta(s,a,s') = 0\Big\}, \\
F &:\mathcal{C}_{\delta} \rightarrow \mathcal{C}_{\delta},
\end{aligned} 
$$

EXPECT_SLOT_0358

$$
\| \delta \|_{\infty} = \max_{s,a,s'}|\delta(s,a,s')|. 
$$

EXPECT_SLOT_0359

$$
\| F(\delta_{1}) - F(\delta_{2}) \|_{\infty} \leq L \| \delta_{1} - \delta_{2} \|_{\infty},\qquad 0 \leq L < 1, 
$$

EXPECT_SLOT_0360

$$
\| \delta_{n} - \delta^{*} \|_{\infty} \leq L^{n} \| \delta_{0} - \delta^{*} \|_{\infty}. 
$$

EXPECT_SLOT_0361

$$
\| F(\delta_{1}) - F(\delta_{2}) \|_{\infty} \leq L_{\mathrm{adj}}L_{\mathrm{occ}}B \| \delta_{1} - \delta_{2} \|_{\infty}, 
$$

EXPECT_SLOT_0373

$$
\begin{aligned}
(T_{\Delta}^{\pi}Q^{\Delta})(x) &= \log c(x) + \gamma_{\Delta}\sum_{x'}K_{\Delta}^{\pi}(x' \mid x)Q^{\Delta}(x'), \\
({\widetilde{T}}_{\Delta}^{\pi}Q^{\Delta})(x) &= (1 - \gamma_{\Delta})\log c(x) + \gamma_{\Delta}\sum_{x'}K_{\Delta}^{\pi}(x' \mid x)Q^{\Delta}(x'),
\end{aligned} 
$$

EXPECT_SLOT_0374

$$
\begin{aligned}
{\widetilde{Q}}^{\Delta,*}(x) &= (1 - \gamma_{\Delta})\,Q^{\Delta,*}(x), \\
\exp\!\big({\widetilde{Q}}^{\Delta,*}(x)\big) &= \Big\lbrack \exp\!\big(Q^{\Delta,*}(x)\big)\Big\rbrack^{1 - \gamma_{\Delta}}.
\end{aligned} 
$$

EXPECT_SLOT_0375

$$
\| {\widetilde{T}}_{\Delta}^{\pi}Q_{1}^{\Delta} - {\widetilde{T}}_{\Delta}^{\pi}Q_{2}^{\Delta} \|_{\infty} \leq \gamma_{\Delta} \| Q_{1}^{\Delta} - Q_{2}^{\Delta} \|_{\infty}. 
$$

EXPECT_SLOT_0376

$$
\begin{aligned}
(1 - \gamma_{\Delta})Q^{\Delta,*}(x) &= (1 - \gamma_{\Delta})\log c(x) + \gamma_{\Delta}\sum_{x'}K_{\Delta}^{\pi}(x' \mid x)\,\big[(1 - \gamma_{\Delta})Q^{\Delta,*}(x')\big] \\
 &= \Big({\widetilde{T}}_{\Delta}^{\pi}\big[(1 - \gamma_{\Delta})Q^{\Delta,*}\big]\Big)(x).
\end{aligned} 
$$
