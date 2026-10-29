EXPECT_SLOT_0193

$$
\begin{aligned}
D_{\mathrm{KL}}(pP_{\mathrm{obs},1} \parallel qP_{\mathrm{obs},2}) &= D_{\mathrm{KL}}(p \parallel q)\\
 &\quad + \sum_{h_{\mathrm{obs}}}p(h_{\mathrm{obs}})\,D_{\mathrm{KL}}\!\left( P_{\mathrm{obs},1}( \cdot \mid h_{\mathrm{obs}}) \parallel P_{\mathrm{obs},2}( \cdot \mid h_{\mathrm{obs}}) \right).
\end{aligned}
$$

EXPECT_SLOT_0207

$$
\begin{aligned}
{\Delta J}_{n + 1,k}(x_{\mathrm{aug}}) &= Q_{n,k}^{O,\mathrm{path}}\big(x_{\mathrm{aug}},v_{\mathrm{nom}}^{C}(k,x_{\mathrm{aug}})\big)\\
 &\quad - Q_{n,k}^{O,\mathrm{path}}\big(x_{\mathrm{aug}},v_{\mathrm{nom}}^{O}(k,x_{\mathrm{aug}})\big) \\
 &\quad + {\Delta J}_{n,k + 1}(x_{\mathrm{aug},C}').
\end{aligned}
$$

EXPECT_SLOT_0376

$$
\begin{aligned}
(1 - \gamma_{\Delta})Q^{\Delta,*}(x) &= (1 - \gamma_{\Delta})\log c(x)\\
 &\quad + \gamma_{\Delta}\sum_{x'}K_{\Delta}^{\pi}(x' \mid x)\,\big[(1 - \gamma_{\Delta})Q^{\Delta,*}(x')\big] \\
 &= \Big({\widetilde{T}}_{\Delta}^{\pi}\big[(1 - \gamma_{\Delta})Q^{\Delta,*}\big]\Big)(x).
\end{aligned}
$$
