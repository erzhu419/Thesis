$$a = \max\Big(0,\ \min\big\{\,t^{\mathrm{d}} - t^{\mathrm{s}},\ a_{\max}\,\big\}\Big) $$

$$a = \min\Big\{\,a_{\max},\ \max\Big(0,\ k_f\,\big(h_f^{*} - h_f\big) + k_b\,\big(\hat{h}_b - h_b^{*}\big)\Big)\,\Big\} $$

$$\mathcal{M} = \big(\mathcal{S},\ \mathcal{A},\ P,\ R,\ \rho_0,\ \gamma\big) $$

$$J(\pi) = \mathbb{E}_{s_0 \sim \rho_0,\, a_t \sim \pi(\cdot\mid s_t),\, s_{t+1} \sim P(\cdot\mid s_t,a_t)}\left[\,\sum_{t=0}^{\infty} \gamma^{t}\, r_t\,\right] $$

$$V^{\pi}(s) = \mathbb{E}_{a\sim\pi(\cdot\mid s)}\Big[\,R(s,a) + \gamma\,\mathbb{E}_{s'\sim P(\cdot\mid s,a)}\big[\,V^{\pi}(s')\,\big]\,\Big], \qquad Q^{\pi}(s,a) = R(s,a) + \gamma\,\mathbb{E}_{s'\sim P(\cdot\mid s,a)}\big[\,V^{\pi}(s')\,\big] $$

$$(\mathcal{T}^{*}Q)(s,a) = R(s,a) + \gamma\,\mathbb{E}_{s'\sim P(\cdot\mid s,a)}\Big[\,\max_{a'\in\mathcal{A}} Q(s',a')\,\Big] $$

$$Q(s_t,a_t) \;\leftarrow\; Q(s_t,a_t) + \eta\,\Big[\,r_t + \gamma\,\max_{a'\in\mathcal{A}} Q(s_{t+1},a') - Q(s_t,a_t)\,\Big] $$

$$y_t = r_t + \gamma\,(1-d_t)\,\max_{a'\in\mathcal{A}} Q_{\bar{\theta}}(s_{t+1},a'), \qquad \mathcal{L}(\theta) = \mathbb{E}_{(s_t,a_t,r_t,s_{t+1},d_t)\sim\mathcal{D}}\Big[\,\big(Q_{\theta}(s_t,a_t) - y_t\big)^{2}\,\Big] $$

$$\nabla_{\theta} J(\mu_{\theta}) \;\approx\; \mathbb{E}_{s_t\sim\mathcal{D}}\Big[\,\nabla_{\theta} \mu_{\theta}(s_t)\; \nabla_a Q_{\phi}(s_t,a)\big|_{a=\mu_{\theta}(s_t)}\,\Big] $$

$$y_t = r_t + \gamma\,(1-d_t)\,\min_{k\in\{1,2\}} Q_{\phi'_k}\big(s_{t+1},\ \mu_{\theta'}(s_{t+1}) + \epsilon\big), \qquad \epsilon \sim \mathrm{clip}\big(\mathcal{N}(0,\sigma^2),\ -c,\ c\,\big) $$

$$\nabla_{\theta} J(\pi_{\theta}) = \mathbb{E}_{\pi_{\theta}}\Big[\,\sum_{t=0}^{\infty} \gamma^t\,\nabla_{\theta} \log \pi_{\theta}(a_t\mid s_t)\; Q^{\pi_{\theta}}(s_t,a_t)\,\Big] $$

$$\mathcal{L}^{\mathrm{CLIP}}(\theta) = \mathbb{E}_t\Big[\,\min\Big(\rho_t(\theta)\,\hat{A}_t,\ \mathrm{clip}\big(\rho_t(\theta),\ 1-\varepsilon,\ 1+\varepsilon\big)\,\hat{A}_t\Big)\Big], \qquad \rho_t(\theta) = \frac{\pi_{\theta}(a_t\mid s_t)}{\pi_{\theta_{\mathrm{old}}}(a_t\mid s_t)} $$

$$J_{\mathrm{SAC}}(\pi) = \mathbb{E}_{\pi}\left[\,\sum_{t=0}^{\infty} \gamma^{t}\,\Big(\,r_t + \alpha\,\mathcal{H}\big(\pi(\cdot\mid s_t)\big)\,\Big)\,\right] $$

$$y_t = r_t + \gamma\,(1-d_t)\,\left(\,\min_{k\in\{1,2\}} Q_{\phi'_k}(s_{t+1},a'_t) - \alpha \log \pi_{\theta}(a'_t\mid s_{t+1})\,\right), \qquad \mathcal{L}_{Q}(\phi_k) = \mathbb{E}_{(s_t,a_t,r_t,s_{t+1},d_t)\sim\mathcal{D}}\Big[\,\big(Q_{\phi_k}(s_t,a_t) - y_t\big)^{2}\,\Big] $$

$$\mathcal{L}_{\pi}(\theta) = \mathbb{E}_{s_t\sim\mathcal{D},\, a_t\sim\pi_{\theta}(\cdot\mid s_t)}\left[\,\alpha \log \pi_{\theta}(a_t\mid s_t) - \min_{k\in\{1,2\}} Q_{\phi_k}(s_t,a_t)\,\right] $$

$$\mathcal{L}(\alpha) = \mathbb{E}_{s_t\sim\mathcal{D},\, a_t\sim\pi_{\theta}(\cdot\mid s_t)}\Big[\,-\alpha\,\big(\log \pi_{\theta}(a_t\mid s_t) + \bar{\mathcal{H}}\,\big)\,\Big] $$

$$J_{\mathrm{R}}(\pi) = \inf_{P\in\mathcal{P}}\ \mathbb{E}_{\pi,\,P}\left[\,\sum_{t=0}^{\infty} \gamma^{t}\, r_t\,\right] $$

$$(\mathcal{T}_{\mathrm{R}}Q)(s,a) = R(s,a) + \gamma\,\inf_{p\in\mathcal{P}_{s,a}}\ \mathbb{E}_{s'\sim p}\Big[\,\max_{a'\in\mathcal{A}} Q(s',a')\,\Big] $$

$$\min_{\|\delta\|\,\le\,\rho}\ \big\langle p^{\circ} + \delta,\ V \big\rangle \;=\; \big\langle p^{\circ},\ V \big\rangle \;-\; \rho\,\|V\|_{*} $$

$$\inf_{q\in\mathcal{P}_{s,a}}\ \mathbb{E}_{s'\sim q}\big[\,V(s')\,\big] \;=\; \mathbb{E}_{s'\sim p^{\circ}}\big[\,V(s')\,\big] \;-\; \sigma_{\mathcal{P}_{s,a}}(-V) $$

$$\mathrm{CVaR}_{\nu}(Z) \;=\; \min_{\xi\in\mathbb{R}}\ \left\{\,\xi + \frac{1}{\nu}\,\mathbb{E}\big[\,(Z - \xi)_{+}\,\big]\,\right\} \;=\; \sup_{0\,\le\,\zeta\,\le\,1/\nu,\ \mathbb{E}[\zeta]=1}\ \mathbb{E}\big[\,\zeta\,Z\,\big], \qquad (x)_{+} = \max\{x,\,0\} $$

$$b^{+}(z) = \frac{b(z)\,P(s'\mid s,a,z)}{\sum_{\tilde{z}\in\mathcal{Z}}\, b(\tilde{z})\,P(s'\mid s,a,\tilde{z})}, \qquad b'(z') = \sum_{z\in\mathcal{Z}}\, \Psi(z' \mid z)\, b^{+}(z) $$

$$p_{\mathrm{sim}}(s,\ a,\ s') = \mu_{\mathrm{sim}}(s,\ a)\,P_{\mathrm{sim}}(s' \mid s,\ a), \qquad p_{\mathrm{tgt}}(s,\ a,\ s') = \mu_{\mathrm{tgt}}(s,\ a)\,P_{\mathrm{tgt}}(s' \mid s,\ a) $$

$$\mathbb{E}_{x \sim p}\big[\,f(x)\,\big] \;=\; \mathbb{E}_{x \sim q}\left[\,\frac{p(x)}{q(x)}\,f(x)\,\right] $$

$$F(s,\ a,\ s') \;-\; G(s,\ a) \;=\; \log \frac{p_{\mathrm{tgt}}(s,\ a,\ s')}{p_{\mathrm{sim}}(s,\ a,\ s')} \;-\; \log \frac{\mu_{\mathrm{tgt}}(s,\ a)}{\mu_{\mathrm{sim}}(s,\ a)} \;=\; \log \frac{P_{\mathrm{tgt}}(s' \mid s,\ a)}{P_{\mathrm{sim}}(s' \mid s,\ a)} $$

$$\mathcal{L}_{V}(\nu) \;=\; \mathbb{E}_{(s,\,a) \sim \mathcal{D}_{\mathrm{off}}}\Big[\, \big|\, \tau_e - \mathbb{I}\big\{A(s,\ a) < 0\big\} \,\big|\; A(s,\ a)^{2} \,\Big] $$

$$\mathcal{L}_{\pi}(\phi) \;=\; -\,\mathbb{E}_{(s,\,a) \sim \mathcal{D}_{\mathrm{off}}}\Big[\, w_{\mathrm{IQL}}(s,\ a)\,\log \pi_{\phi}(a \mid s) \,\Big], \qquad w_{\mathrm{IQL}}(s,\ a) = \min\big(\exp(\beta\, A(s,\ a)),\ M\big) $$

$$\mathcal{L}_{\mathrm{wm}}(\theta_{\mathrm{wm}}) \;=\; \mathbb{E}_{(s,\,a,\,r,\,s') \sim \mathcal{D}}\Big[\, -\log p_{\theta_{\mathrm{wm}}}(s' \mid s,\ a) \;+\; \lambda_r\,\big(\hat{r}_{\theta_{\mathrm{wm}}}(s,\ a) - r\big)^{2} \,\Big] $$

$$\hat{V}_{k}(s_0) \;=\; \sum_{j=0}^{k-1} \gamma^{\,j}\, \hat{r}_j \;+\; \gamma^{\,k}\, \hat{V}(\hat{s}_k) $$

$$\big\|\, V^{\pi}_{\mathrm{true}} - V^{\pi}_{\mathrm{mdl}} \,\big\|_{\infty} \;\leq\; \frac{\varepsilon_r}{1 - \gamma} \;+\; \frac{2\,\gamma\, r_{\max}\, \varepsilon_P}{(1 - \gamma)^{2}} $$

$$\hat{y}_{\mathrm{res}}(x) \;=\; \hat{y}_{\mathrm{sim}}(x) \;+\; \hat{\Delta}(x) $$

$$\min_{\Xi}\; \frac{1}{2}\,\big\|\, Y - \Theta(X)\,\Xi \,\big\|_{F}^{2} \;+\; \lambda_{\Xi}\, \|\Xi\|_{1} $$

$$r^{\mathrm{sched}} \;=\; -\Big[\,w_f\,e_f + (1 - w_f)\,e_b\,\Big] \;-\; \lambda_{\mathrm{imb}}\,\big\vert{}\,h_f - R_h\,h_b\,\big\vert{} \;-\; \big(P_f + P_b\big) $$

$$r_i^{\mathrm{pass}} \;=\; -\int_{t_i}^{t_{i+1}} \Big(\,2\,N^{w}(t) + N^{b}(t)\,\Big)\,\mathrm{d}t $$

$$G \;=\; \frac{1}{N_{\mathrm{dep}}}\sum_{p\in\mathcal{P}_{\mathrm{dep}}} \Big(\,2\,T^w_p + T^b_p\,\Big) \;=\; \frac{1}{N_{\mathrm{dep}}}\int_{0}^{T_{\mathrm{hor}}} \Big(\,2\,N^{w}(t) + N^{b}(t)\,\Big)\,\mathrm{d}t $$

$$y^{\mathrm{d}}_{i,j} = y^{\mathrm{a}}_{i,j} + d^{\mathrm{svc}}_{i,j} + a_{i,j} + d^{\mathrm{exit}}_{i,j}, \qquad y^{\mathrm{a}}_{i,j+1} = y^{\mathrm{d}}_{i,j} + d^{\mathrm{run}}_{i,j} $$

$$h_{j+1} = h_j + \big(d^{\mathrm{svc}}_{i+1,j} - d^{\mathrm{svc}}_{i,j}\big) + \big(a_{i+1,j} - a_{i,j}\big) + \big(d^{\mathrm{exit}}_{i+1,j} - d^{\mathrm{exit}}_{i,j}\big) + \big(d^{\mathrm{run}}_{i+1,j} - d^{\mathrm{run}}_{i,j}\big) $$

$$\mathbb{E}\big[B_v\big] = B \cdot \frac{N_v}{N} $$

$$\boldsymbol{x} = \big[\, E_1(c_1) \oplus E_2(c_2) \oplus \cdots \oplus E_m(c_m) \oplus \boldsymbol{o} \,\big] $$

$$d_j = \min\big(50,\ \lfloor n_j / 2 \rfloor\big) $$

$$\boldsymbol{x} = \big[\, \mathrm{Emb}_{\mathrm{v}}(c_{\mathrm{v}}) \oplus \mathrm{Emb}_{\mathrm{s}}(c_{\mathrm{s}}) \oplus \mathrm{Emb}_{\mathrm{t}}(c_{\mathrm{t}}) \oplus \mathrm{Emb}_{\mathrm{d}}(c_{\mathrm{d}}) \oplus \boldsymbol{o} \,\big] $$

$$\varphi(h) = -\big|\,h - h_s\,\big|, \qquad \omega(h_f, h_b) = \frac{|h_f - h_s|}{|h_f - h_s| + |h_b - h_s| + \epsilon} $$

$$R(h_f, h_b) = \omega(h_f, h_b)\,\varphi(h_f) + \big(1 - \omega(h_f, h_b)\big)\,\varphi(h_b) - 0.5\,\big|\,h_f - h_b\,\big| - 20\,\mathbb{I}\Big(\big|\,h_f - h_s\,\big| > 180 \ \text{或}\ \big|\,h_b - h_s\,\big| > 180\Big) $$

$$e_f = \big|\,h_f - h_f^{*}\,\big|, \quad e_b = \big|\,h_b - h_b^{*}\,\big|, \quad \omega_h = \frac{e_f}{e_f + e_b + 10^{-6}}, \quad R_h = \frac{h_f^{*}}{\max(h_b^{*},\ 10^{-6})}, \quad r_{\mathrm{base}} = -\omega_h\,e_f - (1-\omega_h)\,e_b - \frac{\big|\,h_f - R_h\,h_b\,\big|}{1 + R_h} $$

$$p_f = 20\,I_f\,\tanh\!\left(\frac{e_f - 0.5\,h_f^{*}}{30}\right), \qquad p_b = 20\,I_b\,\tanh\!\left(\frac{e_b - 0.5\,h_b^{*}}{30}\right), \qquad r^{\mathrm{sched}} = r_{\mathrm{base}} - \max\big(0,\ p_f + p_b\big) $$

$$r_{\mathrm{E}}(h_f,\ h_b,\ a) = \exp\!\left(-\frac{|h_f - h_b|}{\tau_h}\right) + w \cdot \exp\!\left(-\frac{a}{60}\right), \qquad a \in [0,\ 60] $$

$$R(h,\ h) = -\big|\,h - h_s\,\big| - 20\,\mathbb{I}\Big(\big|\,h - h_s\,\big| > 180\Big) $$

$$Q^{\pi}(s_i,\ a_i) = \mathbb{E}_{\pi}\left[\, \sum_{j \geq i} \gamma^{\,j-i}\, r_j \;\middle|\; s_i,\ a_i \right], \qquad V^{\pi}(s) = \mathbb{E}_{a \sim \pi(\cdot\mid s)}\big[\, Q^{\pi}(s,\ a) \,\big] $$

$$V^{\pi}_{\mathrm{soft}}(s) = \mathbb{E}_{a \sim \pi(\cdot\mid s)}\Big[\, Q^{\pi}_{\mathrm{soft}}(s,\ a) - \alpha \log \pi(a\mid s) \,\Big] $$

$$\bar{Q}(s,\ a) = \frac{1}{K} \sum_{k=1}^{K} Q_{\phi_k}(s,\ a), \qquad \sigma_Q(s,\ a) = \left[\, \frac{1}{K-1} \sum_{k=1}^{K} \Big(\, Q_{\phi_k}(s,\ a) - \bar{Q}(s,\ a) \,\Big)^2 \,\right]^{1/2} $$

$$\inf_{q \in \mathcal{P}_{s,a}} \mathbb{E}_{s' \sim q}\big[\, V(s') \,\big] \;\geq\; \mathbb{E}_{s' \sim p^{\circ}}\big[\, V(s') \,\big] \;-\; \rho_{\mathrm{ipm}} \cdot \mathrm{Lip}(V) $$

$$\mathbb{E}[W] \;=\; \frac{\mathbb{E}[H^2]}{2\,\mathbb{E}[H]} \;=\; \frac{\mathbb{E}[H]}{2} + \frac{\mathrm{Var}(H)}{2\,\mathbb{E}[H]} \;=\; \frac{\mathbb{E}[H]}{2}\Big(1 + \mathrm{CV}_H^2\Big) $$

$$h_f(a) = h_f + a, \qquad h_b(a) = h_b - a, \qquad a \in \big[0,\ \min(a_{\max},\ h_b)\big] $$

$$h_f(a) + h_b(a) = h_f + h_b, \qquad h_f(a) - h_b(a) = h_f - h_b + 2a $$

$$a_{\mathrm{bal}}^{*} = \Pi_{[0,\, \bar{a}]}\!\left( \frac{h_b - h_f}{2} \right) $$

$$\Delta W(a) = \frac{\lambda}{2}\Big[\big(h_f + a\big)^2 - h_f^2\Big] + \frac{\lambda}{2}\Big[\big(h_b - a\big)^2 - h_b^2\Big] = \lambda a^2 - \lambda a \Delta $$

$$\Delta C(a) = 2\,\Delta W(a) + n \cdot a = 2\lambda a^2 - \big(2\lambda \Delta - n\big) a $$

$$a_{\mathrm{pass}}^{*} = \Pi_{[0,\, \bar{a}]}\!\left( \frac{\Delta}{2} - \frac{n}{4\lambda} \right) $$

$$a_c = \Delta - \frac{n}{2\lambda}, \qquad \Delta C(a) < 0 \iff 0 < a < \Delta - \frac{n}{2\lambda} $$

$$J(\pi_{\theta}) = \mathbb{E}_{\pi_{\theta}}\left[\, \sum_{i=0}^{\infty} \gamma^{\,i} \Big(\, r_i + \alpha\, \mathcal{H}\big(\pi_{\theta}(\cdot\mid s_i)\big) \,\Big) \,\right] $$

$$\pi^{*}(a\mid s) = \frac{\exp\!\big(Q(s,\ a) / \alpha\big)}{\int_{\mathcal{A}} \exp\!\big(Q(s,\ a') / \alpha\big)\,\mathrm{d}a'} $$

$$\Gamma_{\mathrm{epi}}(s,\ a) = \frac{1}{K-1} \sum_{k=1}^{K} \left(\, \hat{Q}_{\phi'_k}(s,\ a) - \frac{1}{K} \sum_{j=1}^{K} \hat{Q}_{\phi'_j}(s,\ a) \,\right)^2 $$

$$\kappa_k = \sum_{l} \big\|\, W^{(l)}_{\phi'_k} \,\big\|_1 $$

$$y_k = r + \gamma\,(1 - d)\,\Big(\, \hat{Q}_{\phi'_k}(s',\ a') - \lambda_{\mathrm{epi}}\, \Gamma_{\mathrm{epi}}(s',\ a') - \lambda_{\mathrm{ale}}\, \kappa_k - \alpha \log \pi_{\theta}(a'\mid s') \,\Big) $$

$$c\,\big\|W_1\big\|_1 + \frac{1}{c}\,\big\|W_2\big\|_1 \;\neq\; \big\|W_1\big\|_1 + \big\|W_2\big\|_1 \qquad (c > 0,\ c \neq 1) $$

$$\mathcal{L}_{Q}(\phi_k) = \mathbb{E}_{(s,\, a,\, r,\, s',\, d) \sim \mathcal{D}}\Big[\, \big( Q_{\phi_k}(s,\ a) - y_k \big)^2 + \beta_{\mathrm{ood}}\, \sigma_Q(s,\ a) \,\Big] $$

$$\mathcal{L}_{\pi}(\theta) = \mathbb{E}_{s \sim \mathcal{D},\, \tilde{a} \sim \pi_{\theta}(\cdot\mid s)}\Big[\, \alpha \log \pi_{\theta}(\tilde{a}\mid s) - \Big(\, \bar{Q}(s,\ \tilde{a}) + \beta_{\mathrm{lcb}}\, \sigma_Q(s,\ \tilde{a}) \,\Big) \,\Big] + \beta_{\mathrm{bc}}\, \mathbb{E}_{(s,\,a)\sim\mathcal{D},\, \tilde{a}\sim\pi_{\theta}(\cdot\mid s)}\Big[\,\big\|\tilde{a} - a\big\|_2^2\,\Big], \qquad \beta_{\mathrm{lcb}} \leq 0 $$

$$(\mathcal{T}_{\Lambda} Q)(s,\ a) = R(s,\ a) + \gamma\,\Big(\, \mathbb{E}_{s' \sim P(\cdot\mid s,a)}\big[\, V^{Q}(s') \,\big] - \Lambda(s,\ a) \,\Big) $$

$$\big\|\, \mathcal{T}_{\Lambda} Q_1 - \mathcal{T}_{\Lambda} Q_2 \,\big\|_{\infty} \;\leq\; \gamma\, \big\|\, Q_1 - Q_2 \,\big\|_{\infty} $$

$$\big|\,(\mathcal{T}_{\Lambda} Q_1)(s,\ a) - (\mathcal{T}_{\Lambda} Q_2)(s,\ a)\,\big| \;\leq\; \gamma \sum_{s'} P(s'\mid s,a)\,\big|\,V^{Q_1}(s') - V^{Q_2}(s')\,\big| \;\leq\; \gamma\,\delta_Q \sum_{s'} P(s'\mid s,a) \;=\; \gamma\,\big\|\,Q_1 - Q_2\,\big\|_{\infty} $$

$$\big\|\, Q_i - Q^{*}_{\Lambda} \,\big\|_{\infty} \;\leq\; \gamma^{\,i}\, \big\|\, Q_0 - Q^{*}_{\Lambda} \,\big\|_{\infty} \;+\; \frac{\varepsilon}{1 - \gamma} $$

$$D_{\mathrm{eff}}(s;\ a_1,\ a_2) \;=\; q_c(s,\ a_1) - q_c(s,\ a_2) \;=\; \Big[\, \bar{Q}(s,\ a_1) - \bar{Q}(s,\ a_2) \,\Big] + \beta_{\mathrm{lcb}} \cdot \Big[\, \sigma_Q(s,\ a_1) - \sigma_Q(s,\ a_2) \,\Big] $$

$$\underset{a\in\mathcal{A}}{\arg\max}\;\Big(\,Q^{\mathrm{ref}}(s,\ a) + c(s)\,\Big) \;=\; \underset{a\in\mathcal{A}}{\arg\max}\; Q^{\mathrm{ref}}(s,\ a) $$

$$\frac{1}{M}\sum_{i=1}^{M} e_{Q,i}^{2} \;=\; \big(\bar{e}_{Q}(s)\big)^{2} \;+\; \frac{1}{M}\sum_{i=1}^{M} \big(e_{Q,i} - \bar{e}_{Q}(s)\big)^{2} $$

$$0 \;\leq\; Q^{\mathrm{ref}}(s,\ a^{*}) - Q^{\mathrm{ref}}(s,\ \hat{a}) \;\leq\; \eta(s,\ \hat{a}) - \eta(s,\ a^{*}) \;\leq\; 2\,\varepsilon_{\mathrm{act}} $$

$$\Delta_{A} = \frac{1}{2}\Big[\big(Y_{10} - Y_{00}\big) + \big(Y_{11} - Y_{01}\big)\Big], \qquad \Delta_{C} = \frac{1}{2}\Big[\big(Y_{01} - Y_{00}\big) + \big(Y_{11} - Y_{10}\big)\Big] $$

$$s',\ \Delta t,\ r \;\sim\; P\big(\,\cdot \;\big|\; s,\ a,\ z\,\big) $$

$$N(\Delta) \;\sim\; \mathrm{Poisson}(\lambda\, \Delta) $$

$$\mathbb{E}\big[N(h)\big] = m_{\mathrm{dem}}\,\lambda\, h, \qquad d^{\mathrm{svc}}(h) = d^{\mathrm{svc}}_0 + \eta^{\mathrm{board}}\, m_{\mathrm{dem}}\,\lambda\, h $$

$$\tilde{P}\big(s',\ z' \;\big|\; s,\ a,\ z\big) \;=\; P_z\big(s' \;\big|\; s,\ a\big)\; \Psi\big(z' \;\big|\; z\big) $$

$$b^{+}(z) \;=\; \frac{b(z)\, L\big(z;\ s,\ a,\ s'\big)}{\displaystyle\sum_{\tilde{z} \in \mathcal{Z}} b(\tilde{z})\, L\big(\tilde{z};\ s,\ a,\ s'\big)} $$

$$b'(z') \;=\; \sum_{z \in \mathcal{Z}} \Psi\big(z' \;\big|\; z\big)\, b^{+}(z) $$

$$\mu_j = 0.9\,\mu_{j-1} + 0.1\,\bar{r}_j, \qquad v_j = 0.9\,v_{j-1} + 0.1\,\big(\bar{r}_j - \mu_j\big)^2, \qquad \chi^{(r)}_j = \frac{\big|\, \bar{r}_j - \mu_j \,\big|}{\max\big(\sqrt{v_j},\ 10^{-6}\big)} $$

$$\chi^{(q)}_j = \begin{cases} \max\!\left(\, \dfrac{\bar{\sigma}_{Q,\,j}}{\bar{\sigma}_{Q,\,j-1}} - 1,\ 0 \,\right), & \bar{\sigma}_{Q,\,j-1} > 10^{-6}, \\[8pt] 0, & \bar{\sigma}_{Q,\,j-1} \le 10^{-6}\ \text{或尚无前次统计值} \end{cases} $$

$$\chi^{(w)}_j = \frac{1}{2K}\, \sum_{k=1}^{K} \big|\, \kappa_k - \kappa'_k \,\big|, \qquad \chi_j = \max\Big\{\, \chi^{(r)}_j,\ \chi^{(q)}_j,\ \chi^{(w)}_j \,\Big\}, \qquad \tilde{\chi}_j = 0.9\,\tilde{\chi}_{j-1} + 0.1\, \chi_j $$

$$L_j(\ell) = \exp\!\left( -\frac{\tilde{\chi}_j^{\,2} \, \big(1 + 0.05\,\ell\big)}{2} \right), \qquad \ell = 0,\ 1,\ \dots,\ 19 $$

$$\lambda_w \;=\; \sum_{\ell = 0}^{19} \rho_j(\ell)\, e^{-0.1\, \ell}, \qquad \beta_{\mathrm{eff}} \;=\; -2 - c_{\mathrm{pen}}\, \lambda_w $$

$$\mathcal{L}_{\pi,\mathrm{base}}(\theta) = \mathbb{E}_{s \sim \mathcal{D},\, \tilde{a} \sim \pi_{\theta}(\cdot\mid s)}\Big[\, \alpha \log \pi_{\theta}(\tilde{a}\mid s) - \Big(\, \bar{Q}(s,\ \tilde{a}) + \beta_{\mathrm{eff}}\, \sigma_Q(s,\ \tilde{a}) \,\Big) \,\Big] + \beta_{\mathrm{bc}}\, \mathcal{L}_{\mathrm{bc}} $$

$$\tilde{a}(x) = \tanh\!\Big(\, \beta_0 + f_{\theta}(x) - f_{\theta}\big(x_{\mathrm{ref}}(x)\big) \,\Big), \qquad a_{\mathrm{hold}}(x) = 30\, \Big(1 + \tilde{a}(x)\Big) $$

$$\tilde{a}_{\mathrm{sym}}(x) = \mathrm{clip}_{[-0.999,\, 0.999]}\!\left(\, \sum_{k=0}^{12} \tilde{a}_k \cdot \frac{\exp\!\big(-\hat{c}(x,\ \tilde{a}_k) / \tau_{\mathrm{boltz}}\big)}{\displaystyle\sum_{k'=0}^{12} \exp\!\big(-\hat{c}(x,\ \tilde{a}_{k'}) / \tau_{\mathrm{boltz}}\big)} \,\right) $$

$$\ell_{\mathrm{sym}}(x) = \frac{\big\|\, D\pi(x) - D\pi_{\mathrm{sym}}(x) \,\big\|_F^2}{1 + \big\|\, D\pi_{\mathrm{sym}}(x) \,\big\|_F^2}, \qquad \mathcal{L}_{\mathrm{JC}}(\theta) = \frac{1}{B}\sum_{i=1}^{B} \ell_{\mathrm{sym}}(x_i) $$

$$y_i = \frac{r_i}{\sigma_r} + \gamma\,(1 - d_i)\,\left(\, \min_{k\in\{1,2,3\}} \hat{Q}_{\phi'_k}(s'_i,\ a'_i) - \alpha \log \pi_{\theta}(a'_i\mid s'_i) - \lambda_{\mathrm{sym}}\, \mathrm{sg}\big(\bar{\mathcal{L}}_{\mathrm{JC}}'\big) \,\right) $$

$$\sigma_{r,\,j+1}^2 = 0.99\, \sigma_{r,\,j}^2 + 0.01\, \mathrm{Var}\big(r_{\mathrm{batch}}\big), \qquad \tilde{r} = \frac{r}{\max\big(\sigma_{r,\,j+1},\ 10^{-6}\big)} $$

$$\mathcal{L}_{\mathrm{bc}}(\theta) = \frac{1}{B} \sum_{(s,\, a) \sim \mathcal{D}_B,\ \tilde{a}\sim\pi_{\theta}(\cdot\mid s)} \big\|\, \tilde{a} - a \,\big\|_2^2 $$

$$\hat{\rho}_j(0) = p_{\mathrm{reset}}, \qquad \hat{\rho}_j(\ell) = (1 - p_{\mathrm{reset}})\,\tilde{\rho}_j(\ell - 1)\quad (\ell = 1,\dots,19), \qquad \rho_j(\ell) = \frac{\hat{\rho}_j(\ell)}{\sum_{m=0}^{19} \hat{\rho}_j(m)} $$

$$D(H \circ G)(x) - D(H \circ G)(y) \;=\; \Big[\, DH\big(G(x)\big) - DH\big(G(y)\big) \,\Big]\, DG(x) \;+\; DH\big(G(y)\big)\, \Big[\, DG(x) - DG(y) \,\Big] $$

$$\big\|\, D(H \circ G)(x) - D(H \circ G)(y) \,\big\| \;\leq\; \Big(\, L_H\, K_G^2 + K_H\, L_G \,\Big)\,\big\|\, x - y \,\big\|, \qquad L_{H \circ G} \;\leq\; L_H\, K_G^2 + K_H\, L_G $$

$$\tilde{a}_{\mathrm{arth}}(x) = \tanh\!\Big(\, \eta\, W_{\mathrm{NAU}}\, G(x) + (1 - \eta)\, \boldsymbol{c}_{\mathrm{quad}} \odot \big( U_{\mathrm{quad}}\, G(x) \big)^{\odot 2} \,\Big) $$

$$\pi(x) - \pi_{\mathrm{ref}}(x) \;=\; \Big[\, \pi(x_0) - \pi_{\mathrm{ref}}(x_0) \,\Big] + \Big[\, D\pi(x_0) - D\pi_{\mathrm{ref}}(x_0) \,\Big]\,d + \int_{0}^{1} \Big\{\, \big[D\pi(\zeta(u)) - D\pi(x_0)\big] - \big[D\pi_{\mathrm{ref}}(\zeta(u)) - D\pi_{\mathrm{ref}}(x_0)\big] \,\Big\}\,d\;\mathrm{d}u $$

$$\big\|\, \pi(x) - \pi_{\mathrm{ref}}(x) \,\big\| \;\leq\; \delta \;+\; \varepsilon\, \big\|\, d \,\big\| \;+\; \frac{L_{\mathrm{err}}}{2}\, \big\|\, d \,\big\|^2 $$

$$V^{Q}(s) = \mathbb{E}_{a \sim p_{\theta}(\cdot\mid s)}\Big[\, Q(s,\ a) - \alpha \log p_{\theta}(a\mid s) \,\Big] $$

$$(\mathcal{T}_z Q)(s,\ a) = R_z(s,\ a) + \gamma\, \Big(\, \mathbb{E}_{s' \sim P_z(\cdot\mid s,a)}\big[\, V^{Q}(s') \,\big] - \lambda_{\mathrm{epi}}\, \Gamma^{z}_{\mathrm{epi}}(s,\ a) - \lambda_{\mathrm{sym}}\, \Gamma^{z}_{\mathrm{sym}}(s,\ a) - \kappa^{z}_{\mathrm{ale}}(s,\ a) \,\Big) $$

$$\mathcal{T}_{\rho} Q \;=\; \sum_{z \in \mathcal{Z}} \rho(z)\, \mathcal{T}_z Q $$

$$J_{T_{\mathrm{hor}}}(\pi) = \mathbb{E}_{\pi}\left[\, \sum_{k:\, t_k \le T_{\mathrm{hor}}} r_k \,\right] $$

$$\bar{m}_{\mathrm{dem}} = \frac{1}{T_{\mathrm{hor}}}\int_{0}^{T_{\mathrm{hor}}} m_{\mathrm{dem}}(t)\,\mathrm{d}t $$

$$\Delta N_{w,\,m,\,k} = N^{p}_{w,\,m,\,k} - N^{0}_{w,\,m,\,k}, \qquad \varepsilon_{m,\,k} = \max\Big\{\,0.1\,N^{0}_{w,\,m,\,k},\ 20\,\Big\} $$

$$a_k = G_{\hat{z}_k}^{-1}\, G_{\mathrm{ref}}\, \mu_{\mathrm{ref}}(s_k) $$

$$\pi_{\mathrm{off}} = \mathcal{L}\big(\mathcal{D}_{\mathrm{off}}\big), \qquad \pi_{\mathrm{on}} = \mathcal{L}\big(\mathcal{D}_{\mathrm{off}},\ \mathcal{D}_{\mathrm{tgt}}\big), \qquad \pi_{\mathrm{cf}} = \mathcal{L}\big(\mathcal{D}_{\mathrm{off}},\ \mathcal{D}_{\mathrm{sim}}\big) $$

$$p_{\mathrm{tgt}}(s,\ a,\ s',\ \Delta t) = \mu_{\mathrm{tgt}}(s,\ a)\,P_{\mathrm{tgt}}(s',\ \Delta t \mid s,\ a), \qquad p_{\mathrm{sim}}(s,\ a,\ s',\ \Delta t) = \mu_{\mathrm{sim}}(s,\ a)\,P_{\mathrm{sim}}(s',\ \Delta t \mid s,\ a) $$

$$\mathrm{Var}\big(\hat{p}_m\big) \;=\; \mathbb{E}\Big[\,\mathrm{Var}\big(\hat{p}_m \;\big|\; \hat{p}_n\big)\,\Big] + \mathrm{Var}\Big(\,\mathbb{E}\big[\hat{p}_m \;\big|\; \hat{p}_n\big]\,\Big) \;=\; \frac{1}{m}\,\mathbb{E}\Big[\,\hat{p}_n\big(1-\hat{p}_n\big)\,\Big] + \mathrm{Var}\big(\hat{p}_n\big) \;=\; p(1-p)\left(\,\frac{1}{n} + \frac{1 - 1/n}{m}\,\right) $$

$$\hat{h}_k^b = \min\left\{\, \max\big(2400,\ 3\,h_k^{b,*}\big),\ \frac{\big(x_j(t_k) - x_{j+1}(t_k)\big)_{+}}{\max(\hat{v}_k,\ 1)} \,\right\} $$

$$\hat{\lambda}_{ol,q} = \sum_{d:\, l \in \mathcal{L}_{od}} \lambda_{od,q} \, \frac{H_l^{-1}}{\displaystyle\sum_{m \in \mathcal{L}_{od}} H_m^{-1}} $$

$$C_k = 2\int_{t_k}^{t_k + \Delta t_k} N^{w}(\tau)\,\mathrm{d}\tau + \int_{t_k}^{t_k + \Delta t_k} N^{b}(\tau)\,\mathrm{d}\tau, \qquad r_k = -C_k $$

$$C_{\mathrm{sys}} \;=\; \int_{0}^{T_{\mathrm{hor}}} \Big(\,2\,N^{w}(\tau) + N^{b}(\tau)\,\Big)\,\mathrm{d}\tau \;=\; \sum_{p \in \mathcal{P}_{\mathrm{dep}}} \left(\, 2\int_{0}^{T_{\mathrm{hor}}} \mathbb{I}_p^{w}(\tau)\,\mathrm{d}\tau + \int_{0}^{T_{\mathrm{hor}}} \mathbb{I}_p^{b}(\tau)\,\mathrm{d}\tau \,\right) \;=\; \sum_{p \in \mathcal{P}_{\mathrm{dep}}} \Big(\, 2\,T_p^{w} + T_p^{b} \,\Big) $$

$$Q^{\pi}(s_i,\ a_i) \;=\; \mathbb{E}_{\pi}\left[\, \sum_{j \geq i} r_j \;\middle|\; s_i,\ a_i \right], \qquad V^{\pi}(s) \;=\; \mathbb{E}_{a \sim \pi(\cdot\mid s)}\big[\, Q^{\pi}(s,\ a) \,\big] $$

$$V_{\mathrm{soft}}(s') \;=\; \mathbb{E}_{a' \sim \pi_{\theta}(\cdot\mid s')}\Big[\, \min_{k \in \{1,2\}} \hat{Q}_{\phi'_k}(s',\ a') - \alpha \log \pi_{\theta}(a' \mid s') \,\Big] $$

$$\mathcal{L}_{V}(\nu) \;=\; \mathbb{E}_{(s,\,a) \sim \mathcal{D}_{\mathrm{off}}}\Big[\, \big|\, \tau_{\mathrm{IQL}} - \mathbb{I}\big\{A(s,\ a) < 0\big\} \,\big|\; A(s,\ a)^{2} \,\Big], \qquad \mathcal{L}_{Q,\mathrm{off}}(\phi_k) \;=\; \mathbb{E}_{(s,\,a,\,r,\,s',\,d) \sim \mathcal{D}_{\mathrm{off}}}\Big[\, \big(Q_{\phi_k}(s,\ a) - y_{\mathrm{off}}\big)^{2} \,\Big] $$

$$\mathcal{L}_{\pi,\mathrm{off}}(\theta) \;=\; -\,\mathbb{E}_{(s,\,a) \sim \mathcal{D}_{\mathrm{off}}}\Big[\, w_{\mathrm{IQL}}(s,\ a)\,\log \pi_{\theta}(a \mid s) \,\Big], \qquad w_{\mathrm{IQL}}(s,\ a) = \min\Big(\exp\big(3\,A(s,\ a)\big),\ 100\Big) $$

$$B_{\mathrm{mix}}(s') \;=\; 0.1\,V_{\nu}(s') + 0.9\,V_{\mathrm{soft}}(s'), \qquad y_{\mathrm{mix}} \;=\; r + (1 - d)\,B_{\mathrm{mix}}(s') $$

$$F_{\psi_{\mathrm{J}}}\big(s,\ a,\ s',\ \Delta t / T_{\mathrm{hor}}\big) - G_{\psi_{\mathrm{M}}}\big(s,\ a\big) \;\approx\; \log \frac{P_{\mathrm{tgt}}(s',\ \Delta t \mid s,\ a)}{P_{\mathrm{sim}}(s',\ \Delta t \mid s,\ a)}, \qquad w_{\mathrm{sim}} \;=\; \mathrm{clip}\!\left(\exp\!\Big(F_{\psi_{\mathrm{J}}} - G_{\psi_{\mathrm{M}}}\Big),\ 10^{-5},\ 1\right) $$

$$\mathcal{L}_{Q,\mathrm{mix}}(\phi_k) \;=\; \mathbb{E}_{(s,\,a,\,r,\,s',\,d)\sim\mathcal{D}_{\mathrm{off}}}\Big[\,\big(Q_{\phi_k}(s,\ a) - y_{\mathrm{mix}}\big)^2\,\Big] \;+\; \mathbb{E}_{(s,\,a,\,r,\,s',\,\Delta t,\,d)\sim\mathcal{D}_{\mathrm{sim}}}\Big[\, w_{\mathrm{sim}}\,\big(Q_{\phi_k}(s,\ a) - y_{\mathrm{mix}}\big)^2 \,\Big] $$

$$\mathcal{L}_{\pi,\mathrm{off\text{-}RESAC}}(\theta) \;=\; \mathbb{E}_{s \sim \mathcal{D}_{\mathrm{off}},\, \tilde{a} \sim \pi_{\theta}(\cdot\mid s)}\Big[\, \alpha \log \pi_{\theta}(\tilde{a} \mid s) - \Big(\, \bar{Q}(s,\ \tilde{a}) + \beta_{\mathrm{lcb}}\,\sigma_Q(s,\ \tilde{a}) \,\Big) \,\Big] + \beta_{\mathrm{bc}}\,\mathbb{E}_{(s,\,a)\sim\mathcal{D}_{\mathrm{off}},\, \tilde{a}\sim\pi_{\theta}(\cdot\mid s)}\Big[\,\big\|\tilde{a} - a\big\|_2^2\,\Big] $$

$$\mathrm{clip}_{60}(x) = \min\big\{60,\ \max(0,\ x)\big\} $$

$$a_{\mathrm{D}} = \mathrm{clip}_{60}\Big(\,0.6\,\big(h_f^{*} - h_f\big)\,\Big) $$

$$a_{\mathrm{A}} = \mathrm{clip}_{60}\Big(\,0.75\,\big(h_f^{*} - h_f\big) - 0.35\,\big(h_b^{*} - \hat{h}_b\big)\,\Big) $$

$$g_{M} = \frac{\bar{M}_{\mathrm{sim}} - \bar{M}_{\mathrm{tgt}}}{\bar{M}_{\mathrm{tgt}}} $$

$$\Delta M_{\iota}(\pi) = \bar{M}_{\iota}(\pi) - \bar{M}_{\iota}(\pi_0), \qquad e_{M}(\pi) = \frac{\Delta M_{\mathrm{sim}}(\pi) - \Delta M_{\mathrm{tgt}}(\pi)}{\bar{M}_{\mathrm{tgt}}(\pi_0)} $$

$$P_{\mathrm{sim}}\big(s',\ \Delta t,\ r \;\big|\; s,\ a\big) \;\neq\; P_{\mathrm{tgt}}\big(s',\ \Delta t,\ r \;\big|\; s,\ a\big) $$

$$J_{\mathrm{tgt}}(\pi) \;=\; \mathbb{E}_{\mathrm{tgt},\,\pi}\left[\, \sum_{t=0}^{\infty} \gamma^{t}\, r_t \,\right] $$

$$\mathcal{L}_{\mathrm{wm}} \;=\; \frac{1}{6}\Big(\, \mathcal{L}_{\mathrm{sys}} + \mathcal{L}_{\mathrm{veh}} + \mathcal{L}_{\mathrm{num}} + \mathcal{L}_{\mathrm{line}} + \mathcal{L}_{\mathrm{stop}} + \mathcal{L}_{\mathrm{nb}} \,\Big) $$

$$\hat{y}_{\mathrm{res}}(x) \;=\; \hat{y}_{\mathrm{sim}}(x) + \hat{\Delta}(x), \qquad \hat{y}_{\mathrm{dir}}(x) \;=\; \hat{f}_{\mathrm{dir}}(x) $$

$$c_i \;=\; \frac{1}{1 + e_i / \tau_e} $$

$$q_i \;=\; \log c_i \;+\; \gamma_{\Delta}\,(1 - d_i)\,Q^{\Delta}_{\bar{\xi}}\Big(s'_i,\ a'_{\mathrm{det}},\ \sigma_{\mathrm{wm}}\big(s'_i,\ a'_{\mathrm{det}}\big)\Big) $$

$$w^{\Delta}_i \;=\; \exp\!\Big(\, Q^{\Delta}_{\xi}\big(s_i,\ a_i,\ \sigma_{\mathrm{wm}}(s_i,\ a_i)\big) \,\Big) \;\in\; (0,\ 1] $$

$$q^{\mathrm{norm}}_i \;=\; (1 - \gamma_{\Delta})\,\log c_i \;+\; \gamma_{\Delta}\,(1 - d_i)\,Q^{\Delta}_{\bar{\xi}}\Big(s'_i,\ a'_{\mathrm{det}},\ \sigma_{\mathrm{wm}}\big(s'_i,\ a'_{\mathrm{det}}\big)\Big) $$

$$\mathcal{L}_{Q}(\phi_k) \;=\; 0.75\,\mathcal{L}_{\mathrm{env}}(\phi_k) \;+\; 0.25\,\mathcal{L}_{\mathrm{mod}}(\phi_k), \qquad \mathcal{L}_{\mathrm{mod}}(\phi_k) \;=\; \mathbb{E}_{i}\Big[\, w^{\Delta}_i\; \ell_{H}\big(Q_{\phi_k}(s_i,\ a_i) - \hat{y}_i\big) \,\Big] $$

$$z_i \;=\; \big[\, s_{i+1} - s_i\;;\; r_i \,\big] \in \mathbb{R}^{n_s + 1} $$

$$R_m \;=\; \left[\, \frac{1}{|\mathcal{I}_{\mathrm{cal}}|\,(n_s + 1)} \sum_{i \in \mathcal{I}_{\mathrm{cal}}} \sum_{k=1}^{n_s+1} \left(\, \frac{\hat{z}^{(m)}_{i,k} - z_{i,k}}{\varsigma_k} \,\right)^{2} \,\right]^{1/2} $$

$$m^{\star} \;=\; \begin{cases} \text{直接目标模型（Direct Model）}, & R_{\mathrm{dir}} < R_{\mathrm{res}}, \\[6pt] \text{残差世界模型（Residual Model）}, & R_{\mathrm{dir}} \geq R_{\mathrm{res}}. \end{cases} $$

$$w_i \;=\; w^{\Delta}_i\,\chi_i $$

$$\ell_H(\delta) \;=\; \begin{cases} \dfrac{1}{2}\,\delta^{2}, & |\delta| \leq 1, \\[8pt] |\delta| - \dfrac{1}{2}, & |\delta| > 1. \end{cases} $$

$$\tilde{P}\big(\cdot \;\big|\; s,\ a\big) \;=\; P_{\mathrm{sim}}\big(\cdot \;\big|\; s,\ a\big) + \Delta P\big(\cdot \;\big|\; s,\ a\big) $$

$$\big\|\, \mathcal{T}^{\pi}_{\tilde{P}} V - \mathcal{T}^{\pi}_{\tilde{P}} V' \,\big\|_{\infty} \;\leq\; \gamma\, \big\|\, V - V' \,\big\|_{\infty} $$

$$\big\|\, V^{\pi}_{\mathrm{tgt}} - V^{\pi}_{\mathrm{mdl}} \,\big\|_{\infty} \;\leq\; \frac{\varepsilon_r}{1 - \gamma} \;+\; \frac{2\,\gamma\, R_{\max}\, \varepsilon_P}{(1 - \gamma)^{2}} $$

$$J_{\mathrm{tgt}}\big(\pi^{\star}\big) - J_{\mathrm{tgt}}\big(\hat{\pi}^{\star}\big) \;=\; \Big[\,J_{\mathrm{tgt}}\big(\pi^{\star}\big) - J_{\mathrm{mdl}}\big(\pi^{\star}\big)\,\Big] + \Big[\,J_{\mathrm{mdl}}\big(\pi^{\star}\big) - J_{\mathrm{mdl}}\big(\hat{\pi}^{\star}\big)\,\Big] + \Big[\,J_{\mathrm{mdl}}\big(\hat{\pi}^{\star}\big) - J_{\mathrm{tgt}}\big(\hat{\pi}^{\star}\big)\,\Big] \;\leq\; 2\,\bar{\varepsilon} $$

$$w^{\star}_{\mathrm{orig}} \;=\; \exp\!\big(Q^{\Delta}_{\mathrm{orig}}\big) \;=\; c^{\,\frac{1}{1 - \gamma_{\Delta}}}, \qquad w^{\star}_{\mathrm{norm}} \;=\; \exp\!\big(Q^{\Delta}_{\mathrm{norm}}\big) \;=\; c $$