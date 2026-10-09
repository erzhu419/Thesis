# Soft-return uncertainty reference, 2026-09-13

This is an executable algorithm-development reference, outside the manuscript.
Implementation: `common/return_uncertainty.py`; experiment: `controlled_return_uncertainty.py`.

## Object and objective

Fix a policy pi, entropy temperature alpha, discount gamma, horizon H, and
initial state/action. A complete Markov decision process (MDP) hypothesis M
is drawn once from its posterior given dataset D and held fixed throughout
the rollout. The random soft return is

`G = sum(t=0..H-1) gamma^t r_t - alpha sum(t=1..H-1) gamma^t log pi(a_t|s_t)`.

Termination cuts both sums short. The prescribed first action has no entropy
term. Our primary training target is its expectation, as in soft actor-critic
(SAC); task performance is evaluated using ordinary episodic reward. Variance
penalties define additional risk preferences, not corrections required by SAC.

Let m_M be the conditional mean of G. A_M and U_M are environmental and policy
variance contributions, defined by the following Bellman-local allocation.
For a joint reward/transition outcome o with successor s', set
`c_o = r_o + gamma v_M(s')`. Terminal continuation terms are zero. Then

```
m_M = E_o[c_o]
A_M = Var_o(c_o) + gamma^2 E_o[A_M^V(s')]
U_M = gamma^2 E_o[U_M^V(s')]
v_M = E_a[m_M - alpha log pi(a|s)]
A_M^V = E_a[A_M]
U_M^V = E_a[U_M] + Var_a(m_M - alpha log pi(a|s))
```

Superscript V denotes continuation before action sampling; all quantities
start at zero at horizon zero. The policy and environmental components sum
to Var(G|M). This is an ordered allocation, not a unique causal attribution.
The posterior predictive variance is exactly `A + U + E`, where
`A = E_M[A_M]`, `U = E_M[U_M]`, and `E = Var_M(m_M)`.
The three sources are separate; policy randomness is not labeled environment
noise. A known stochastic model has E=0, not necessarily A=0.

## Estimation and decisions

The finite-data reference knows categorical reward support but not its
probabilities. With counts n_j and Dirichlet prior concentration 0.5 per
outcome, let b_j=n_j+0.5, B=sum_j b_j, p_j=b_j/B, and mu=sum_j p_j r_j.
Writing V=sum_j p_j(r_j-mu)^2, the exact single-step posterior moments are
`A = B V/(B+1)` and `E = V/(B+1)`. Under increasing independent coverage,
the posterior concentrates: E approaches zero and A approaches true noise.
Higher noise can legitimately increase E at fixed coverage.

Decision controls use `f = mu - lambda_A sqrt(A) - lambda_E sqrt(E)`.
They comprise no penalty, either single penalty, both penalties, and a shared
action-independent shift of the same average magnitude. A shared shift
preserves the action ranking. Primary expected-return decisions use both
coefficients zero. Secondary controls fix lambda_A=0.1 and lambda_E=1;
the latter is a posterior-risk heuristic, not a calibrated confidence level.

For nonnegative variance estimates and uniform errors e_mu, e_A, e_E over
candidate actions, `|f_hat-f| <= e_mu + lambda_A sqrt(e_A) + lambda_E sqrt(e_E)`.
This follows from `|sqrt(x)-sqrt(y)| <= sqrt(|x-y|)`. Maximizing f_hat has
regret at most twice this bound **for the same f objective**. This statement
does not convert risk-score guarantees into expected-reward guarantees.

## Norms and ambiguity

For a feed-forward network with 1-Lipschitz activations and layer kernels W_l,
`Lip(network) <= product_l ||W_l||_2`. Composition proves the product bound;
biases do not affect it. Two scalar layers of weight 3 have Lipschitz constant
9 but a layerwise L1 sum of 6. Rescaling their weights to 30 and 0.3 preserves
the function and product while changing the L1 sum to 30.3. Parameterization
dependence alone does not invalidate an upper bound; L1 magnitude is simply
not an estimate of environmental noise.

An integral probability metric (IPM) ball around an estimated transition law
bounds expectation error only for its specified test-function class and
radius. With a Lipschitz test-function class, its value bound uses the
Lipschitz constant above. Uncertainty about the transition law is distinct
from randomness under a known law. Adding penalties for two descriptions of
the same ambiguous law is not justified by the variance identity; independent
ambiguity components require an explicitly specified joint uncertainty set.
This experiment uses a categorical posterior, not the manuscript's IPM model.

## Executed controls

Default seed 20260913; 100 independent datasets per condition. Noise standard
deviations are 0, 1, 5, 10 at fixed mean zero and support [-10,0,10]; coverage
is 4, 16, 64, 256, 1024 with nested prefixes within each dataset. Decision
controls cross both signs of a 0.2 mean advantage, noise levels 2/5, and
coverage 16/1024. All five decision rules see identical data.

- At true noise variance 25, increasing coverage 4 to 1024 reduces mean E
  from 4.99936 to 0.02451; mean A changes from 27.4965 to 25.1366.
- For a known model at noise variances 0, 1, 25, 100, E is exactly zero.
- In a two-model deterministic chain, modelwise backups give root (A,E)=(0,0.81).
  Sharing a mixture target gives (0.81,0); sharing only its mean gives (0,0).
  On a revisited state, mixture sharing also changes total variance from
  3.0625 to 1.3125 by losing persistent model dependence.
- With independent unbiased Gaussian Q errors of standard deviation 2,
  twin-min bias is -1.13720 (Monte Carlo standard error 0.00738); the exact
  value is -2/sqrt(pi)=-1.12838. Two-head averaging gives -0.00078.
  These are estimator-error controls, not a derivation from reward noise.
- When the better-mean action is noisier (advantage 0.2, noise 5, coverage
  1024), mean-only expected-return regret is 0.034, versus 0.150 for dual
  penalties. Their secondary mean-standard-deviation regrets are 0.083 and
  0.025 respectively. The apparent winner changes with the stated objective.

Outputs: `../diagnostics/uncertainty_control_20260913/summary.json` plus four
small CSV files (about 404 KB total). From the repository root:

```
python -m jax_experiments.experiments.controlled_return_uncertainty --output /tmp/resac-uncertainty-new
python -m unittest discover -s jax_experiments/tests -p '*return_uncertainty.py' -v
```

## Prior work and remaining work

[Depeweg et al. (2018)](https://proceedings.mlr.press/v80/depeweg18a.html)
already separate uncertainty sources for risk-sensitive decisions.
[Clements et al.](https://arxiv.org/html/1905.09638v3) study Bayesian
distributional critics and use shared averaged-quantile targets in Section
3.4; our mixture ablation is not their algorithm.
[Distributional reinforcement learning with epistemic and aleatoric uncertainty estimation (2023)](https://doi.org/10.1016/j.ins.2023.119217)
already combines distributional SAC with uncertainty estimation.
[Epistemic Bellman Operators (2025)](https://ojs.aaai.org/index.php/AAAI/article/view/35393)
addresses target-parameter uncertainty through Bellman operators.
Neither decomposition nor preserving a model identity is claimed novel here.

The reference has known finite support, fixed policies, and an exact Bayesian
model. Finite-data estimates can be prior-dominated, especially with rare
noise outcomes. Neural heads are not established posterior samples; shared
target mixing is only one failure mechanism. The five decision rules are
not five fully trained SAC ablations, and a correct decomposition alone does
not establish better control. The IPM assumptions and original manuscript
inequality have not been repaired in the manuscript, which remains untouched.

## Neural follow-up: completed

`neural_return_uncertainty.py` fits neural moments using Bellman targets from
data-inferred categorical models. True return moments are used for evaluation,
not training labels. Two root actions lead to two terminal reward states;
the continuation policy is deterministic, so its policy-variance term is zero.
The model probabilities come from observed counts and the stated Dirichlet
posterior. Each critic member is assigned one complete model, not interpreted
as a posterior sample merely because its parameters were initialized randomly.

Seeds 8/16/24/32/40 cross candidate reward standard deviations 2/5, counts
16/1024, and mean advantages -0.2/+0.2. The reference action has mean zero,
standard deviation 2, and 256 observations. Counts are nested across coverage;
all four modes share the data, 16 assigned models, and network initialization.
There are 40 conditions and 160 fits, not 160 independent data seeds.

The critic reuses `EnsembleCritic`: separate mean/variance trunks, each with
two width-32 hidden layers, and a softplus variance output. Training minimizes
mean squared mean error plus squared log1p-variance error. Every fit uses 3000
full-model expectation updates, Adam learning rate 0.003, target averaging
rate 0.05, and gamma=0.9. No variance-loss gradients enter the mean trunk.
The four target modes are modelwise, shared mixture, shared mean only, and
shared first-two-member minimum with the selected member's variance.

All 160 fits finished without nonfinite outputs. Across fits, the largest
member-mean root mean squared error (RMSE) against the mode-specific fixed
point was 0.003209; the largest member-variance RMSE was 0.008995.

- Modelwise root-mean mean absolute error (MAE) against the same assigned-model
  reference was 0.000145; environment/epistemic variance MAEs were
  0.000324/0.000096. Greedy actions matched this reference in 40/40 conditions.
- Shared mixture and shared mean-only both had root-mean MAE 0.000216 and
  matched all 40 reference actions. Both collapsed root epistemic variance;
  mixture sharing additionally moved it into environmental variance.
- At noise standard deviation 5 and count 16, modelwise root environmental
  and epistemic variances averaged 20.1666 and 1.12308. Mixture sharing gave
  21.2896 and 0.00000209. With count 1024, modelwise epistemic variance fell
  to 0.016394 while environmental variance remained 20.3767 (truth: 20.25).
- The shared twin-min control had root-mean MAE 0.237147 and changed the
  reference action in 10/40 conditions. Observed true-environment expected
  return regret averaged 0.054, versus 0.036 for the other three modes.
  Reference agreement is not agreement with the unknown true optimal action.

Results: `../diagnostics/neural_uncertainty_20260913/` contains configuration,
all learning curves, per-query estimates, summary, and compressed model and
prediction arrays, totaling about 0.8 MB. No neural checkpoints were needed
for this approximately 100-second CPU experiment. Reproduction:

```
JAX_PLATFORMS=cpu OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
  python -m jax_experiments.experiments.neural_return_uncertainty --output /tmp/resac-neural-new
```

The small fit errors refer to the finite assigned-model reference, not the
unknown true Q. Data error and finite-model sampling error remain; analytic
posterior moments are saved separately. This is a finite-state, known-support,
model-based fitting experiment with exact outcome expectations. It is neither
a stochastic replay-trained SAC comparison nor evidence of bus performance.
The twin-min control still trains 16 members and is not a full SAC baseline.

## Replay follow-up: completed 2026-09-14

`replay_return_uncertainty.py` uses the same observed datasets, without
Dirichlet pseudo-observations or outcome-probability inputs to the learner.
Each of 16 members retains a fixed ordinary bootstrap resample of the replay.
Updates draw 32 independent pairs per state-action query from that member's
replay, using common random indices across members and variance modes.
All 120 fits use 6000 updates, identical initial mean/variance networks, and
cosine learning-rate decay from 0.003 to 0.00003. These are 5 data seeds and
40 conditions, each evaluated with 3 variance objectives.

For paired Bellman mean samples Y_1,Y_2 at the same query, with continuation
variance contributions C_1,C_2, the paired variance target is
`U_pair = (Y_1-Y_2)^2/2 + (C_1+C_2)/2`.
Its expectation is the conditional variance for the fixed continuation
predictors. In contrast, centering squared residuals on the current mean mu
adds `(E[Y]-mu)^2`. A true variance of 25 becomes a residual target of 50
when the mean error is 5; the paired target remains 25. Pairing different
state-action queries is invalid. Continuation approximation errors remain.

The raw-variance regression loss is `E[(v-U)^2]/100`; the divisor is a fixed
numerical scale. Both it and mean regression retain arithmetic expectations.
Directly reusing squared log1p regression on *random* U instead learns
`v = exp(E[log(1+U)])-1`, not E[U]. The `paired_log_target` negative control
therefore deliberately has a different population optimum. The previous
exact-moment loss remains valid for its original deterministic moment labels.

Results in `../diagnostics/replay_uncertainty_20260913/` (about 0.82 MB):
- All 120 fits completed without nonfinite predictions; 24 focused tests pass.
- Root environmental-variance MAE against the finite bootstrap reference is
  0.04830 for squared residuals, 0.04064 for paired targets, and 5.75746 for
  the log-target control. These are descriptive averages, not a significance test.
- At reward standard deviation 5 and coverage 1024, root variance estimates
  are 20.1566, 20.1974, and 3.3659 respectively. The empirical bootstrap
  reference is 20.2166 and the true environment value is 20.25.
- All three modes have identical saved mean predictions, root-mean MAE
  0.005809 against the empirical reference, and the same actions. This
  experiment demonstrates variance-learning behavior, not a control improvement.
- Four count-16 conditions observe no nonzero candidate rewards despite true
  reward variance 4. Their bootstrap variance/disagreement references are zero;
  paired neural root variance is about 0.002, versus true root variance 3.24.
  This is an observed limitation of empirical resampling as an ignorance signal.

Reproduction uses the same module/CLI pattern as above, with
`python -m jax_experiments.experiments.replay_return_uncertainty --output /tmp/resac-replay-new`.
The saved bundles contain raw bootstrap replay and predictions, and the
learning CSV contains all 3720 logged snapshots. Source datasets still match
all 40 conditions from the previous neural experiment.

Remaining: empirical bootstrap disagreement is not a calibrated epistemic
posterior, and exact repeated state-action pairs are unavailable in general
continuous-state SAC replay. Missing rare events cannot be recovered from
resampling alone; their uncertainty needs an explicit, defensible model or
prior. A subsequent estimator must address these two limitations and policy
sampling before it enters SAC. Keep noise estimation separate from Q
pessimism; do not add variance penalties on the strength of these fitting
results. Formal comparisons still require matched compute, B0 and individual
channel/constant-shift controls, matched noise laws, and a shared bus/JAX
target contract. Preserve negative results and do not tune noise to defeat SAC.

## Continuous-query calibration follow-up: 2026-09-14

Protocol fixed before execution: `continuous_return_uncertainty.py` fits a
fixed-feature Bayesian linear return critic to independent, two-step episodes.
Each continuous initial `(x,a)` is unique. Transitions and continuation policy
are deterministic; terminal reward noise is stochastic. No paired outcomes,
true means, true variances, or noise samples enter fitting. The complete return
is observed, so this isolates estimation without learned Bellman targets.

Features are `(1,x,a)`. A normal-inverse-gamma prior has precision `0.01 I`,
zero coefficient mean, shape 2, and scale 1. The conditional Gaussian noise
model is an explicit assumption. Posterior noise expectation is the
environmental channel; posterior variance of the conditional mean is the
epistemic channel. Noise-parameter intervals are separate from the latter.
Implementation reference: [Ekstrom/Forssen, conjugate linear regression,
Sections 25.2-25.3](https://cforssen.gitlab.io/tif285-book/content/BayesianStatistics/BayesianLinearRegression/BayesianLinearRegression_ConjugatePrior.html).
This is a standard calibration baseline, not a new inference contribution.

Use 100 independent dataset seeds, counts 16/256, training ranges
`x in [-0.2,0.2]`/`[-1,1]`, and actions in `[-1,1]`. Counts share nested
observations; models share identical data. Query states are `-0.1,0.1,-0.8,0.8`
with prescribed action zero. Compare a shared noise model with two independent
models split at `x=0`; this observable regime boundary is supplied, not learned.
Five scenarios: Gaussian terminal standard deviation 1/5, sign-dependent 1/5,
rare rewards `-10,0,10` with probabilities `.02,.96,.02`, and a nonlinear
mean term `2*x^2` omitted by both linear feature models. This is 4000 fits,
not 4000 independent seeds. Report 90% mean/noise/predictive interval coverage
and descriptive standard errors across datasets, not individual query points.

Additional controls refit the same physical replay versus falsely counting
32 reuses as new data, and re-read the four actual previous all-zero datasets.
For the latter, retain all Gaussian prior scales `0.1,1,10` alongside the
known-support Dirichlet reference. Nonzero uncertainty alone is not calibration.
The main prior is not selected using these results. No actor, risk penalty,
formal training change, checkpoint, or manuscript update is part of this run.

Completed: 4000 model fits, 100 dataset seeds, 7 new tests; artifacts in
`../diagnostics/continuous_uncertainty_20260914/` occupy 807798 bytes.
Each sign-split model fit contains two regressions. Reproduce with
`python -m jax_experiments.experiments.continuous_return_uncertainty --output /tmp/resac-continuous-new`.
At full coverage and count 256, sign conditioning estimates environmental
variances 0.791/19.166 (truth 0.81/20.25); variance MAE falls from 9.720 to
1.079, but mean MAE rises from 0.199 to 0.304. For shared high-noise regression,
count 16 to 256 reduces near-query epistemic variance from 1.091 to 0.080;
environmental variance remains 19.642 at 256 (truth 20.25). Refit reuse leaves
epistemic variance 1.902 unchanged; falsely counting 32 reuses reduces it to 0.066.

Limitations: these are fixed-feature, complete-return, fixed-policy results.
At count 16, 54/100 rare-noise datasets see no nonzero noise; shared-model
variance is 0.112 versus truth 3.24, with zero noise-interval coverage in that
subset. All three prior scales also miss the true noise variance in all four
previous blind spots. At count 256/full coverage, nonlinear shared-model mean
interval coverage is 0%, despite predictive coverage 88.75%. Even the correctly
specified sign-noise model has noise coverage 84%, below nominal 90%. Neither
nonzero priors nor aggregate predictive coverage certify correct decomposition.
Next: conditional non-Gaussian noise modeling with independent calibration,
then test learned Bellman targets against this independent-return reference.

## Non-Gaussian cross-fit follow-up: 2026-09-14

`crossfit_return_uncertainty.py` estimates conditional moments without a
Gaussian likelihood. Two independent ordinary-least-squares mean fits use
separate episode sets. A third set trains the sign-conditioned noise moments:
`U_cross = (Y-m1)*(Y-m2) = (Y-(m1+m2)/2)^2 - (m1-m2)^2/4`.
Given fixed predictors with errors `e1,e2`, its expectation is `A+e1*e2`.
Averaging over independent unbiased mean fits removes that product; Gaussian
noise is not required. Correlated errors add their covariance and common bias
remains. The disagreement term estimates the sampling variance of the averaged
mean under these assumptions, not a Bayesian epistemic posterior. All signed
estimates are retained, including negative values. Noise losses never alter Q.

Protocol fixed before execution: 100 seeds, 16/256 episodes per fitting role
(48/768 total), 128 separate calibration episodes, and 1024 test episodes per
domain. Calibration and same-distribution testing are iid within prespecified
sign groups. An additional shifted test uses `abs(x) in [1,2]`, versus `[0,1]`
for fitting/calibration. Degree-1 and degree-2 mean features are retained as
controls, including `x^2` only in the latter. Both use identical datasets and
compare training residuals, held-out squared residuals, and cross-products.
Reuse the five earlier noise/mean scenarios, plus zero-mean skewed Bernoulli
noise (event probability .04, variance 4) and scaled Student-t(5) noise
(variance 25). This yields 2800 model fits, not a matched-budget comparison
with the previous 4000 fits. The base RNG seed is 20260915, not a run date.

Groupwise split-conformal absolute-residual intervals use the
`ceil((n_cal+1)*0.9)`th order statistic. Calibration modifies interval radii
only; evaluation records future-return coverage and rare-event coverage
separately. These are not confidence intervals for Q or the noise variance.
The coverage argument requires exchangeability within groups and does not
apply to shifted tests. Reference: [Angelopoulos and Bates, Sections 1.1 and
4.1](https://arxiv.org/html/2107.07511v6). Cross-fit identities here are algebraic
controls, not a claim of a novel inference method. No policy optimization,
learned Bellman targets, formal training change, or paper edit is included.

Completed: 2800 unique model fits, 9 new tests, 1478830 artifact bytes in
`../diagnostics/crossfit_uncertainty_20260914/`. Reproduce with
`python -m jax_experiments.experiments.crossfit_return_uncertainty --output /tmp/resac-crossfit-new`.
The exact independent-error control changes residual noise 37.5 to 25 (truth
25); shared errors or common bias still produce 50. For degree 2/high Gaussian
noise/count 256, estimated mean sampling variance is 0.10948 versus repeated-fit
variance 0.10724. Mean-Q bias is -0.00705; taking the same two heads' minimum
gives -0.26761. Same-distribution return coverage across all groups is
90.01%-91.97%; rare-event coverage for degree 2 is only 0.71%-1.25%.

Limitations: cross-products do not uniformly improve finite-sample error.
At high Gaussian noise/count 16, noise bias changes from +2.683 to -0.223,
but MAE increases from 8.975 to 9.608. Count-16 rare/skew-rare fitting sets
miss every tail event for 20/15 seeds. Calibration sees tails in all these
cases, yet moment estimates remain near zero versus true variance 3.24;
the skew-rare subset also has mean MSE 0.135. Calibration of 90% return
coverage does not certify tail coverage or either moment channel. Negative
estimates and shifted-test failures are retained. Next: an independent small
Bellman-learning experiment, with complete-return references, to measure
target-induced error correlation and minimum-induced Q underestimation.

## Fitted Bellman follow-up: 2026-09-14

`bellman_return_uncertainty.py` performs finite-horizon least-squares policy
evaluation. The continuous environment has mean reward `.2+.5*s-.2*a`,
transition `s'=.8*s+.2*a+xi`, independent `xi~Normal(0,.15^2)`, and fixed
policy `a'=-.5*s'`. Its true Q is linear; return variance is state-independent
and exactly computable. Each head retains its own independent transition
fold at every horizon. A third fold estimates environmental moments. Fitting
receives observed one-step rewards/successors, never true moments or full returns.

Protocol fixed before execution: 100 data seeds, 16/256 transitions per fold
per stage, discount .9, and horizons 1/8/32. Each 32-stage fit yields three
snapshots. Four reward-noise laws are Gaussian variance 1/25, zero-mean skewed
Bernoulli variance 4, and Student-t(5) variance 25. Three targets preserve head
identity, share the head mean, or share the head minimum. Root evaluation
always averages both heads. This is 2400 full fits and 7200 snapshots; fitting
cost at horizon h is `3*n*h` transitions. All modes share observed data and
exact regression solves, with no optimization-convergence confound.

For `d_k=r+gamma*v_k(s')-q_k(s,a)`, the matched local noise target is `d_1*d_2`,
with `gamma^2*A_next` propagated separately. Retain two controls: squared
mean residual, and a single-mean-target product that subtracts current-Q
disagreement instead of residual disagreement. All variance estimates remain
signed and have no effect on mean regression. Independent complete trajectories
(1024 per query) validate true return moments and diagnose full-return residual
products. Across-seed error covariance is separate from within-fit disagreement:
`Var((q1+q2)/2) - Var((q1-q2)/2) = Cov(q1,q2)`, using centered sample variances.

The exact intercept-only enumeration isolates sharing/minimum effects without
feature approximation. In the continuous experiment the minimum target can
be nonlinear and its variance state-dependent, unlike the true linear-policy
Q; this additional projection effect must remain distinct from shared-error
covariance. The experiment has no actor, entropy term, or performance claim
for full SAC. No paper or formal training changes are included.

Completed: 2400 Bellman fits, 7200 unique snapshots, 9 new tests; artifacts in
`../diagnostics/bellman_uncertainty_20260914/` occupy 3144827 bytes. Reproduce
with `python -m jax_experiments.experiments.bellman_return_uncertainty --output /tmp/resac-bellman-new`.
For Gaussian reward variance 25 and n=16, shared-min Q bias grows from -0.0137
at horizon 1 to -5.0509 at 8 and -8.9145 at 32. At horizon 32, independent
and shared-mean bias are -0.0082 and +0.0029. Shared-mean within-fit
disagreement is 2.3194, versus mean sampling variance 7.5329; head-error
covariance is 5.2019. The centered covariance identity holds numerically.

For independent targets in the same n=16/horizon-32 condition, environmental
variance is 132.9523 (SE 1.7305), versus truth 131.6753. Subtracting current-Q
disagreement instead of matched-residual disagreement gives 106.2364; the
paired difference is 26.7159 (SE 2.7757). Shared-min matched Bellman moments
are 134.6231, while complete-return residual products average 218.2394: a
nearly correct local noise estimate can coexist with badly biased Q.

Limitations: across-seed covariance is a diagnostic, not an estimator available
from one trained pair. Independent-head disagreement also has finite-seed
sampling error; e.g. at Gaussian variance 25/n=256/horizon 32 it is 0.5040
versus empirical sampling variance 0.3600. Complete-return Monte Carlo checks
are stored separately with their sampling errors; analytic moments define the
truth comparisons. This fixed-policy experiment does not test actor-induced
overestimation, and does not justify removing the SAC minimum universally.
Next: neural-critic Bellman smoke comparisons with matched residuals and the
same three target modes, before any formal high-noise SAC/bus experiment.

## Neural Bellman smoke: protocol 2026-09-14

`neural_bellman_uncertainty.py` replaces the two least-squares Q heads with
the existing two-layer `EnsembleCritic` (width 32, two independent heads).
Keep data seeds 8/16/24/32/40, counts 16/256 per stage/fold, horizons 1/8/32,
and Gaussian variance-25 / skew-rare variance-4 rewards from the previous
experiment. Preserve independent, shared-mean, and shared-min targets.
These 60 fits produce 180 snapshots, not 60 independent data seeds.

Protocol fixed before execution: 1000 full-batch Adam updates per stage,
cosine rate .001 to .0001, logged every 250 updates. Warm-start weights from
the previous horizon but reset the optimizer and schedule. Terminal targets
have zero continuation; later targets use the frozen previous-horizon network.
Initialization seed 20260914 is fixed across data/mode comparisons, with two
distinct independently initialized heads. No output normalization, clipping,
early stopping, or held-out checkpoint selection is used.

The third independent fold estimates the same three scalar Bellman variance
recursions as before. These are not neural variance heads. It also compares
network error against the current frozen labels with linear regression on
the identical labels; lower training error is not proof of good generalization.
Completed linear-chain results are read from existing snapshots, not rerun.
Full-return Monte Carlo samples and analytic moments are evaluation-only.
Across-seed covariance here conditions on fixed initialization and has only
five data replicates. This tests neural function approximation in fixed-policy
evaluation, not actor learning or whole-SAC performance. Formal training and
paper sources remain unchanged.

Completed: 60 fits, 180 unique snapshots, 9600 training-log rows, 5 new tests,
and no nonfinite snapshots. Artifacts occupy 879945 bytes in
`../diagnostics/neural_bellman_uncertainty_20260914/`; runtime was 118.61 s on
two local physical CPU cores. Reproduce with
`python -m jax_experiments.experiments.neural_bellman_uncertainty --output /tmp/resac-neural-bellman-new`.
At Gaussian noise/n=256, shared-min Q bias is -0.1685/-5.0446/-13.7107 at
horizons 1/8/32. Shared-mean at horizon 32 has bias -0.1840 but Q MSE 8.6467,
versus 0.4389 for the same five completed linear-reference seeds. Its
within-fit disagreement is 2.2410 versus across-data mean variance 10.0482.

Limitations: shared-mean's same-target training MSE is 17.8900 versus linear
29.9414, but held-out MSE is 36.1697 versus linear 29.3704. The flexible
network therefore fits training labels better without generalizing as well.
Final-quarter loss drops remain positive; fixed-budget fits are not claimed
converged. Three matched-variance snapshots are negative and remain unaltered.
Five data seeds and fixed initialization do not establish calibrated uncertainty
or an actor-level improvement. Next: frozen-feature versus trainable-feature
critic controls to separate representation overfitting from target bias before
modifying the formal SAC algorithm. Do not infer that replacing min with mean
alone fixes high-noise learning.

## Frozen-feature follow-up: protocol 2026-09-14

`frozen_feature_bellman.py` uses the same five data seeds, two noise laws,
counts, initialization, architecture, and 1000-step Adam budget as the neural
smoke. Freeze both hidden layers at their random initialization and train only
the output layer. Full-network recursive results are read from the completed
baseline; no paper or formal training changes are made.

Two distinct comparisons are fixed before execution. First, reconstruct the
linear teacher coefficients to supply identical one-step Bellman labels to
trainable-feature and frozen-feature networks, both reset to identical initial
weights for each label set. Compute a minimum-norm least-squares output layer
on those same frozen features as an optimization/generalization diagnostic.
It is not bootstrapped and does not supply labels. No regularization or
held-out selection is introduced. Terminal labels are mode-invariant and are
fitted only once per seed/noise/count, giving 140 label sets, 280 Adam fits,
and 140 exact readout fits. Second, run 60 frozen-feature recursive chains
with warm-started output layers, producing 180 snapshots. Their future labels
evolve with their own networks and are not identical to the trainable chains.

Use paired held-out MSE differences across the five data seeds for the
fixed-label comparison, retaining all readout results and remaining training
loss changes. Freezing features changes both representation capacity and
optimization; comparing its Adam output with the exact readout distinguishes
incomplete fitting from architectural restriction. The third fold supplies
diagnostics and scalar noise estimates, not model selection. Initialization
is fixed, so these runs do not assess sensitivity to representation seed.

Completed: 60 frozen chains, 420 fixed-label fits, 180 chain snapshots, and
9600 learning rows; 11 tests pass (6 new, 5 original). Artifacts occupy
1026607 bytes in `../diagnostics/frozen_feature_uncertainty_20260914/`.
The run took 144.39 s. Reproduce with
`python -m jax_experiments.experiments.frozen_feature_bellman --output /tmp/resac-frozen-feature-new`.
Trainable terminal-label MSE exactly matches all 20 corresponding existing
baseline cases, verifying unchanged behavior on those repeated labels.

At Gaussian noise/n=256/horizon 32/shared-mean, frozen-chain Q MSE is 0.3413
versus trainable 8.6467, with bias -0.1072 versus -0.1840. Shared-min bias is
-1.8929 versus -13.7107. However, on identical linear-teacher labels at the
same horizon, held-out MSE is 26.5579 frozen versus 26.1948 trainable; the
paired difference (trainable minus frozen) is -0.3631 with SE 0.6804. The
recursive gain is not evidence of universally better single-stage fitting.

Limitations: at n=16/shared-mean/horizon 32, frozen-chain Q bias remains
-2.4215. On those fixed-label problems, unregularized exact frozen readouts
have held-out MSE 98431.97 despite almost zero training error, versus 32.89
for the fixed-budget frozen Adam fit; all outliers are retained. Freezing
alone does not prevent noise interpolation, and finite-budget optimization
contributes implicit regularization. These remain fixed-initialization,
linear-true-Q, fixed-policy diagnostics. Next: select explicit readout
regularization on separate development seeds, then evaluate untouched seeds
for Q error and disagreement calibration before changing formal SAC.

## Explicit readout regularization: protocol 2026-09-14

`regularized_feature_bellman.py` keeps the frozen 2x32 ReLU features and
initialization 20260914. Fit each output head exactly with objective mean
squared training error plus lambda times squared non-bias weights. The
intercept is unpenalized; features are not normalized. Float32 features are
converted to float64 for a centered ridge solve. Only folds 0/1 train Q;
fold 2 retains signed scalar variance diagnostics and never updates Q.

Before execution, fix candidate lambdas .0001/.001/.01/.1/1, development
seeds 1000--1009, and untouched test seeds 2000--2019. Select one global
lambda using independent complete-return prediction MSE on development
seeds, equally weighted over both noise laws, both counts, all three target
modes, horizons 1/8/32, and the four prescribed queries (1024 trajectories
per query). No analytic Q or variance enters selection. Save selection.json
before generating any test data. This uses extra simulator returns for
development, not a purported tuning procedure for arbitrary offline RL.

Run the selected ridge, original frozen-feature 1000-step Adam, and linear
least-squares Bellman chains on the same test transition folds and return
samples. Expected: 600 development chains/1800 scores and 720 test
chains/2160 snapshots. Compare paired Q error and bias, matched variance,
head-error covariance, and disagreement versus across-data mean variance.
Do not tune again after inspecting test results. No paper, actor, or formal
SAC changes; no remote dispatch or checkpoint collection. These remain
fixed-policy, linear-true-Q, fixed-initialization controls, not a new algorithm
or an uncertainty-calibration guarantee.

Completed in `../diagnostics/regularized_feature_uncertainty_20260914/`:
600 development chains, 720 test chains, 2160 test snapshots, 38400 learning
rows; 17 tests pass. All expected artifact keys are present exactly once and
all numbers are finite (4118996 bytes, 526.80 s). Development selected lambda
1, the grid endpoint; this is not a claim of globally optimal regularization.
At Gaussian/n=256/h=32/shared-mean, ridge Q MSE is 1.0964 versus Adam 0.7552
(paired increase 0.3412, SE 0.1339), despite bias improving -0.4882 to -0.2043.
The range of across-seed mean predictions at the four queries is only 7.75%
of the true Q range. Disagreement is 0.0523 versus sampling variance 0.2289;
matched noise is 132.0121 versus true 131.6753. At n=16/shared-min, bias
worsens from -3.1943 to -6.9239. Retain all 36 negative variance snapshots
from the Adam/linear sparse skew-rare terminal controls. This selected ridge
variant is not adopted. Next: held-out action-ranking/regret comparisons,
including a state-only common-shift control, before actor-learning runs.

## Root-action follow-up: protocol 2026-09-14

`action_selection_bellman.py` fixes new seeds 3000--3009, Gaussian reward
variance 1/25 and skew-rare variance 4, counts 16/256, horizons 1/8/32, and
the three target modes. Compare frozen Adam (unchanged 1000-step budget),
the rejected but locked ridge lambda=1 control, and linear least squares.
Use 17 states in [-.9,.9] and 81 root actions in [-1,1], with both mean and
min readouts. Evaluate greedy actions and a finite-grid Gibbs policy with
fixed temperature .1. Exact Q measures expected regret/gain of changing
only the root action; the continuation policy remains a=-.5*s. This is not
actor learning or repeated policy improvement. Initialization stays fixed.

After fitting, compare raw scores, a state-only shift -10*(1+.25*s), and
an oracle state-wise centering correction. The latter uses analytic Q only
for diagnosis, not training or a deployable method. Check action/probability
invariance while retaining Q bias, centered action-value error, ranking,
and entropy-regularized regret. No new hyperparameter selection. Expected
540 fitted chains and 9720 evaluation rows, with only 10 independent data
seeds; offsets/readouts are paired views, not additional trained models.
No checkpoints, server jobs, paper changes, or formal SAC edits.

Completed: 540 chains, 9720 unique decision rows, 28800 learning rows,
23 passing tests, all finite; 6368735 bytes in
`../diagnostics/action_selection_uncertainty_20260914/`, runtime 754.34 s.
Both common-offset controls change zero greedy actions; maximum probability
change is 5.50e-15 and oracle-centered absolute mean bias is below 4.85e-16.
At Gaussian-high/n=256/h=32 with frozen Adam and mean readout, shared-mean
versus shared-min targets improve Q bias -1.7106 to -0.3701, but greedy
regret changes 0.07354 to 0.07207 (paired difference -0.00147, SE 0.00504).
The locked ridge at that shared-mean setting lowers entropy-objective regret
0.10241 to 0.02985 but raises ordinary soft-policy return regret 0.07619 to
0.08988. Do not conflate entropy benefits with task-return improvements.
Next: small online actor-critic controls, separating training-time shared
shifts from action-dependent error. Post-fit offset invariance does not test
training dynamics; no formal algorithm replacement is supported by this run.

## Online actor-critic pilot: protocol 2026-09-14

`online_bias_control.py` reuses SACBase's Gaussian policy, three-layer twin
critic, optimizers, actor-mean readout, target averaging, and replay buffer.
An experiment-only subclass changes the critic target to min, mean, or
min with training reward +1; the zero-offset min is tested against SACBase.
Fresh seeds 4000--4004 cross reward-noise standard deviations 1/5 and the
three settings: 30 online runs, not additional frozen-policy fits.

The observable state includes remaining time. Episodes have exactly 32
steps, s_next=.8*s+.5*a+.15*xi and r=1-s^2-.1*a^2+sigma*eta, with independent
standard-normal shocks and initial state uniform on [-1,1]. Fixed gamma=.9,
alpha=.1, width=32, Adam rate .0003, tau=.005, 16 environments, 1024 random
warmup transitions, then eight updates per 128 collected transitions with
batch size 128. Each run ends at 32768 transitions/1984 updates. Evaluate
at 1024/4096/8192/16384/32768 transitions without selection or early stopping.
One point per 32-step ordinary-return episode is the rough scale worth
expanding, rather than small changes in the preceding action-grid loss.

Evaluation uses unshifted observed rewards and reports both deterministic
and stochastic policies, with paired external RNG streams and 256 episodes.
Also report the conditional-mean reward return, integrating out independent
additive reward noise. Soft-Q probes use 3 states, 2 paired prescribed
actions and 256 current-policy continuations, exclude root entropy, and
integrate out reward noise only for estimating the return mean. Subtract
the known offset*(1-gamma^32)/(1-gamma) before Q-error comparison; raw values
are retained. This is a controlled online learning pilot, not a bus/MuJoCo
result or a verified dual-uncertainty algorithm. No checkpoints or remote jobs;
formal training modules and manuscript sources stay unchanged.

Completed in `../diagnostics/online_bias_uncertainty_20260914/`: 30 runs,
150 evaluations, 7440 learning rows and 900 probe rows, all expected keys
present exactly once and all numbers finite; runtime 109.34 s, about 1.3 MB.
At high noise, final deterministic returns are 29.088 (min), 28.904 (mean),
28.357 (+1 offset); these final differences are below the 1-point expansion
scale. The offset's learning curve is consistently worse after warmup:
descriptive time-weighted mean difference -4.632 (paired SE 1.283), using
all five fixed evaluation times, not a selected checkpoint. Corrected final
Q bias worsens -3.332 to -6.564 despite raw Q rising 4.719 to 10.895.
No replacement is adopted. Next: distinguish reward-origin/target-scale
effects from action-dependent noise effects; the positive-only offset test
does not settle arbitrary signs, adaptive offsets, or the historical bus run.

## Action-dependent online noise: protocol 2026-09-14

`action_dependent_online.py` locks seeds 5000--5009 and min/mean targets,
with no reward offset: 60 runs at the preceding pilot's unchanged budget,
networks, optimizer, observations, transitions and conditional mean rewards.
Variance is 13+12*d*tanh(4*s*a), for d=0 (homogeneous), -1 (state-opposing
actions noisier), and +1 (state-reinforcing actions noisier). These are
mirror controls, not labels for universally optimal/suboptimal actions.
Both action-dependent profiles average to variance 13 under symmetric
uniform actions at every state. Learned policies need not have equal noise
exposure, so log actual collection/training/evaluation variance and the
evaluation fraction s*a<0. No parameters are selected from these results.

Use final and all-checkpoint time-weighted full-episode expected returns,
retaining the approximately 1-point expansion scale. Report both observed
rewards and conditional-mean rewards; only evaluation integrates out reward
noise. Policy input contains state/time, not past rewards, and this noise
does not affect transitions. This integration preserves expected return,
but observed noise no longer cancels between different learned policies.
Also retain Q error separately for opposing/reinforcing root actions and
paired noise-layout effects versus homogeneous, including the target-mode
interaction. Two local CPU cores, no checkpoints or remote dispatch;
manuscript and formal algorithm sources remain unchanged.

Completed in `../diagnostics/action_dependent_online_20260914/`: 60 runs,
300 evaluations, 14880 learning rows and 1800 probes; expected keys present
exactly once, finite throughout, 13 tests pass. Runtime 164.06 s, 2891584
bytes. Mean targets reduce final signed Q underestimation by 0.288--0.311,
but mean-minus-min final conditional-mean episode-return differences have
magnitude below .08 for both policy modes in all three layouts. The min
agent's realized training variance is 13.00/15.52/10.59 for homogeneous,
opposing-noisy/reinforcing-noisy: the noise intervention is active, not
equalized away by learning. Under reinforcing-noisy, mean's deterministic
time-averaged return is lower by 1.655 (paired SE .911); the largest losses
are seeds 5005/5009. No consistent return benefit supports adopting mean.

Next: inspect the historical bus degradation segments and their actual
reward/target definitions before expanding toy experiments. This pilot
changes reward variance only, not transition noise, and does not reproduce
the claimed sustained bus collapse or test the full RE-SAC mechanism.

## Bus archive diagnosis: 2026-09-14

`bus_archive_diagnostics.py` reads 24 local runs into 120 unselected
100-episode windows and executes the original bus loss functions through
AST extraction. Six tests pass in the existing LSTM-RL environment; no
training, environment installation, checkpoint reads or paper edits.
Artifacts: `../diagnostics/bus_archive_diagnostics_20260914/`.

SAC reward degradation is present in the stored curves. Comparing episodes
101--200 with 401--500, seed 8 changes -443950 to -542983, seed 32 -456361
to -498345, and seed 40 -444838 to -469475. On seeds 32/40, K=10 wins the
last-100 comparison by 55719/20415, so the observed performance advantage
survives this audit. Legacy SAC has different alpha/regularization traces;
it is not a verified third matched replicate of the new SAC launcher.

The actual bus loss uses headwise self-bootstrap, batch-centered/scaled
rewards and a positive detached target-norm bonus. These three operations
are unchanged from the initial commit. Numerical checks confirm that the
same raw reward -100 maps to -15 or +15 in different batches, and unequal
head bonuses can reverse an actor LCB action gradient with raw Q fixed.
The replay-std penalty has zero direct common-shift gradient. Aggregate
Q decline is not an absolute bias estimate or evidence of identifiable
aleatoric/epistemic separation. Full versus epistemic-only also changes BC,
and the historical training loop never inserts a terminal-true transition.

Next: an independent instrumented bus entry point, keeping the historical
runner and data intact. Resolve day-end replay semantics, log reward moments
and comparable soft-Q references, and distinguish the historical headwise
baseline from a clipped-target baseline before a small paired online run.

## Independent bus integration pilot: protocol 2026-09-14

`bus_instrumented_pilot.py` uses the original simulator, full input timetable,
route sigma 1.5 and original width-64 PyTorch networks. Network seed 6000;
one uniform warmup day (environment 6001), one training day (6002) and one
frozen stochastic-policy evaluation day (6003) for each of headwise/min
targets. Collection, replay, network and environment RNGs are separate.
Each target receives the identical warmup and at most 200 gradient updates,
batch 256, with one final update exercising a newly closed terminal action.
This is a bounded integration run, not convergence or performance evidence.

Retain same-vehicle transitions across trips and same-station events. At
true day end only, close each last action with the zero sum of subsequent
decision rewards and no bootstrap. A time-limited prefix remains incomplete.
Report initial pre-action rewards separately from transition rewards and
retain the full environment sum. Fixed reward divisor 100, no centering or
norm/BC/disagreement terms, per-batch reward moments logged. Correct the
action-density affine Jacobian to log(30), since a=30*tanh(z)+30; the original
log(60) differs by log(2). This change is confined to the independent runner.

Evaluation freezes the policy throughout the day. Compute complete sampled
vehicle-day returns without a learned tail, and soft returns in the same
scaled units as Q, excluding root entropy. These correlated trajectory
samples are not repeated-state Q expectations or independent training seeds.
No checkpoint writes, remote jobs, data changes or manuscript edits.

Completed valid integration in
`../diagnostics/bus_instrumented_pilot_20260914_r2/`: 8 tests pass,
5 full simulator days, 26170 transitions, 95 terminal closures, zero pending
actions and 400 finite gradient updates (200 per target). Frozen-policy
evaluation has 5234 transitions and 19 complete vehicle-days per target;
saved raw/soft returns satisfy their trajectory recursions. Simulation and
updates took 44.56 s, artifacts 1732188 bytes. Two independent executions
of warmup seed 6001 matched all 5234 transitions exactly, including states,
actions, rewards and timestamps; the r2 warmup matches them as well.

Real-data correction to the earlier boundary interpretation: the legacy
station-ID filter skips 243 nonterminal transitions in this warmup, all
cross-trip changes of direction at the same station ID. Their rewards sum
to -7717.97. The new explicit vehicle-day convention retains these links;
it is not a faithful replay of the historical filtered graph. Concrete
examples are in `warmup_boundaries.json`. Cross-trip links and day-end zero
closures are distinct changes and must not be silently mixed with old curves.

The first attempt failed writing NumPy counters before training; r1 completed
integration but failed the full-day repeat because route speeds use Python
`random`, not NumPy. Both attempts remain preserved and r1 is explicitly
marked invalid for seed-paired evidence. r2 seeds/restores both random
sources, including environment construction. Next: extend this corrected
bus pilot with paired training seeds and a longer fixed budget to test
whether sustained reward degradation recurs. The 200-update integration
results do not establish convergence, algorithm superiority or Q-bias causality.

## Paired bus learning curves: protocol 2026-09-14

Before execution, freeze `bus_paired_training.py`: new training seeds
7000--7002, headwise versus min targets, eight full training days and 512
updates per day (4096 per agent), batch 256. Update every ten collected
transitions until the daily cap, using uniform growing replay; no forced
terminal samples. Each pair starts from identical networks and one shared
full uniform-action warmup day. Keep the independent pilot's vehicle-day
graph, fixed reward divisor 100, corrected action density and original
shared-embedding networks. Only target aggregation differs between arms.

Freeze stochastic-policy evaluations at days 0/2/4/8 using two fixed unseen
environment/action seed pairs per training seed. Environment, action,
replay and network RNGs remain separate. Average evaluation days within
training seed; report paired seed differences and the day-2 to day-8 change,
never select a best checkpoint. Primary outcome is complete environment
reward. Also retain action saturation/entropy, sampled soft-return residuals
and Q predictions at 256 fixed warmup state-action pairs. Those fixed probes
measure Q drift, not an off-policy Q-bias oracle. No full return advantage
can be inferred from Q-residual improvement alone.

This extends integration by 20.48 times the update budget; it is still not
a replication of the historical 500-day collapse or a dual-uncertainty
method comparison. Inspect learning trajectories and substantial full-return
changes before choosing the next mechanism intervention. Use one bounded
local CPU process, one Torch thread on the existing two-core affinity;
save small aggregate CSV/JSON only. No paper edits or checkpoint writes.

Completed in `../diagnostics/bus_paired_training_20260914/`: 12 tests pass,
99 full simulator days (3 warmup, 48 training, 48 evaluation), 24576 critic
and 12288 actor updates, 768 learning blocks, 6144 fixed-query predictions
and 875 evaluated vehicle-day summaries. All six baseline evaluation-day
pairs match exactly; saved budgets, finite values, reward accounting and
terminal closures pass. Runtime 1255.01 s; artifacts 792151 bytes.

Final mean raw day returns: headwise -606970, min -593599. Paired
headwise-minus-min differences are +3667/-33139/-10640 (seeds 7000--7002),
mean -13371, paired SE 10712. Headwise raises fixed-query Q by 15.64
(paired SE 1.02, reward/100 units), with no consistent return gain. Both
arms improve at every prespecified evaluation checkpoint in all seeds;
their day-2 to day-8 mean gains are 436633/431466. Meanwhile their sampled
soft-return residuals become more negative by 34.53/34.35. Q level and
sampled residual magnitude do not track return quality monotonically here.

Do not adopt headwise as an underestimation remedy from these results.
The eight-day independent recipe neither reproduces nor disproves the
historical 500-day decline, and does not validate dual uncertainty. Next:
fixed full bus-state counterfactual continuations for candidate actions,
testing whether critic errors misrank actions and incur meaningful realized
return loss, instead of another global-Q-shift comparison. No new campaign
or manuscript update follows automatically from this pilot.

## Full-state bus action continuations: protocol 2026-09-14

Freeze `bus_counterfactual_actions.py` before execution: new training seed
7100, headwise/min, the same eight-day/4096-update recipe, identical warmup
within pair. One final full-day evaluation per arm, without model selection.
First require exact repeated real-state continuation in a separate smoke
prefix. No historical or manuscript edits; models/snapshots stay in memory.

Select four states from one independent uniform-action prefix at the first
decision at/after times 36000/39000/42000/45000, smallest bus ID for ties.
Preserve the entire simulator graph, including arrival histories used for
headways, passengers, routes, trips and all already chosen actions. Replace
only the focal pending action with 10 or 50 seconds, before its application.
Future decisions use the frozen agent policy, with independent Gaussian
streams per scheduled trip so a change in cross-vehicle event order does
not permute random actions. Future environment noise is resampled separately
from the fixed prefix, with common seeds within each paired action contrast.

For each of four snapshots and two agents, execute 16 paired continuations
to true day end: 256 suffixes. First eight pairs choose the soft-return and
system-return preferred actions separately; last eight independently measure
signed gains over the critic's two-candidate argmax. Do not clip negative
confirmation gains or choose an oracle on the confirmation data. Save action
differences and Monte Carlo SEs for matching vehicle-day soft return (without
root entropy or learned tail), discounted/raw focal reward and raw system
suffix reward. Prefix rewards are excluded. Verify actual root dwell time.

This is one training pair and four correlated contexts, not a population
policy-regret estimate. Full-state-conditional returns are not the same as
observation-marginal Q calibration; soft and system objectives may disagree.
Candidate argmax is a critic diagnostic, not the deployed stochastic actor.
Use compact CSV/JSON on the existing single-thread/two-core local runtime.

Completed in `../diagnostics/bus_counterfactual_actions_20260914/`: 18 tests
pass, real-state same-action replay is exact, 8192 critic/4096 actor updates
and 256 unique completed suffixes. All root dwell times equal the assigned
actions; persisted pairing, day ends, finite values and split confirmation
statistics pass. Runtime 1307.96 s; artifacts 131601 bytes. New seed 7100's
final day returns are -585874/-587667 (headwise/min), not a multi-seed result.

Both critics choose (50,50,10,50) seconds across the four fixed contexts.
Confirmation soft-return differences, 50 minus 10, are headwise
0.804/3.391/-0.365/2.567 (MC SE .003/.265/1.247/1.029) and min
0.802/3.310/-0.352/3.466 (.008/.207/1.191/1.127). The near-balanced third
context remains unresolved; the other contexts support the chosen ordering.
The selection-split soft-return selector always picks the same candidate as
the critic, so its confirmation gain over that candidate policy is zero.
The system-return selector differs only in the third context, with signed
confirmation gains -246 (SE 294) / -45 (SE 272), not reliable improvements.
Future noise is active: first-context focal/system raw-return standard
deviations are about 903/10199 even though its paired local effect is stable.

No reproducible consequential critic misranking was found in this finite
two-action diagnostic; this does not establish optimality of the continuous
stochastic actor or calibration of Q. Preserve this negative result. Next:
isolate the historical bus reward recipe's batch centering and scaling,
controlling their separate effects and the reward/entropy tradeoff, before
attributing the historical training decline to uncertainty or Q level.

## Bus reward recipe controls: protocol 2026-09-14

Before execution freeze `bus_reward_controls.py`: fresh seed 7200, six arms,
eight full training days and 512 updates/day (4096 each), batch 256, headwise
targets throughout. Identical warmup, networks and random streams; evaluations
at days 0/2/4/8 on the same two unseen full days. One local Torch thread on
the existing two-core affinity; compact CSV/JSON, no checkpoints or paper edits.

Freeze warmup mean m0 and scale c0=10/sample_std(warmup). The controls are
`fixed`: c0*r; `frozen_center`: c0*(r-m0); `batch_center`: c0*(r-mean(batch));
`batch_scale`: cB*r; `batch_center_scale`: cB*(r-mean(batch)); and
`legacy_recipe`: cB*(r-mean(batch)), where cB=10/sample_std(batch) with the
historical below-1e-6 standard-deviation fallback to 1. In the two matched
batch-scale arms, both target and actor entropy coefficients are
alpha_reference*cB/c0. All other arms use alpha_reference, the existing
cap-.6 temperature optimizer. Thus matched arms hold the instantaneous
alpha/reward-scale ratio fixed relative to the warmup reference; the legacy
arm retains the historical fixed-temperature reward recipe.

Raw full-environment reward is primary. Report all checkpoints, final-minus-
day2 changes, constant-center effect, moving-versus-frozen centering effect,
adaptive-scale effect, center/scale interaction and legacy-versus-matched-
temperature contrast. Log batch center/scale, actual transformed reward
moments, effective alpha/raw reward coefficient, gradient norms and the
implied transformed reward of a raw-zero terminal closure. Centering applies
to terminal rewards too; only the bootstrap is masked. Evaluation does not
compute incomparable transformed-soft-return residuals.

Limitations: one training seed, not six independent seeds. This frozen scale
is not the earlier r/100 recipe, so those runs are not pooled. Matching the
instantaneous entropy ratio does not make moving target units or optimizer
dynamics invariant. Constant reward shifts are not generally neutral in this
finite, policy-dependent vehicle-day graph. The legacy arm copies only the
reward/temperature recipe, not the historical runner or 500-day horizon.

Completed in `../diagnostics/bus_reward_controls_20260914/`: 26 tests pass;
six arms, 97 complete simulator days, 24576 critic/12288 actor updates,
768 learning blocks and 48 evaluations. Saved budgets, finite values, reward
accounting, initial pairing (all non-timing fields), affine transformations,
entropy controls and all summary contrasts pass. Runtime 1010.92 s;
artifacts 309568 bytes, with no checkpoints. Warmup m0=-225.17244,
sample standard deviation 179.63127 and frozen c0=0.05566959.

Final raw day returns, averaging the two evaluation scenarios: fixed
-960306; frozen_center -532341; batch_center -516930; batch_scale -975599;
batch_center_scale -536344; legacy_recipe -524171. Frozen centering improves
over fixed by 427965 (paired scenario differences +449807/+406122).
Moving versus frozen centering adds 15411; scaling alone changes return by
-15293. Centering under matched batch scaling adds 439255, with interaction
-4121. Legacy versus entropy-matched center-and-scale differs by +12173.
All six arms worsen initially at day 2, then improve at days 4 and 8.

The dominant contrast in this seed is the reward offset, not adaptive
scaling or entropy matching. The fixed offset also assigns +12.53526 to
raw-zero terminal closures; this finite-horizon intervention cannot be
described as a harmless Q-coordinate change. This pilot neither reproduces
the historical long-run collapse nor validates uncertainty decomposition or
a causal Q-underestimation explanation. Next: independently repeat only
fixed/frozen_center/legacy_recipe on fresh training seeds 7201 and 7202,
using the same frozen eight-day budget and within-seed paired evaluation.
These confirmation runs have not been executed. No manuscript changes.

## Bus reward offset confirmation: protocol 2026-09-14

Freeze `bus_reward_confirmation.py` before execution: independent seeds
7201/7202, fixed/frozen_center/legacy_recipe only, unchanged per-seed recipe,
eight full training days and 4096 critic updates per arm. Each seed uses
its own identical-within-arm-pair warmup, initialized networks and random
streams; warmup mean/scale stay frozen where specified. Budget: 24576 critic
updates, 48 training days, 48 evaluations at days 0/2/4/8, and two warmup
days. No early stopping or checkpoint selection; final raw day return is
primary, with every prespecified checkpoint retained.

Report frozen_center-minus-fixed, legacy-minus-fixed and legacy-minus-
frozen_center, each as per-training-seed paired differences and their mean
and paired seed SE. Average the two evaluation scenarios within each seed
first; exploratory seed 7200 is excluded from confirmation statistics.
The earlier result selected these arms but does not count as an independent
confirmation replicate. Retain the finite-horizon reward-offset limitation;
the test concerns reproducibility of the performance effect, not a claim
that Q underestimation or dual uncertainty has been established. Run locally
with one Torch thread and the existing two-core affinity; small CSV/JSON
only, with no checkpoints, historical-trainer changes or manuscript edits.

Completed in `../diagnostics/bus_reward_confirmation_20260914/`: 29 tests
pass, six runs and 98 complete simulator days, 24576 critic/12288 actor
updates, 768 learning blocks and 48 evaluations. Saved budgets, finite
values, initial pairing, full-day reward accounting, reward/temperature
controls and independently recomputed seed statistics pass. Runtime
1017.04 s; artifacts 311381 bytes, no checkpoints.

Final raw returns (fixed/frozen_center/legacy_recipe) are seed 7201:
-988879/-1036653/-1040892; seed 7202: -889005/-566813/-538067.
Frozen-center-minus-fixed is -47774/+322193, confirmation mean +137209
and paired seed SE 184983. Legacy-minus-fixed is -52013/+350938, mean
+149463, SE 201475. Legacy-minus-frozen-center is -4239/+28745, mean
+12253, SE 16492. Exploratory seed 7200 is not pooled. In seed 7201 the
centering advantage at day 4 reverses by day 8; in seed 7202 it emerges
only after day 4. All six runs finish above their own day-2 raw returns.

These two new seeds show heterogeneous effects, not stable improvement.
A simple reward offset can still produce a large gain without uncertainty
decomposition, so it remains an explanatory control, not an adopted remedy.
The finite-horizon objective and long-run-collapse limitations remain.
Next: isolate the original bus target's norm bonus under one shared reward
recipe, with no-bonus and prespecified frozen constant-bonus controls and
matched non-target components. That comparison is not yet executed; the
historical trainer and manuscript remain unchanged.

## Bus target norm bonus controls: protocol 2026-09-19

Freeze `bus_norm_bonus.py` before execution: fresh seeds 7300/7301, three
arms none/norm/constant, eight full training days and 4096 critic updates
per arm. All use the historical batch-centered/sample-std reward recipe,
headwise twin targets, identical within-seed initial networks, warmup and
random streams. Keep width 64, Adam 1e-5, actor ratio 2, target tau .01,
critic gradient clip 1 and alpha cap .6. BC, actor dispersion and critic
disagreement regularization are zero in every arm.

The positive bonus is .01 times the original target critic's per-head sum
of layer L1 weights and hidden biases, excluding embeddings and final bias.
It is added inside the continuation before gamma*(1-done), not to terminal
reward and not as a differentiable weight penalty. Constant freezes each
head's initial bonus before any updates, matching both initial magnitude
and head heterogeneity; norm recomputes it from the changing target network.
At zero actor dispersion the detached bonus in the original actor mean
has no action gradient, verified against the original method, and is
omitted from the independent runner's reported actor loss.

Budget: 24576 critic/12288 actor updates, 98 complete simulator days,
48 evaluations at days 0/2/4/8. Final raw full-day reward is primary.
Report norm-minus-none, constant-minus-none and norm-minus-constant per
training seed and their mean/paired seed SE; average the two unseen
evaluation scenarios within each seed first. Log actual per-head bonuses,
norm drift, added target magnitude and reward/temperature metrics. No
early stopping, checkpoint selection or tuning after seeing results.

This isolates the target bonus, not the full K=10 RE-SAC or its interaction
with dispersion/BC. The finite-horizon continuation bonus can change the
objective; higher Q is not calibrated underestimation repair. Two seeds
and eight training days do not settle long-run collapse or dual uncertainty.
Use the existing local one-thread/two-core runtime, small CSV/JSON only;
no checkpoints, historical-trainer changes or manuscript edits.

Completed in `../diagnostics/bus_norm_bonus_20260919/`: 35 tests pass,
including exact agreement with the original norm method, positive masked
target placement, unchanged no-bonus updates and the zero-dispersion actor
gradient. Six runs complete 98 simulator days, 24576 critic/12288 actor
updates, 768 learning blocks and 48 evaluations. Saved budgets, finite
values, initial pairing, reward accounting, reward/temperature controls,
applied per-head bonuses and recomputed paired seed statistics pass.
Runtime 1096.80 s; artifacts 415952 bytes, without checkpoints.

Final raw returns (none/norm/constant) are seed 7300:
-513884/-980469/-981835; seed 7301: -501857/-1067488/-1051573.
Norm-minus-none is -466585/-565631, mean -516108, paired seed SE 49523.
Constant-minus-none is -467950/-549716, mean -508833, SE 40883.
Norm-minus-constant is +1365/-15915, mean -7275, SE 8640. Both bonus arms
underperform none in all four paired final evaluation scenarios. Norm and
constant also finish below their own day-4 returns in both seeds.

Initial mean bonuses are 7.83804/7.83168; the norm arms' last 32-update
block means are 7.98525/7.93527, with mean maximal head-relative drift
1.881%/1.325%. Thus this short run has limited norm adaptation and shows
no reproducible benefit over the initially matched constant bonus. Preserve
the negative result: isolated positive norm compensation did not repair
performance here. This does not identify the cause of historical RE-SAC's
gains or justify reversing its sign. Next: a matched full-component
comparison retaining the same actor dispersion, BC, critic regularizer and
ensemble size across no-norm/norm/frozen-constant arms, to test interactions
that this isolation deliberately removes. That comparison is not executed;
no manuscript or historical-trainer change follows from this pilot.

## Full-component bus norm controls: protocol 2026-09-19

Freeze `bus_full_components.py` before execution: fresh seeds 7400/7401,
none/norm/constant, K=10 and width 64 in all arms. Retain the historical
resac_K10 loss settings beta=-2, beta_bc=.001 and beta_ood=.01 from
`scripts/run_bus_seed_extension.py`. All arms use batch-centered rewards
scaled by 10/sample_std, alpha cap .6, Adam 1e-5, target tau .01 and critic
gradient clip 1. Only the norm intervention differs: zero, .01 times the
current target critic norm, or each head's initial value frozen before
training. Apply that intervention both inside the masked Bellman
continuation and inside the actor's mean and sample standard deviation.
The actor reuses the pre-critic-update target norm, as in the source.

Keep BC and the critic disagreement penalty on when removing the norm,
unlike the old epistemic-only script that also removed BC. Direct tests
compare both losses and gradients with the original source methods. Log
all ten applied and actual norm bonuses, BC penalty, raw/shifted actor
dispersion and critic disagreement penalty to establish actual activation.

Use the same independent eight-day recipe: 512 critic updates/day, batch
256, one update per ten new transitions until the daily cap, actor and
temperature every two updates. Six runs total 24576 critic/12288 actor
updates, 98 full simulator days and 48 evaluations at days 0/2/4/8.
Final raw day reward is primary, with all checkpoints preserved; two
unseen evaluation scenarios are averaged within each training seed before
paired means/SEs. No early stopping, result-driven tuning or pooling with
earlier K=2 seeds. Run locally with one Torch thread and two-core affinity.

Limitations: full historical loss components, not the entire historical
runner. Batch size (historically 2048), update budget and temperature
cadence, complete vehicle-day transitions and corrected action density
remain those of the independent diagnostic. The no-norm arm retains the
other RE-SAC components and is not SAC. Two seeds and eight days neither
identify long-run collapse nor establish Q calibration or uncertainty
decomposition. Small CSV/JSON only; no checkpoints or manuscript edits.

Completed in `../diagnostics/bus_full_components_20260919/`: 41 tests pass,
including original-source actor/critic loss and gradient comparisons and
pre-critic target-bonus reuse in the actor. Six runs complete 98 simulator
days, 24576 critic/12288 actor updates, 768 learning blocks and 48 evaluations.
Saved budgets, finite values, initial pairing, complete reward accounting,
all ten per-head bonus controls, active BC/dispersion penalties, loss
component arithmetic and independently recomputed paired statistics pass.
Runtime 1261.22 s; artifacts 746793 bytes, without checkpoints.

Final raw returns (none/norm/constant): seed 7400
-508596/-968956/-949063; seed 7401 -668388/-990302/-981149.
Norm-minus-none is -460360/-321914, mean -391137, paired seed SE 69223.
Constant-minus-none is -440468/-312761, mean -376614, SE 63853.
Norm-minus-constant is -19893/-9153, mean -14523, SE 5370. Both bonus
arms underperform none in all four final paired evaluation scenarios.
Seed 7401's norm arm leads none at days 2 and 4, but trails at day 8;
seed 7400 trails throughout the evaluated training checkpoints. Preserve
that time-dependent ordering, not just the final negative contrast.

Initial mean bonuses are 7.80305/7.79334; norm-arm last-block means are
7.93014/7.92404, with mean maximal head-relative drift 1.909%/1.822%.
All 768 blocks retain nonzero BC, actor dispersion and critic disagreement
penalties. Within this short full-component test, there is no performance
benefit from retaining the norm bonus or adapting it instead of freezing
it. None is not SAC; this test does not compare full RE-SAC against SAC
or settle the historical long-run claim. Next: a prespecified longer
budget for the same three matched arms to distinguish learning delay
from persistent impairment, before retaining or redesigning the bonus.
That extension has not run. Manuscript and historical trainer unchanged.

## Fixed 32-day full-component extension: protocol 2026-09-19

Freeze `bus_long_budget.py` before execution. Extend all three K=10
full-component arms none/norm/constant to 32 training days, 512 critic
updates/day, with every loss, coefficient, optimizer, batch size and
transition rule unchanged. Reuse seeds 7400/7401: this investigates the
same learning trajectories, not additional independent training replicates.
No checkpoints exist from the short run, so recompute from initialization.
Keep the first eight training environment/action seeds unchanged, extend
their disjoint streams through day 32 and reuse the same evaluation seeds.

Prespecify evaluations at days 0/2/4/8/16/24/32. Require exact agreement
with all non-timing fields in the old 0/2/4/8 evaluation rows as each arm
runs; a mismatch stops that branch before interpreting it as an extension.
Primary: raw day-32 return. Secondary: mean paired contrasts across days
24/32 and within-arm day32-minus-day8. Report every checkpoint and signed
per-training-seed contrasts; average scenarios inside seeds. Do not select
the best checkpoint, stop early, change coefficients or extend past day 32
in response to the observed curves.

Budget: 98304 critic/49152 actor updates, 192 training days, 84 evaluation
days and two warmups (278 complete simulator days). The first 24576 critic
updates reconstruct the old prefix; 73728 extend it. Two spawned local
workers each run one seed's arms sequentially, one Torch thread pinned to
CPU 0 and CPU 2 respectively (distinct physical cores verified). Small
CSV/JSON only, no checkpoints, remote transfers or manuscript edits.

Limitations: seeds and arm selection are inherited from the exploratory
short run. Longer performance curves do not prove convergence, statistical
uncertainty decomposition, Q calibration or historical 500-day replication.
The no-norm arm retains BC and both dispersion terms and is not SAC.

Completed in `../diagnostics/bus_long_budget_20260919/`: 45 tests pass.
All six runs finish: 278 complete simulator days, 98304 critic/49152 actor
updates, 3072 finite learning blocks and 84 evaluations. All 48 old prefix
evaluation rows match exactly except timing. Saved budgets, reward accounting,
per-head interventions, active loss components and independently recomputed
paired statistics pass. Runtime 3176.66 s; artifacts 2942096 bytes, no checkpoints.

Day-32 raw returns (none/norm/constant): seed 7400
-487691/-984238/-982405; seed 7401 -486077/-1024976/-1015771.
Norm-minus-none is -496547/-538899, mean -517723, paired seed SE 21176;
constant-minus-none is -494714/-529693, mean -512204, SE 17490.
Norm-minus-constant is -1833/-9206, mean -5519, SE 3686. The prespecified
24/32-day mean contrasts are respectively -512311, -508615 and -3696.
Both bonus arms trail none in all four paired final evaluation scenarios.
From day 8 to 32, none improves by +20905/+182311, while norm changes by
-15282/-34674 and constant by -33342/-34622. Norm-arm last-block mean
bonuses reach 8.42553/8.45952; mean maximal head-relative drift reaches
9.247%/9.235%, compared with roughly 1.9% at day 8.

Extending this diagnostic to 32 days does not rescue the positive norm
bonus. Freezing it remains similarly detrimental, so norm growth alone
does not explain the impairment. This is the same two-seed extension, not
independent confirmation or a comparison against SAC. Next: a matched
budget comparison of standard SAC, same-size ensemble SAC and no-norm
RE-SAC to determine whether the remaining components provide a benefit.
That comparison has not run; manuscript and historical trainer unchanged.

## Matched SAC bus controls: protocol 2026-09-19

Freeze `bus_sac_controls.py` before execution: fresh seeds 7500/7501,
32 training days, 512 critic updates/day, batch 256, one update per ten
transitions, with evaluations at 0/2/4/8/16/24/32 on two fixed unseen
scenarios per seed. Three arms: twin-min SAC (K2, minimum in both target
and actor); ensemble SAC (K10, headwise targets and mean-Q actor); and
no-norm RE-SAC (K10, same headwise targets, actor beta=-2/BC=.001 and
critic disagreement=.01). No arm has a norm bonus. The last comparison
tests the remaining components jointly, not their individual attribution.

Keep the historical shared embedding, width64, batch-centered/scaled
reward, alpha cap .6, Adam1e-5, target tau .01 and gradient clip1 in all
arms. Reinitialize actor layers with the separate seed training_seed+500
so ensemble size does not alter the initial policy. K10 critic/target
initializations are identical between the two K10 arms; action, environment
and replay RNG streams are paired. This is a twin-min SAC control under
the shared bus recipe, not a separately tuned canonical SAC reproduction.

Primary: day32 raw return difference no-norm RE-SAC minus ensemble SAC.
Secondary: contrasts with twin-min SAC, paired mean across days24/32 and
within-arm day32-minus-day8. Average scenarios within training seed before
paired SE; preserve all checkpoints. No early stopping, selected checkpoints,
tuning or extension after observing results. Six runs: 98304 critic/49152
actor updates and 278 full simulator days. No pooling with previous seeds.

Device benchmark `../diagnostics/bus_device_benchmark_20260919/timings.json`
uses independent seed7900, K10 no-norm and the same batch/update recipe.
CPU/CUDA updates cost 17.14/16.89 ms and complete training days cost
28.44/28.51 s in this single timing sample. CUDA is supported, not clearly
faster end to end. Choose local CUDA for both formal workers, one CPU
thread each on CPUs0/2, sharing the RTX4060; no CPU-server dispatch.
All random action noise is generated by independent CPU generators and
copied to the network device; TF32 is off. Device tests cover placement,
CPU/CUDA short-update agreement and evaluation. Small CSV/JSON only;
no checkpoints, remote data transfers, historical-trainer or paper edits.

Completed in `../diagnostics/bus_sac_controls_20260919/` on local CUDA:
52 tests pass, including CPU/CUDA short-update agreement and baseline
loss/gradient definitions. Six runs finish 278 complete simulator days,
98304 critic/49152 actor updates, 3072 finite learning blocks and 84
evaluations. Saved budgets, terminal reward accounting, initial-policy
pairing, active/inactive loss coefficients and independently recomputed
paired statistics pass. Runtime 2758.91 s; artifacts 1630478 bytes.
All arms use the existing high-noise bus factory, route_sigma=1.5.

Day32 raw returns (twin-min SAC/ensemble SAC/no-norm RE-SAC): seed7500
-494984/-495421/-489973; seed7501 -481251/-482873/-495660.
The primary no-norm-minus-ensemble difference is +5448/-12786, mean
-3669, paired seed SE9117. No-norm-minus-twin-min is +5011/-14409,
mean -4699, SE9710. The prespecified day24/32 primary mean is
+5323/-8485 across seeds, overall -1581, SE6904. Thus the remaining
component bundle shows no consistent gain here. This is not an
equivalence test or evidence of universal inferiority.

Both seeds trail ensemble SAC at days4/8/16; the paired mean deficits
are -124518/-88554/-19779 respectively. Late recovery does not establish
faster learning. Preserve this learning delay alongside the mixed final
contrast; do not count day32-minus-day8 improvement as a baseline gain.
The shared reward/temperature recipe and two training seeds bound these
conclusions; no calibrated Q-bias or uncertainty-decomposition claim follows.

Next: separate actor dispersion, BC and critic disagreement into individual
matched K10 interventions to locate their contributions and interactions.
That attribution experiment has not run. Historical trainer and paper unchanged.

## Single-component bus attribution: protocol 2026-09-19

Freeze `bus_component_attribution.py` before execution. Inventory confirms
the two K10 controls in `bus_sac_controls_20260919` are complete, with no
singleton runs. Reuse ensemble_sac and resac_no_norm evaluations for seeds
7500/7501; run only actor dispersion (beta=-2), BC (.001) and critic
disagreement (.01) individually. All six new runs retain K10, width64,
headwise targets, norm=0, the same actor initialization, RNG streams,
route_sigma=1.5, reward recipe and optimizers. This extends the exploratory
same-seed study rather than adding independent confirmation seeds.

Training remains 32 complete days, 512 critic updates/day, batch256,
update every10 transitions; evaluations at 0/2/4/8/16/24/32 with two paired
unseen scenarios. Primary: all three day32 singleton-minus-ensemble raw
return contrasts. Secondary: days24/32 averages, day32-minus-day8, and
joint nonadditivity: (full-baseline) minus the sum of three singleton gains.
Compute the residual within each seed before mean/SE. It combines pairwise
and three-way effects; it does not identify any individual interaction.
No selected winner, retuning, early stopping or budget extension.

Reuse 56 archived evaluations from four completed control runs. New budget:
98304 critic/49152 actor updates, 192 training and 84 evaluation days.
Three CUDA workers each run one component's two seeds sequentially, with
one CPU thread pinned to CPU0/2/4 respectively (distinct physical cores).
Each run recomputes a warmup, giving six warmups and 282 new simulator days.
Require matching reference protocol/device/library versions and matching
non-timing day0 evaluation fields before continuing each branch. Tests
check singleton loss/gradient definitions and their reconstruction of the
full loss at fixed inputs; performance need not be additive after training.
Log actual actor dispersion, BC and critic penalties separately. Preserve
new and combined evaluation CSVs with distinct new/reused accounting.
Small CSV/JSON only; no checkpoint downloads, paper or historical-trainer edits.

Completed in `../diagnostics/bus_component_attribution_20260919/`: six new
CUDA runs plus four reused controls, 59 passing tests. The new runs finish
282 complete simulator days, 98304 critic/49152 actor updates, 3072 finite
learning blocks and 84 evaluations; combined evaluations total 140.
Saved loss coefficients, budgets, terminal accounting, all 12 initial
evaluation matches, reference reuse and independently recomputed paired
statistics pass. Runtime 2557.55 s; new artifacts 1783613 bytes.

Day32 singleton-minus-ensemble raw returns for seeds7500/7501 are:
actor dispersion +2201/-9057 (mean -3428, paired seed SE5629);
BC +3575/+4538 (mean +4056, SE481); critic disagreement +9557/+4590
(mean +7074, SE2484). Days24/32 paired means are respectively -1768,
+2601 and +5285; their per-seed values are +2835/-6372, approximately
0/+5203, and +4141/+6428. BC improves the final endpoint but trails the
baseline in both seeds at days4/8; mean deficits are -116966/-66538.
Actor dispersion also trails in both seeds at days4/16. Critic disagreement
has the more consistent late-window benefit, not an across-training advantage.

The full-minus-additive-singleton residual at day32 is -9885/-12857,
mean -11371 (SE1486); the days24/32 mean is -7699 (SE6046). Singleton
gains therefore do not simply carry over to the original combination.
These are same-seed exploratory effects, not independent confirmation,
identified individual interactions, Q-bias calibration or dual uncertainty.

Next: add only BC plus critic disagreement, retaining norm=0 and removing
actor dispersion, on the same two seeds and budget. Reuse all existing arms:
comparison with the full bundle tests removing actor dispersion, and with
critic-only tests whether BC helps when critic regularization is present.
This two-run follow-up has not run; no coefficient tuning or environment
expansion is planned for it. Historical trainer and manuscript unchanged.

## BC plus critic bus follow-up: protocol 2026-09-19

Freeze `bus_bc_critic.py`: only the BC=.001 plus critic-disagreement=.01
arm is new, with actor dispersion and norm bonus both zero. Reuse all five
existing K10 arms for seeds7500/7501 from the SAC and singleton studies.
Keep the same width64 networks, initial actor, headwise targets, random
streams, route_sigma=1.5, reward/temperature recipe and optimizers. This
pair was selected after the singleton results; these are not new independent
confirmation seeds. No retuning, early stopping or budget extension.

Primary: day32 raw return of pair minus the full no-norm bundle, isolating
removal of actor dispersion with both other terms on. Secondary: pair minus
critic-only (BC contribution with critic regularization on), ensemble SAC
and BC-only; paired days24/32 averages of all four contrasts and within-arm
day32-minus-day8. Average the two evaluation scenarios within each seed
before paired mean/SE. Report every frozen checkpoint, not a selected peak.

Two new CUDA runs on local RTX4060, one Torch thread each on CPU0/2,
with matching non-timing day0 evaluations required against archived controls.
Each run has 32 training days at 512 critic updates/day, batch256, update
every10 transitions; evaluations at 0/2/4/8/16/24/32 on two unseen scenarios.
New totals: 32768 critic/16384 actor updates, 64 training days, 28 evaluation
days and two warmups, or 94 complete simulator days. Reuse ten completed
runs and 140 evaluations; the combined analysis has 12 runs and 168 evaluations,
but still only two training seeds. Tests cover both active penalties, absent
actor/norm penalties, loss gradients, CUDA agreement and paired accounting.
Small CSV/JSON only; no checkpoints, historical-trainer or manuscript edits.

Completed in `../diagnostics/bus_bc_critic_20260919/`: 62 tests pass. Both
CUDA runs finish 94 complete simulator days, 32768 critic/16384 actor
updates, 1024 finite learning blocks and 28 new evaluations. Saved budgets,
active BC/critic and inactive actor/norm terms, terminal accounting, four
initial evaluation matches, reuse of 140 archived evaluations and independent
paired-statistic recomputation pass. Runtime 1159.59 s; artifacts 658109 bytes.

Day32 pair raw returns are -493563/-476909 for seeds7500/7501. The primary
pair-minus-full contrast is -3591/+18751, mean +7580 (paired seed SE11171).
Pair-minus-critic is -7700/+1374, mean -3163 (SE4537); pair-minus-ensemble
is +1858/+5964, mean +3911 (SE2053); pair-minus-BC is -1717/+1427, mean
-145 (SE1572). Days24/32 mean contrasts are respectively +4522, -2344,
+2941 and +340. Against critic-only, both seeds trail at days4/8/16/24;
mean deficits at days4/8/16 are -95958/-73334/-16845.

Removing actor dispersion does not improve both seeds, and retaining BC
has no consistent advantage over critic-only. The modest final gain against
ensemble SAC does not justify the extra component given the learning delay
and mixed late-window comparison. No new combination is adopted. These
same-seed exploratory results do not establish population-level effects,
Q-underestimation correction or uncertainty decomposition.

Next: stop expanding same-seed component combinations; independently confirm
critic-only versus ensemble SAC on five fresh training seeds at the unchanged
budget and coefficients. Inventory and freeze the new seeds before execution;
exclude exploratory seeds from confirmation statistics. That confirmation
has not run. Historical trainer and manuscript remain unchanged.

## Independent critic bus confirmation: protocol 2026-09-19

Freeze `bus_critic_confirmation.py` before execution. Inventory of existing
bus configurations finds no use of seeds7600--7604 and no running duplicate.
Run exactly those five new training seeds with ensemble SAC and critic-only
regularization=.01. Both arms have K10, width64, headwise targets, mean-Q
actor and zero actor dispersion, BC and norm bonus. Keep route_sigma=1.5,
the shared reward/temperature recipe and all optimizer settings unchanged.
Initialize policy with seed+500 and pair environment/action/replay streams
within each seed. Seeds7500/7501 selected the candidate and are excluded
from confirmation statistics. No new coefficient search or replacement seeds.

Primary: fixed day32 raw-return difference critic-only minus ensemble SAC.
Average the two evaluation scenarios within each seed, then report all five
differences, mean, paired SE and approximate two-sided 95% Student-t interval
with four degrees of freedom. Secondary: paired days24/32 mean and each
arm's day32-minus-day8; retain every checkpoint. The late window cannot
replace the primary endpoint. No early stopping, best-checkpoint selection
or budget extension. Five seeds do not validate the t-model assumption or
establish Q-bias correction, dual uncertainty or generality beyond this recipe.

Each run has 32 full training days with 512 critic updates/day, batch256,
update every10 transitions; evaluations at 0/2/4/8/16/24/32 with two unseen
scenarios. Ten new runs total 163840 critic/81920 actor updates, 320 training
days, 140 evaluation days and ten warmups, or 470 complete simulator days.
Four CUDA workers share the local RTX4060, one Torch thread each pinned
to CPU0/2/4/6 on distinct physical cores. Each slot runs its fixed job list
sequentially; paired arms recompute identical warmups. Require non-timing
day0 evaluation equality within every completed pair before interpreting
results. Tests verify unchanged baseline updates, active/inactive components,
GPU agreement, seed-level statistics and exclusion of exploratory data.
Small CSV/JSON only; no checkpoints, historical-trainer or manuscript edits.

Completed in `../diagnostics/bus_critic_confirmation_20260919/`: 67 tests
pass. Ten CUDA runs finish 470 full simulator days, 163840 critic/81920
actor updates and 140 evaluations. Saved budgets, 5120 finite learning
blocks, actual active/inactive losses, terminal accounting, paired warmups,
ten initial evaluation matches and independently recomputed seed statistics
pass. Runtime 6069.82 s; artifacts 2864879 bytes, no checkpoints.

Day32 critic-minus-ensemble differences for seeds7600--7604 are
-3954/+19436/+1751/+1477/-1672: mean +3408, paired SE4144, approximate
95% t interval [-8097,+14912]. Three seeds improve and two worsen. The
prespecified days24/32 contrast is -5419 (SE8657), interval [-29456,+18617],
with four negative seed differences. The late deficit includes seed7602's
delayed recovery; its day24 difference is -76837 but day32 is +1751.

This fresh-seed confirmation does not establish a stable benefit from the
.01 critic penalty over matched K10 ensemble SAC. It does not establish
zero effect either. Do not adopt critic-only as the rescued method or pool
the exploratory seeds into this confirmation. Both arms improve from day8
to day32 in every seed, leaving the historical long-run decline unresolved.

Next: return to that long-horizon question. First align the archived 500-episode
training horizon with simulator days and gradient-update counts, then freeze
a matched long-horizon SAC control to test whether decline persists under
the corrected transition and action-density conventions. No further penalty
combinations, coefficient search or result-dependent extensions are launched.
Historical trainer and manuscript remain unchanged.

## Fixed-schedule 500-day bus controls: protocol 2026-09-19

Budget alignment finds that the historical launcher uses 500 full simulator
days, batch2048, replay capacity1000000 and training_freq5, not the recent
batch256/512-update diagnostic recipe. Its update gate uses buffer length
modulo5 rather than total insertions. At capacity the length stays divisible
by5, so each event batch with a new stored transition can trigger an update.
Before capacity, multi-transition batches can also skip modulo boundaries.
The archived per-day means do not record exact optimizer counts. Therefore
500*512 is not an established historical budget, and the late schedule change
is a real implementation issue, not yet a causal explanation of the decline.

Freeze `bus_long_horizon.py`: fresh seeds7700--7702, K2 headwise target/mean-Q
actor versus K2 clipped-min target/min-Q actor. Both share initial networks,
policy seed+500, reward centering/sample-std scaling*10, optimizer settings,
route_sigma1.5, corrected vehicle-day transitions and affine action density.
All norm, BC and disagreement penalties are zero. Use one full uniform
warmup, batch2048 and exactly1024 critic/512 actor updates per training day,
one update per5 collected transitions until the cap, independent of replay
occupancy. Bounded uniform replay holds at most1000000 transitions.

Each of six GPU runs has500 training days,512000 critic/256000 actor updates.
Evaluate two fixed unseen scenarios at days0/8/32/50 and100:25:500. Primary:
within-arm mean return at425/450/475/500 minus125/150/175/200. Secondary:
paired difference of those changes, final headwise-minus-min, and training
reward means401:500 minus101:200. Average scenarios/checkpoints inside each
training seed before mean/SE. No selected peaks, replacement seeds, tuning
or result-dependent extensions. Totals:3072000 critic/1536000 actor updates,
252 evaluation days and six warmups, or3258 complete simulator days.

Long-run random streams use disjoint ranges; the old short-run seed offsets
would overlap training and evaluation by day200 and are not reused. Save
only latest.pt after days0/1 and every10 days, including agent, optimizers,
replay, RNG and completed records. Atomic replacement leaves the prior full
checkpoint recoverable during a write; interruption replays at most ten days.
Keep checkpoints on the execution server and auto-pull only CSV/JSON results.
Scheduler launches separate GPU tasks from a small frozen source/data copy;
the turn ends after launch validation, not after the500-day training.

Limitations: this is a corrected fixed-budget long-horizon diagnostic, not
an exact legacy rerun. Both actor and target aggregation differ between arms.
It does not isolate the historical update-gate bug or calibrate Q bias, and
three seeds cannot establish uncertainty decomposition. Manuscript unchanged.

Deployed through scheduler as t95596--t95601 on jtl110gpu, three per3080Ti,
with distinct physical CPUs0--5 and one Torch thread each. A small frozen
runtime copy is under `../diagnostics/bus_long_horizon_20260919/runtime/`;
task commands and result/checkpoint paths are in the sibling `tasks.json`.
All27 targeted tests pass, including actual CUDA resume and an interrupted
day replay. The server's full batch2048 CUDA update probe also passes.
Startup inspection confirms six CUDA PIDs, about362MiB each, paired initial
return logs and all six day0 checkpoints; all are collecting/training day1.
Only latest checkpoints remain server-side in the campaign's `checkpoints/`;
per-run result directories exclude them and are configured for automatic
CSV/JSON pull-back. Training is running, not completed or interpreted.
Per user request, stop after deployment; analyze after the user reports completion.

Completed and analyzed 2026-09-20. All six original scheduler tasks exit0,
with server success logs, no detected crash/nonfinite markers and no resumes.
Each completes500 days,512000 critic/256000 actor updates and42 evaluations
in6.23--6.26 hours. Result auto-sync succeeds for all six:51386088 bytes of
CSV/JSON, no checkpoint downloads. All96000 learning blocks are finite;
daily/block budgets, zero extra losses, full-day accounting, replay bounds,
paired warmups/day0 evaluations and independent summary recomputation pass.
Every replay reaches capacity on day191; updates remain1024/day afterwards.
Artifacts: `../diagnostics/bus_long_horizon_20260919/{summary,verification}.json`
and `window_metrics.csv`.

Primary late-minus-reference evaluation changes for seeds7700/7701/7702:
headwise -7955/-13863/-15978, mean -12598 (paired seed SE2401);
clipped-min -11665/-13669/-20300, mean -15211 (SE2609). Both decline in
all three seeds, roughly2.8%/3.4% of the reference return magnitude. Training
windows also decline in all six runs (arm means -12122/-11461). The paired
difference of evaluation declines is +3709.97/-194.07/+4322.48, mean +2613
(SE1415). Day500 headwise-minus-min is +11090/+6075/+10190, mean +9118
(SE1544), but descriptive late-window superiority is mixed across seeds.

Long-run degradation therefore persists under a capacity-independent update
schedule, including without clipped-min aggregation; it is modest rather
than catastrophic. This does not identify the cause of the historical curves
or demonstrate a RE-SAC benefit. Descriptively, both arms' mean alpha falls
from about.441 to.095 and boundary-action fraction rises from about.18 to.40.
Headwise native Q changes120.71 to5.93; clipped-min12.92 to-160.89. Changing
entropy terms and visited state-action distributions preclude interpreting
these levels alone as calibrated underestimation.

Next candidate intervention: freeze alpha at its day100 value, keeping each
arm's prefix, reward recipe and remaining updates unchanged, and compare
against these adaptive-alpha controls. This would test temperature evolution,
not merely raise Q; it has not been implemented or launched. No manuscript
edits, new training submissions or coefficient tuning in this analysis turn.

## Day100 alpha freeze: protocol 2026-09-20

Freeze `bus_alpha_freeze.py`: six new runs, seeds7700--7702 and the same K2
headwise/clipped-min arms, paired to the completed adaptive-alpha controls.
Recompute days0--100 because only final checkpoints were retained. Require
exact non-timing warmup records,100 training rows,3200 learning blocks and
ten evaluation rows per run to match the archive before day101. A mismatch
stops that task; it is not an excuse to relax matching or replace its seed.
On subsequent updates, set only the alpha optimizer learning rate to zero,
preserving its day100 log_alpha and all other updates. All six archived
day100 effective alphas equal0.6. This is not a newly tuned coefficient.

Reuse the full500-day,1024-updates/day,batch2048 schedule, disjoint random
streams, initial networks, bounded replay and fixed evaluation scenarios.
Primary: within each aggregation, frozen-minus-adaptive raw return averaged
over days425/450/475/500 and scenarios within seed, then three-seed mean/SE.
Secondary: day500 gain, all checkpoint contrasts, each treatment's late
return minus its common day100 reference, and the previous125:200-to425:500
window changes. Days125--200 are post-intervention, not a pretreatment anchor.
Do not pool aggregation arms as independent seeds or select a favorable peak.

New totals remain3072000 critic/1536000 actor updates and3258 simulator days;
614400 critic updates recompute prefixes,2457600 occur after intervention.
Reuse six completed controls without retraining them. Latest checkpoints
include optimizer state/lr, replay, RNG and records, with the same10-day
cadence plus days0/1; only CSV/JSON results return to local. Runtime matching
and restored frozen temperature are checked before continuation. Source is
copied to a dedicated campaign runtime; historical code/results stay intact.

This is a same-seed mechanistic follow-up selected after observing alpha
decay, not independent algorithm confirmation. Changes in native Q are not
calibrated bias. No manuscript edits, additional coefficients or budget
extensions. Deploy through scheduler on GPU, verify startup, then return
without waiting for the prefix or the full training to finish.

Deployed as t97436--t97441 on jtl110gpu, same GPU assignments and physical
CPUs0--5 as the archived controls, three tasks per3080Ti. Local20 relevant
tests and all five new tests on the server pass, including exact prefix
updates, freeze-boundary behavior, CUDA optimizer restore and interrupted
post-freeze continuation. Archived reference configs and prefix row counts
load successfully for all six. Startup confirms six CUDA processes around
362MiB each, distinct live affinities, matched warmup/day0 records and six
day0 checkpoints. All are training day1; the full100-day matching gate has
not yet completed. Task specs, frozen runtime and startup evidence are in
`../diagnostics/bus_alpha_freeze_20260920/`. Return now; await the user's
completion notice instead of monitoring training in this turn.

Completed and analyzed 2026-09-20. All six original tasks exit0 without
retry/resume, each with500 days,512000 critic/256000 actor updates and42
evaluations in6.23--6.30 hours. All day100 prefixes match exactly; subsequent
learning/evaluation alpha stays0.6. All96000 learning blocks are finite,
budgets are intact and extra penalties remain zero. Auto-sync returned
47785411 bytes of CSV/JSON; no checkpoints were downloaded. Frozen-runtime
analysis is saved in `../diagnostics/bus_alpha_freeze_20260920/summary.json`.

Primary frozen-minus-adaptive late-window gains for seeds7700/7701/7702:
headwise +1800/+11171/-4811, mean +2720 (paired seed SE4636);
clipped-min +11643/+5766/+4788, mean +7399 (SE2141). Approximate t95% intervals
with two degrees of freedom are [-17229,22669] and [-1811,16609]. Day500 gains
are positive in all six, but do not replace the prespecified window result.
Clipped-min late-minus-common-day100 remains negative for all three seeds:
mean -10399 frozen versus -17798 adaptive. Training days401--500 also favor
frozen clipped-min in all three (+6046/+3387/+4675). Early checkpoint means
at days125--200 favor adaptive alpha in both arms.

Late clipped-min sampled entropy rises1.124->2.363 and boundary-action
fraction falls.403->.218 under freezing; headwise shows the same directions.
This supports a partial temperature-related explanation within this recipe,
not complete prevention of degradation. Limitations: three reused seeds,
uncertain primary effect, mixed headwise gains, and no calibrated Q-bias or
RE-SAC/dual-uncertainty validation. The next candidate is a prespecified
fresh-seed paired confirmation of the clipped-min temperature intervention;
it is not launched or adopted as a method. No manuscript or training edits.

## Parallel temperature confirmation and Q calibration: 2026-09-20

Freeze `bus_alpha_confirmation.py`: seeds7800--7804, clipped-min SAC only,
adaptive versus freeze-after-day100 at each run's learned alpha. Ten500-day
runs retain the prior1024-updates/day recipe and all evaluation checkpoints.
Primary is frozen-minus-adaptive return over425/450/475/500, scenarios averaged
inside seed, then five-seed mean/SE and approximate df4 t95% interval. Compare
all non-timing prefix records through day100 before interpreting results.
No tuning, replacement seeds, peak selection or extensions. Total5120000
critic/2560000 actor updates and5430 full simulator days. Resumable latest
checkpoints stay remote; only CSV/JSON returns. Seeds7700--7702 are excluded
from this confirmation. Both treatments run concurrently, not sequentially.

In parallel freeze `bus_q_calibration.py`: reuse all12 completed day500
checkpoints (three seeds, two aggregations, two temperature treatments).
Hold each joint policy, alpha and checkpoint replay distribution fixed.
Define g_D(r) as the expected minibatch-normalized focal reward, including
the focal reward in the mean/sample std with2047 iid replay co-rewards.
Approximate using512 fixed independent co-batches, preserving raw-zero
terminal transformations; log the difference between two256-batch references.
Full vehicle-day soft returns exclude root entropy and learned bootstrap.
Eight fresh full days fit a residual correction; eight different days test
it. Select256 queries uniformly per day without consulting Q or returns.
Independent environment/action/query/co-batch streams use offsets10000--13016
within seed*100000, disjoint from the historical training/evaluation streams.

Compare unchanged native Q, same-head mean/min, a fit-only constant offset,
and a fixed ridge(.01) residual readout on128 frozen critic features. Only
fit days determine scaling/intercept/coefficients. No shared embedding,
policy or original critic updates occur. Primary: held-out signed Q-minus-MC
error and paired sampled-MSE changes (feature-minus-native/constant). First
average inside day/source/seed, then report the three seeds separately and
their mean/SE;192 full days and49152 queries total. Two GPU scheduler workers,
one per aggregation, resume at completed-day boundaries; checkpoints and
trajectory-feature caches stay remote. This tests checkpoint-frozen
policy-weighted calibration and a candidate critic repair, not historical
stationary Q*, online policy improvement or RE-SAC/dual uncertainty. A repair
must improve held-out prediction beyond constant shift before actor-level
testing. Preserve negative results. Deploy, verify startup, then stop without
waiting for either campaign. No manuscript or historical-trainer changes.

Deployed under `../diagnostics/bus_parallel_followup_20260920/`: t99634--t99641
train seeds7800--7803 on jtl110gpu (four/card, physical CPUs0--7);
t99642--t99643 train seed7804 on jtl110gpu2 (CPUs8/9), with Q workers
t99644--t99645 on CPUs10/11. The initial four/card scheduler gate was retained;
only four still-queued tasks were moved, with no cancellation or retry.
Seven new tests pass locally and on both servers, including CUDA readout
fitting, freeze-boundary pairing and exact interrupted-day continuation.
Startup confirms all12 CUDA PIDs, distinct within-host physical affinities
and no tracebacks; gpu1 training is past day5, gpu2 training has initialized,
and both Q workers have completed two days with saved progress. For GPU2,
150635412 bytes of calibration-only inputs (agent state and the complete
replay reward array, without replay states/actions/history) were exported
from GPU1 and streamed through SSH without local disk storage. Original
checkpoints remain unchanged; provenance is in the remote `inputs/manifest.json`.
Task specifications record the updated input paths; these inputs cannot
resume policy training. Result directories exclude checkpoints and caches.
Both campaigns are running; no performance or calibration conclusion yet.

Completed and analyzed 2026-09-21. All12 original tasks exit0 with server
success logs and no retries/resumes. Ten training runs finish500 days,
512000 critic/256000 actor updates and42 evaluations each; all160000 learning
blocks are finite and all five complete non-timing day100 prefixes match.
Q diagnostics finish192 full days/49152 queries. Two missing result folders
(t99635/t99640) were pulled as16363427 bytes of CSV/JSON; no checkpoints.
Frozen-runtime temperature summary and independently recomputed held-out Q
day statistics retain every seed and the original fit/test split.

Fresh-seed primary temperature gains (7800--7804):
+13982/+11046/+8524/+11706/-564, mean +8939 (SE2530), approximate df4
t95% interval [1915,15963]. Day500 alone is +10274 (SE3890), interval
[-527,21076]; it does not replace the primary. Late-minus-day100 means remain
-26970 adaptive and -18031 frozen. Freezing partially mitigates degradation,
not universally or completely. Summary: `bus_parallel_followup_20260920/temperature/summary.json`
under the diagnostics directory.

Q-minus-MC errors on held-out days9--16, in matched soft-return units:
SAC adaptive -146.77 (seed SE13.58), frozen -140.39 (SE7.03); each seed is
negative in both treatments. The frozen-feature correction gives +9.37 and
+5.94, respectively. Held-out sampled-MSE changes versus native are -23240
and -20461; versus fit-only constant offset they are -1492 (SE272) and -860
(SE303), with negative changes in all three seeds for each treatment.
These latter gains are6.43%/3.59% of constant-corrected sampled MSE; most
native-error reduction is already attainable by constant shift. Headwise
mean errors are positive (+23.63/+26.46), and correction beyond constant is
mixed (-153/+137 mean MSE change). Normalizer-half mean-return differences
for SAC are +.11/-1.70, much smaller than its measured negative bias.

This supports SAC underprediction and held-out critic repair for the fixed
checkpoint/replay objective, separately from the fresh temperature result.
Limitations: Q diagnostics reuse three training seeds, sampled MSE includes
Monte Carlo noise, and neither action-ranking improvement nor RE-SAC/dual
uncertainty is established. Next: prespecified full-state action contrasts
and realized bus returns to test decision relevance beyond constant shift.
No new jobs, coefficient changes or manuscript edits in this analysis turn.

## 2026-09-21: Frozen SAC Q-repair decision relevance

Before launch: reuse all six completed day500 SAC checkpoints (seeds7700--7702,
adaptive/frozen alpha), their existing fit-days1--8 ridge.01 corrections and
the same512 replay co-batches. No refitting, policy training or seed replacement.
For each source, two new frozen-policy prefix days provide full simulator
snapshots at the first decision at/after12000/24000/36000/45000s, smallest bus
ID on ties. Retain other pending actions and the whole simulator graph. The
candidates are the sorted unique union of 0/15/30/45/60s and the original
sampled actor action. Native min-Q, constant-corrected Q and feature-corrected
Q select before any continuations; constant must leave the native choice
unchanged. Only the focal root action changes, then the original joint actor.

Each context/action has16 paired suffix noise draws; MC soft/system selectors
use draws0--7 only and are assessed on8--15. All primary effects also use8--15.
The primary endpoints are feature-selected minus native-selected matched
soft return and raw whole-system suffix reward, averaged over draws/contexts
inside each seed, then three-seed mean/SE separately by temperature. Secondary:
feature versus original actor, action switches, and action-contrast squared
residual change anchored to actor. Preserve signed gains, unchanged choices
and every seed. Lower level-prediction MSE alone is not decision improvement;
consider a later paired policy-training test only if soft/system gains and
action-contrast error support it, not from a selected context or MC selector.

Streams are seed*100000 +20000/21000+prefix for prefix environment/actions,
+30000/40000+100*probe+replicate for suffix environment/actions. Trip action
streams are independent; the common future RNG coupling is not an assertion
that differing actions encounter identical events. Rewards use the previous
checkpoint-native mapping including transformed terminal zeros; omit root
entropy and any learned tail. Budget: six GPU tasks, eight contexts each,
at most4608 suffixes plus one exact replay smoke per task. Deduplicate actor
actions equal to grid points. Save progress after each completed branch;
regenerate and compare prefix probes on resume. Export CSV/JSON only.

Limitations: reused training seeds, correlated contexts within seed, finite
action candidates and one-action interventions. This tests a new residual
correction, not full-policy improvement or original RE-SAC/dual uncertainty.
No manuscript or historical training code changes. Deploy via scheduler and
verify startup only; do not wait for experiment completion.

Deployed: `../diagnostics/bus_q_decisions_20260921/`, t99766--t99771,
ordered adaptive/frozen within seeds7700/7701/7702. All on jtl110gpu, three
per card, distinct physical CPU affinities0--5; live CUDA processes each use
324MiB. Thirteen local tests and five remote decision tests pass, including
CUDA branch interruption/resume and CSV-only aggregation. All six real-state
same-action replay smokes pass exactly, with two saved branches each and no
tracebacks at startup inspection. Checkpoints stay remote; scheduler result
directories contain only small result tables/JSON. Jobs are running; no
decision-effect conclusion yet. Stop after deployment as requested.

Completed and analyzed 2026-09-21. All six original tasks exit0, retry_count0,
one fresh start each and no server tracebacks. Every run finishes8 contexts
and768 branches (4608 total, plus six replay smokes), in7719--7926 seconds.
All824181 bytes of small result files were already automatically synchronized;
no checkpoint download. Frozen-runtime aggregation verifies complete unique
action/noise grids and finite outcomes. Direct CSV lookup reproduces all seed
primary effects and contrast-MSE differences; all six correction coefficient
sets equal their prelaunch copies, and constant offsets never change choices.
Aggregate: `bus_q_decisions_20260921/summary.json` under diagnostics.

Feature-selected minus native-selected confirmation means (seed SE):
adaptive soft +0.755 (0.476), raw system +8.590 (8.556);
frozen soft -4.808 (5.108), raw system -64.597 (74.650).
Per-seed raw-system effects7700/7701/7702 are +25.383/+3.043/-2.655 adaptive
and +73.969/-182.022/-85.737 frozen. Approximate df2 t95% intervals for the
primary soft/system means are [-1.293,2.802]/[-28.223,45.404] adaptive and
[-26.785,17.168]/[-385.791,256.598] frozen; none establishes a stable benefit.
Feature-minus-original-actor raw gains are +6.711 (SE7.651) adaptive and
-72.354 (SE67.872) frozen. Decisions change13/24 and18/24 contexts, respectively.

Action-contrast sampled MSE, anchored to the original actor action, increases
from345.074 to664.997 adaptive and536.748 to775.557 frozen. Differences are
+319.922 (SE164.510; all three seeds positive) and +238.808 (SE392.735; two
of three positive). Thus better on-policy absolute-Q calibration does not
transfer to accurate action contrasts under this finite-grid intervention.
The held-out MC-soft selector gives raw-system +49.851 (SE20.777) adaptive
and +115.029 (SE79.941) frozen, but uses simulator continuations and is not
the learned correction or a demonstrated deployable policy improvement.

Conclusion: this frozen residual readout does not pass the prespecified
decision-relevance criterion; do not promote it into policy training. The
earlier SAC level-underprediction and fresh alpha-freeze results remain
separate. Candidate next step: fit within-state action advantages, separating
state offsets from action differences, then test on new full-state contexts
and independent future noise before actor training. Existing outcomes are
now inspected and cannot be relabeled as untouched confirmation for a new
fit. Three reused seeds and one-action interventions limit generalization;
neither reliable policy improvement nor original RE-SAC is established.
No new jobs, coefficient changes or manuscript edits in this analysis turn.

## 2026-09-21: Within-context advantage calibration, fresh-context pilot

Freeze before launch. For each of the six day500 SAC sources (7700--7702,
adaptive/frozen), use every old decision context and all16 suffix replicates
from `bus_q_decisions_20260921` as fit/development data only. Previously
inspected confirmation outcomes are no longer held out. Compute each
candidate's mean matched soft return and MC-minus-native residual. No new
training-day simulation, checkpoint changes, candidate selection or tuning.

Fit two matched ridge.01 readouts on the same frozen128 critic features:
level uses globally centered features/residuals with a free intercept;
advantage subtracts each context's candidate mean from features/residuals,
with no intercept. Both use the same fit-only pooled feature scales and
equal context weights (each candidate weight1/(contexts*candidates)). This
cancels action-independent state offsets in the advantage fitting objective.
At prediction, center the advantage correction over that context's candidate
set; its mean Q level is unchanged. The comparison tests within-context versus
global residual fitting, not a standalone solution to absolute Q underprediction.
Keep native Q, original failed level readout, and original actor as controls.

New evaluation: four fresh frozen-policy prefix days, first decision at/after
12000/24000/36000/45000s with smallest bus ID tie break,16 contexts/source.
Actions remain the sorted unique union of0/15/30/45/60s and sampled actor action.
All learned choices are fixed before any new outcomes; model coefficients are
saved before the first test prefix and never refitted, including on resume.
Full-state one-action interventions and matched reward/entropy/terminal
accounting are unchanged. Future draws0--7 select MC reference actions only;
draws8--15 evaluate all reported gains and action-contrast errors.

Primary: advantage-minus-native soft and raw-system gains, and actor-anchored
action-contrast sampled MSE change. Average noise/context effects within seed
then report seed mean/SE separately by temperature. Matched-level comparison
is the main explanatory control; old level, original actor, action switches
and MC selectors are secondary. Preserve every seed, unchanged choice and
negative gain; no promotion based on fit MSE, one selected context or the
best temperature. Any favorable result remains a pilot requiring fresh-seed
confirmation before an algorithm-level policy improvement claim.

Streams: seed*100000 +50000/51000+prefix for new prefix environment/actions,
+60000/70000+100*probe+replicate for suffixes. Retain +12000 co-batches solely
to keep the checkpoint-native objective fixed. No overlap with old fit
contexts or future draws. Six GPU tasks, at most9216 suffixes plus six exact
replay smokes. Save each branch; regenerate and compare contexts on resume.
GPU1, three tasks/card, one distinct physical CPU/task. Small CSV/JSON outputs
only; inputs and progress checkpoints stay remote. No manuscript edits.

Limitations: eight fit contexts/source,128 frozen features, three reused
training seeds and finite action candidates. This is advantage calibration,
not a validated online policy or original RE-SAC/dual-uncertainty method.
Deploy, verify CUDA execution/checkpoints/smoke, then stop without waiting.

Deployed under `../diagnostics/bus_advantage_calibration_20260921/`:
t99801--t99806, adaptive/frozen in seed7700/7701/7702 order, all jtl110gpu,
three/card, physical CPU0--5. Seven tests pass locally and on the server,
covering state-offset cancellation, matched ridge objectives, disjoint
streams, signed confirmation, small-file aggregation and exact CUDA resume
without refitting. All six fit8 old contexts/48 actions and then start new
contexts; each has passed real-state exact replay smoke and saved its first
branch. Six live CUDA processes use360MiB each; no tracebacks. Parameters and
progress stay remote; scheduler will return only small result files. Stop
after startup; no fresh-context outcome conclusion or manuscript changes.

Completed and analyzed: all six original tasks exit0 with no retries or server
tracebacks;16 contexts/1536 branches each,9216 total plus six replay smokes.
Runtime16425--16879s. All1661459 bytes of small result files were already
synced; no checkpoints downloaded. Frozen-runtime aggregation and independent
CSV lookup agree on primary, matched-level/actor contrasts and all readout
MSEs. Aggregate: `bus_advantage_calibration_20260921/summary.json`.

Advantage-minus-native soft/system means (seed SE): adaptive -0.859 (2.059)
and -14.727 (31.709); frozen +0.194 (.954) and +9.116 (2.089). Frozen raw
gains are +4.987/+11.734/+10.626, with approximate unadjusted df2 t95%
interval [.127,18.104]; its soft interval is [-3.912,4.299]. Against the
original actor, frozen system +4.377 (2.323), interval [-5.619,14.373].
Action-contrast MSE native/advantage/matched-level: adaptive724.113/695.605/
1285.461; frozen595.202/3582.939/6799.369. Advantage beats matched-level MSE
in all six sources, but frozen MSE exceeds native in all three sources.
Advantage changes27/48 adaptive and25/48 frozen choices. MC-system selector
raw gains are +66.160 (4.487) adaptive and +40.189 (10.993) frozen; these
use simulation at selection time and are not gains from the learned repair.

Post-outcome diagnosis, not a changed analysis: frozen7702 probes6/7 dominate
its error. Feature26 fit scale .004535, raw coefficient -297.409, maximum
within-fit-context action span .031754; new probe7 action span2.470213 gives
a +734.663 correction contribution. Net action gap grows from95.346 to725.992
and probe MSE from360.065 to130804.342. A read-only source-checkpoint forward
pass reproduces this finite extrapolation, not a NaN or incomplete-run error.
That probe keeps the native action, so large prediction error is not itself
proof of reward loss there. Every context remains in the primary result.

Conclusion: within-context fitting is better than matched absolute fitting,
but does not yet support reliable control improvement over native Q. Retain
the small frozen raw-reward signal without selecting it over the other
endpoints. Next address feature-scaling/extrapolation amplification and test
by whole-prefix-day holdout on development data before another fresh-context
campaign. No online actor training, new tasks or manuscript edits this turn.

## 2026-09-21: Shared-scale advantage, whole-day development holdout

Freeze before execution. Keep the previous per-feature readout unchanged.
New candidate: a single fit-only scalar scale, squared equal to the average
of the128 pooled feature variances, with equal weight per context. Center
features/residuals within context, no intercept, ridge.01 unchanged. This
penalizes raw feature coefficients isotropically, unlike coordinate scaling
which weakly penalizes coefficients of low-variance features. No feature
exclusion, clipping, ridge sweep, policy update or source-checkpoint change.
This removes that unequal-penalty mechanism, not all possible extrapolation.

Reuse both inspected campaigns (two old and four newer prefix days/source,
24 contexts and2304 branches/source). Six GPU workers each extract frozen
features and perform seven paired fits: chronological old2/new4 days, then
six leave-one-prefix-day-out folds fitting five days and holding out the
sixth. All four contexts and all suffix replicates of a held-out day are
excluded from fitting and scale estimation. Each context occurs once in
the day-holdout view; keep that view separate from chronological replay.
Training means use all16 draws of training contexts; held-out scores use
draws8--15 as before. Native Q and identical-data per-feature advantage are
controls. Report shared-minus-native/standardized/actor soft and system
effects, actor-anchored action MSE, choices, raw coefficient norm and maximum
context error. Aggregate contexts inside seed, then three seeds per temperature;
do not treat folds or branches as independent training seeds.

Budget: six tasks,42 paired fit/evaluation splits, zero new simulator rollouts.
Save feature caches and fold progress remotely; return small JSON only.
All outcomes were inspected before designing this change: whole-day exclusion
prevents row leakage but cannot restore confirmatory status. Retain known
extrapolation contexts and negative effects. Judge scaling repair separately
from control performance; no policy-training promotion from a smaller Q span
alone. No manuscript edits; stop after dispatch verification, not a new long
training campaign.

Executed via scheduler t99869--t99874 on jtl110gpu, three/card, CPU0--5.
All six were already complete at startup inspection (2.87--3.25s each),
exit0/no retries, seven splits each, zero new rollouts. Thirteen local tests
and six remote tests pass, including day isolation, raw-space penalty
equivalence, CUDA extraction and exact fold resume. Retrieved965726 bytes of
JSON only. Chronological per-feature predictions reproduce all prior values
exactly (max difference0); every fold excludes all held-out prefix days.
Aggregate: `bus_advantage_day_holdout_20260921/summary.json`.

Chronological shared-scale action MSE adaptive/frozen628.919/721.069 versus
per-feature695.605/3582.939 and native724.113/595.202. On known frozen7702
new/7, action span725.992 falls to26.511 and MSE130804.342 to3325.026, still
worse than native360.065. In whole-day holdout for that same context, span
1722.084 falls to96.250 and MSE372983.068 to275.785. The unequal-penalty
amplification is substantially reduced; smaller span alone is not success.

Whole-day holdout shared-minus-native soft/system means (seed SE): adaptive
-2.013 (.547)/-22.860 (11.934), frozen +1.007 (.580)/+18.090 (5.436).
Native/shared/per-feature MSE is597.767/611.707/705.567 adaptive and
575.717/543.009/5909.799 frozen. Frozen raw gains7700--7702 are
+20.073/+7.841/+26.356; MSE changes -1.019/-13.450/-83.654. Frozen gains
against the original actor are soft +.691 (.420), system +12.345 (2.806).
Chronological system effects remain -42.097 (20.846) adaptive and +6.294
(8.875) frozen; keep both views, not only the favorable larger-fit view.

Conclusion: scaling repair reduces the observed extrapolation failure, but
does not establish general control improvement. Frozen-temperature whole-day
development effects are consistently positive across three seeds; adaptive
effects are not. These inspected-data diagnostics support a candidate for
fresh-context confirmation, not policy-training promotion or RE-SAC claims.
No further jobs, source-policy changes or manuscript edits this turn.

## 2026-09-21: Fresh-context shared-scale confirmation

Freeze before deployment. Seeds7700--7702, adaptive and frozen temperature,
all six completed day500 SAC sources. Fit the shared-scale advantage readout
once on all24 inspected contexts/source (six prefix days, all16 future draws),
with ridge.01, equal context/action weighting and no intercept. Matched
per-coordinate-standardized advantage uses exactly the same data and ridge;
native Q and original actor are controls. Save both models before generating
new prefixes and reuse them on resume. No tuning or test-dependent exclusions.

Each source: four new prefix days, query times12000/24000/36000/45000,
actions0/15/30/45/60 plus the original actor action,16 paired future draws.
Offsets from seed*100000: prefix environment80000, prefix action81000,
suffix environment85000+100*probe+replicate, suffix action90000+100*probe+replicate.
These are disjoint from both development campaigns and earlier source/MC
streams; keep the native reward co-batch mapping at12000 intentionally.
Complete simulator snapshots and unchanged future joint actor; terminal
rewards transformed, no root entropy and no learned tail, as previously.

Primary: shared-minus-native soft-return gain on draws8--15, separately for
each temperature. Report raw-system gain and actor-anchored action-contrast
MSE alongside it; no promotion based only on reduced MSE. Secondary: matched
standardized/actor comparisons, switches, spans and coefficient norms.
MC selectors choose using draws0--7, then score on8--15. Aggregate contexts
within each seed, report all three seed effects and mean/SE; no pooling
temperatures. Negative adaptive effects remain part of the result.

Budget: six scheduler GPU jobs, three/card on jtl110gpu, distinct physical
cores; at most9216 branches plus six exact-repeat smoke branches. Save
progress after each branch, keep checkpoints remote, autosync small result
folders only. Confirm launch/CUDA/pinning/checkpoint/smoke, then stop waiting.
Fresh contexts/noise confirm only finite-grid single-action generalization
for reused training seeds, not whole-policy gain or original RE-SAC. No
policy update or manuscript changes.

Deployed scheduler t99875--t99880 on jtl110gpu, three/card, physical CPU0--5.
All six running with zero retries at startup inspection; CUDA3080Ti,
24 fit contexts/2304 fit branches each, fitted models and branch checkpoints
saved. All six real-simulator exact-repeat smoke checks passed. Twelve local
tests and six frozen-runtime remote tests passed, including exact interrupted
resume without refitting. Stop after startup verification; results pending.

Completion audit: t99875--t99880 all done, exit0, no retries, 4.799--4.902h.
All96 contexts/9216 branches complete; frozen-runtime aggregation validates
the small outputs. Scheduler already synced1543478 bytes, no checkpoints
downloaded. Aggregate: `bus_shared_scale_confirmation_20260921/summary.json`.
Shared-minus-native soft/system mean (seed SE): adaptive -1.280 (.350)/
-16.145 (5.248), frozen -1.206 (1.057)/-3.687 (6.546). Primary soft gains
are negative in all three seeds of each temperature; frozen system effects
are -15.574/+7.006/-2.492. Native/shared MSE:683.669/280.207 adaptive,
1188.603/863.084 frozen. These mean MSE decreases are not control gains.
Adaptive switches19/48 contexts (soft:8 positive,7 negative,4 zero); frozen
switches26/48 (12 positive,12 negative,2 zero). Loss magnitude outweighs
beneficial switches. Adaptive7700/probe2 contributes approximately95% of
the adaptive mean MSE decrease. Frozen7702/probes5,6 incur soft losses
-40.757/-40.369, retained with all other outcomes. The development-only
frozen gain did not replicate. This readout fails fresh-context control
confirmation; do not promote it to actor training. Next investigate harmful
action-ranking changes using existing branches, not another same-readout
training expansion. No new tasks or manuscript changes in this audit.

## 2026-09-21: Retrospective action-switch diagnosis

`bus_action_switch_diagnostics.py` consumes only the six existing small
result folders, retaining all96 contexts and the original confirmation
endpoints. Six tests pass; full-sequence attribution reproduces the saved
gains. No simulation, source edits, new fitting or checkpoint transfers.
Artifacts: `bus_action_switch_diagnostics_20260921/{summary,contexts}.json`.
Of7 negative adaptive switches,6 have negative soft gains in both independent
halves; frozen11/12. Their all-context soft contributions are -2.054/-2.549
respectively, versus +.744/+.974 from switches positive in both halves.
Frozen7702/probe6 predicts a +8.299 soft gap after a +9.785 correction of the
native gap -1.486; measured half means are -44.433/-40.369 (paired MC SE
4.648/5.159). Large losses recur across noise halves, not only sign flips.
Adaptive switch-pair MSE rises116.311->133.074 while all-action MSE falls;
95.3% of the latter net decrease comes from one context (frozen76.9%).
The paired MSE decomposition separates squared mean error and sampled
variance; the shared-target variance term cancels in method differences.

Simulator-only diagnostic veto: choose the fixed shared action only if its
paired soft advantage on draws0--7 is positive, otherwise retain native;
score on8--15 with every context retained. Adaptive/frozen soft gains are
+.718 (.490)/+.966 (.691), system +10.732 (7.875)/+20.533 (17.434), all three
seed means positive in each group. It reads extra simulator continuations,
showing useful pair-gain information in those outcomes, not a deployable
policy, proof of learnability from critic features, or restored confirmation.
Half-sign agreement is descriptive, not a significance test.
Next: develop action-pair gain prediction and selective switching with native
fallback, assess by whole-prefix-day exclusion before another fresh test.
Do not select candidates using global Q-MSE alone. No new jobs or paper edits.

## 2026-09-21: Native-anchored pair gain and selective switching

Freeze before execution. Reuse all three inspected campaigns:40 contexts,
ten prefix days/source,3840 existing branches/source; no new simulation.
For each of six day500 sources, perform ten outer whole-day folds, fitting
nine days and assessing all four contexts of the tenth once. Fixed ridge.01,
128 frozen features, shared fit-only RMS scaling, equal context and then
non-anchor action weight. Regress MC-minus-native gain residual relative to
the native argmax, not all action pairs (whose unweighted squared loss would
reproduce context centering). No intercept; native gain remains exactly zero.
Same-data centered shared-scale readout and ungated pair readout are controls.

Each outer fit calibrates an empirical switching margin using nine inner
whole-day exclusions. Inner models use eight days and select their candidate
before reading the ninth day's returns. Record predicted-minus-observed
soft gain versus native, take the nonnegative maximum over the four queries
per day, then the fixed nearest-rank80th percentile over nine day maxima.
Refit pair readout on all nine outer-training days; switch only if predicted
gain exceeds that margin, otherwise retain native. No outer-test labels,
feature scaling or runtime simulation enters this decision. No sweep.
This cross-fitted margin is empirical, without a claimed coverage guarantee.

Primary: selective/paired/shared-minus-native soft and raw-system gains on
draws8--15; selective-minus-paired/actor secondary. Report switches, harmful
switches, all seed effects and zero-switch cases alongside MSE. Keep all
contexts and temperatures; zero-switch fallback is not demonstrated gain.
Budget: six scheduler GPU workers,60 outer folds,660 small ridge solves,
zero new rollouts. Cache features and fold checkpoints remotely; sync only
small JSON outputs. These are inspected-data development results, not new
confirmation or dual-uncertainty/RE-SAC validation. No paper edits.

Executed scheduler t99918--t99923 on jtl110gpu GPU0/1, three/card, CPU0--5.
All60 outer folds completed at startup inspection (6.167--6.582s/source),
CUDA enabled, zero new rollouts; six local and six remote tests passed,
including inner/outer exclusion, strict fallback and exact fold resume.
Retrieved1574540 bytes of JSON only. Aggregate:
`bus_pair_gain_holdout_20260921/summary.json`.
Ungated pair soft/system gain vs native: adaptive -1.899 (.881)/-18.276
(9.129), frozen +.407 (1.214)/+13.001 (13.961), with mixed frozen seeds.
Selective adaptive rejects every switch (0/120), hence exactly zero gain.
Selective frozen switches5/120, two harmful soft switches; soft/system gain
-.122 (.312)/-2.668 (3.629). Frozen per-seed soft gains+.410/-.106/-.671.
Frozen7702 recent/6 still passes the gate: predicted gain21.653 > margin
10.741 but actual gain-40.369. The unchanged shared-scale control gives
adaptive/frozen soft -.994 (.349)/+.824 (1.329); pair anchoring is not a
demonstrated improvement. Retain all controls and zero-switch cases.
Conclusion: native-anchored linear correction plus a global empirical margin
does not resolve action-ranking error. Do not tune the gate against these
results or promote to actor training. Next compare predictive information
in frozen critic features versus observable raw bus state/action features,
under the same whole-day exclusions, to separate representation from fitting
and coverage limitations. No new simulation or manuscript edits this turn.

## 2026-09-21: Matched observable-state representation comparison

Freeze before execution. Reuse the completed pair-gain caches and recorded
28-dimensional observations,40 contexts/ten prefix days per source. Three
representations under identical native-anchored ridge.01 and nested-day
empirical80th-percentile switching margin: frozen critic128; action-only
[u,u^2], u=a/60; observable [u*x,u^2*x], where x contains1, categorical
one-hot bus/station/hour/direction, the two headways and demand-plus-travel
proxy divided by360, and21 route speeds divided by10. Interactions retain
state-dependent action contrasts, unlike additive state/action linear fits.
Each inner/outer training split builds its own category vocabulary. Unseen
categories produce zero indicators within that block; values are not clipped.
The existing pooled RMS feature scaling is still fitted only on training data.
No exact query time, hidden simulator state or outcome enters the features.

Use the identical ten whole-day outer folds and nine inner exclusions;
both ungated and selective policies reported, always retaining native and
actor comparisons. Critic gains/margins/choices must reproduce the previous
run. Report paired observable-minus-critic/action-only gains, switches,
harmful switches, MSE, and descriptive unseen-category/outside-training-range
counts. Coverage flags never affect selection or exclusions. Average within
seed then mean/SE over all three seeds, separately by temperature. No sweeps.
Six scheduler GPU workers,180 representation-folds/1800 small ridge solves,
zero new simulations. Keep source caches/checkpoints remote, sync JSON only.
Inspected-data development; different map dimensions and inductive biases
preclude an information-theoretic claim about feature sufficiency. No actor
training, fresh confirmation, original RE-SAC claim or manuscript edits.

Executed scheduler t99924--t99929 on jtl110gpu GPU0/1, three/card, CPU0--5.
All180 representation-folds completed at startup inspection,9.907--10.132s
per source; six local and six remote tests passed, including split-local
encoders, held-out label exclusion, exact resume and critic regression.
Critic predictions reproduce the previous experiment exactly (max difference0).
Only small JSON results synchronized; no checkpoint download or new rollout.
Aggregate: `bus_representation_holdout_20260921/summary.json`.

Observable ungated soft/system gains vs native, mean (seed SE): adaptive
-1.387 (2.017)/-18.866 (27.014); frozen -5.783 (.967)/-72.443 (14.735),
with all three frozen seeds negative. Observable selective gains: adaptive
+.364 (.207)/+6.169 (3.569), four switches/120, no negative soft switches;
frozen -.043 (.043)/-1.306 (1.306), two switches/120, one negative.
Selective soft seed effects are adaptive0/+.376/+.717 and frozen0/0/-.130.
Action-only selective adaptive/frozen soft gains are -.027 (.027)/0, with
one/zero switches; critic selective remains0/-.122 (.312), zero/five switches.
Observable action-contrast MSE is585.525/1145.368 versus critic452.749/592.944.
Observable dimensions vary112--130 over outer fits. Unseen-category contexts
number21/120 adaptive and28/120 frozen; outside-numeric-range55/120 and61/120.
These are overlapping descriptive flags, not uncertainty calibration.

Frozen7702 recent/6 now selects60 instead of45 and gains+16.146 soft, but
recent/7 predicts+18.002 > margin14.714 and loses-21.340 soft/-295.572 raw.
The latter has neither unseen categories nor out-of-training-range numeric
coordinates; simple range exclusion would not catch this error.
Replacing the critic map with this fixed observable map does not establish
reliable control improvement. The sparse adaptive positive result remains
development evidence; retain both temperatures, all seeds and all controls.
Next diagnose action-curve model mismatch and sensitivity to training-day
composition using existing branches before choosing another estimator or
collecting fresh data. Do not retune this gate or expand actor training.

## 2026-09-21: Action-curve mismatch and training-day sensitivity

Freeze before execution. Reuse representation caches, all40 contexts/ten
days per source, three seeds and both temperatures. No new simulation.
At each context, anchor to native Q argmax and project MC-minus-native
action gains onto [u-u0,u^2-u0^2], u=a/60, equal non-anchor action weight.
The mean product of off-basis residuals from draws0--7 and8--15 estimates
squared nonquadratic mean-curve error under independent suffix halves;
retain negative estimates. For saved observable/action-only predictions,
split the mean cross-product of prediction errors exactly into this term
and in-basis coefficient error. This separates a fixed action-basis limit
from the remaining learned-coefficient error, not sources of uncertainty.
Also fit the per-context quadratic residual on0--7 alone, choose its best
action and evaluate on8--15; compare unrestricted first-half MC choice,
native and actor. These MC-based choices use extra simulator information.

Within every outer fold, delete each of its nine training days in turn,
refitting all three maps on eight days with their own scales/encoders and
unchanged ridge.01. Predict the still-excluded outer day. Keep the saved
full-fit empirical margin fixed to isolate prediction sensitivity from
margin recalibration. Report any argmax/selective-choice flip, fraction
of nine choices changed, range of predicted gain for the original best
candidate, and unchanged harmful switches. Do not build an ensemble,
select a deletion, add a stability veto or tune a threshold from this.
Saved full-fit predictions/choices must reproduce before aggregation.

Six scheduler GPU workers,1620 small ridge fits,60 outer folds,240 contexts.
Cache progress remotely per fold; synchronize small JSON only. Aggregate
within seed then mean/SE over three seeds separately by temperature.
Deletion variants are correlated sensitivity probes, not independent
replications or calibrated confidence intervals. This inspected-data
diagnosis cannot establish hidden-state insufficiency or a causal error
attribution. No new actor training, RE-SAC validation or manuscript edits.

Executed t99930--t99935 on jtl110gpu GPUs0/1, CPU0--5, three/card. All60
outer folds/1620 deletion fits completed at startup inspection in8.663--9.146s
per source. Six local and six remote tests passed; full-fit control prediction
difference is exactly0. Retrieved3549244 bytes of JSON, no checkpoints.
Aggregate: `bus_curve_stability_20260921/summary.json`.
Adaptive/frozen observable error cross-products split into nonquadratic
2.217 (.249)/1.271 (.203) and coefficient421.745 (284.160)/838.534 (134.899).
These native-anchored mean-gain errors differ from prior actor-anchored
single-draw MSE. Positive cohort estimates do not make each context positive.
First-half per-context quadratic MC selection gains+3.097 (.228)/+2.878
(.745) soft and+57.609 (8.135)/+54.166 (23.902) raw on the second half,
all six seed means positive. Unrestricted MC selection gives+3.186 (.274)/
+3.330 (1.025) soft. Both read simulator outcomes at the query context.
Observable ungated choices flip under at least one training-day deletion
in77/120 adaptive and85/120 frozen contexts; critic65/120 and60/120.
None of the six original observable selective switches survive every
deletion. One of the two harmful frozen critic switches does survive all
deletions, so stability is not equivalent to correct ranking.
Frozen7702 recent/7 has nonquadratic error.302 vs coefficient error504.764;
its predicted60-second gain varies-2.212 to+27.749 across deletions, while
both measured halves give negative gains (-23.039/-21.340). A per-context
quadratic fit chooses0 seconds instead, second-half soft gain+.694.
Conclusion: on these data, the fixed quadratic action basis is not the main
aggregate limitation; cross-state coefficient prediction is inaccurate and
sample-sensitive. Next test a lower-dimensional physically specified bus
state map under unchanged whole-day exclusion/ridge/margin and retained
critic/action-only/observable controls. Do not infer that simplification is
already effective, retune this gate or promote simulator-only gains to a
deployable RE-SAC result. No additional rollout or manuscript edits.

## 2026-09-21: Eight-feature headway/demand residual correction

Freeze before execution. The only new map is `headway_demand`:
x=[1,forward_headway/360,backward_headway/360,demand_travel_proxy/360],
phi=[u*x,u^2*x], u=action/60, exactly8 columns. The recorded proxy is
1.5*next_station_waiting_passengers + current_route.distance/speed_limit
(env_original/bus.py), not an on-board load or isolated passenger count.
No vehicle/station/hour/direction category, full route-speed vector,
hidden simulator field, fitted encoder or clipping enters this residual
map. Native Q still uses the full original observation. Keep the previous
critic/action-only/observable controls, action grid, native anchoring,
ridge.01, pooled fit-only RMS scaling, equal context/candidate weights,
and nested whole-day80th-percentile margin rule. Margins are fitted per
map from training days, not copied numerically from another map.

Use all40 contexts/source, ten outer folds, nine inner exclusions, three
seeds/temperature. Repeat delete-training-day diagnostics for all four maps
at their own fixed full-fit margins, without adding a stability veto.
Saved old-map predictions/margins/choices/effects must reproduce exactly
up to existing numerical tolerances. Report gains vs native/actor and
paired new-map-minus-each-control gains, MSE, switches/harmful switches,
coefficient-error cross-products and choice sensitivity. No feature sweep.
The old runners' default representation lists remain unchanged.

Six scheduler GPU workers,4560 small ridge solves, zero new simulations.
Remote fold checkpoints/results; synchronize only config/runtime/completion
and run/aggregate summary JSON, not folds or checkpoints. Same ridge/scale
procedure across feature maps is not identical functional regularization.
This is inspected-data development, not a sufficiency test, fresh
confirmation or original RE-SAC validation. No manuscript edits.

Executed t99938--t99943 on jtl110gpu GPUs0/1, three/card, CPU0--5. All60
outer folds/4560 fits completed in23.477--24.731s per source. The local16
tests passed; remote output-directory test caught a missing details mkdir,
fixed before submission, then all four affected tests passed alongside the
twelve passing control regressions. All three real-data controls reproduce
exactly (max gain difference0). Only55630 bytes of run JSON retrieved;
full fold details and resumable checkpoints remain remote.
Aggregate: `bus_lowdim_holdout_20260921/summary.json`.

Headway/demand ungated soft gains vs native are adaptive -.346 (.033),
all three seeds negative, and frozen -.205 (1.068), one seed positive.
Raw-system gains+1.880 (4.949)/+.936 (22.199) are mixed across seeds.
Selective soft gains are -.218 (.218)/-.314 (.209); raw -3.518 (3.518)/
-1.566 (1.011). Adaptive switches5/120 (two negative soft); frozen3/120,
all three negative. Selective soft seed effects0/-.653/0 and-.709/0/-.232.
Coefficient-error cross-products fall421.745->277.504 adaptive and
838.534->375.878 frozen compared with the full observable map. Ungated
choice-flip contexts fall77->41 and85->52 out of120. Thus reduced model
complexity helps these diagnostics but does not establish control gain.
Retain this negative outcome; neither MSE nor stability substitutes for
better decisions. Next hold the model fixed and measure whole-day sample
size learning curves on existing data to assess whether further independent
training days are a justified next investment. No threshold retuning,
feature sweep, fresh confirmation or actor-training expansion this turn.

## 2026-09-21: Fixed-model whole-day sample-size learning curves

Freeze before execution. Retain the8-feature headway/demand map, ridge.01,
native residual anchoring, equal context/candidate weights and fit-only RMS
scale. In each of ten outer-day folds, vary training days m=4,6,8,9 (16,24,
32,36 contexts). Shuffle sorted nine eligible IDs with RandomState(20260921
+canonical outer-day index), identically across seeds/temperatures. Nine
circular rotations give nested first4/6/8-day subsets: each eligible day
occurs exactly m times at budget m. Use all nine days once. No outcome-based
subset selection; nine balanced subsets are not exhaustive combinations.

For every subset, fit its empirical80th-percentile margin using m inner-day
exclusions and m-1-day fits, then refit on m days. No full-budget margin or
feature statistics may enter a smaller budget. The same four held-out-day
queries are scored on draws8--15 at every budget, with all outcomes retained.
Primary learning curves: ungated soft/system gain vs native and coefficient
error cross-product. Secondary: selective gain, actor comparisons, MSE,
switch/harm rates and margin. Gated curves also vary calibration sample
size/order statistic; they do not isolate fitted-readout changes alone.

Average outcomes over subset policies, not predictions or voted actions,
then all40 contexts within seed; mean/SE across three seeds separately by
temperature. Report all four sizes and paired9-minus-4/6/8 differences.
Overlapping folds/subsets are not independent replications. Nine-day gains,
margin, decisions/effects and coefficient errors must reproduce the lowdim
source. Six GPU workers,1680 subset fits with nested calibration,11940 total
small ridge solves; zero new rollouts. Keep fold detail/checkpoints remote,
sync only config/runtime/completion and summary JSON. No best-size choice,
extrapolation beyond9 days, fresh confirmation, actor training or paper edits.

Deployed t99944--t99949 through scheduler to jtl110gpu GPU0/1, three/card,
CPU0--5. Five local tests (40.982s) and five remote tests (30.511s) passed,
covering balanced nested subsets, actual inner/outer day budgets, exclusion
of held-out labels, nine-day control reproduction, averaging/subset integrity,
source loading, exact interrupted resume and summary-only aggregation.
Startup inspection: all six on CUDA/RTX3080Ti, distinct CPU affinities0--5,
nonempty checkpoints,3/10 outer folds complete. No result interpretation yet;
return after deployment as requested rather than wait for experiment finish.
Frozen task/runtime root: `bus_day_learning_curve_20260921`.

Completed all six tasks with exit0/retry0,44.205--45.412s each. Nine-day
control predictions reproduce exactly. Only small run JSON collected;
aggregate `bus_day_learning_curve_20260921/summary.json`. At4/6/8/9 days,
adaptive ungated soft gains are-.479/-.338/-.276/-.346; frozen-2.223/-1.135/
-.557/-.205. Coefficient errors decline342.671/306.681/283.049/277.504 and
489.582/427.072/387.660/375.878, decreasing at every step for all six seeds.
Nine-minus-four ungated soft change: adaptive+.133 (SE.166), frozen+2.018
(1.193); raw+4.511 (5.357)/+27.589 (14.706). Frozen raw change is positive
in all three seeds, but soft change is negative in7702. Selective gains
remain negative at all four sizes in both groups. More data improves this
prediction diagnostic and part of frozen control performance, not reliable
native-relative benefit. This supports only a bounded fresh-data increment
test, not extrapolation or broad experiment expansion.

## 2026-09-21: Bounded fresh-day training-data increment

Freeze before simulation. Six unchanged SAC day500 sources, seeds7700--7702,
adaptive/frozen temperature. Fit and save the8-feature headway/demand base
readout on all ten inspected days (40 contexts). Generate four independent
new training days with four existing query times/day, finite grid plus actor,
16 paired suffix draws per action. Then fit/save expanded readout on14 days
(56 contexts), before generating four disjoint new test prefix days. Both
models fixed throughout testing. Ridge.01, fit-only scale, native anchoring,
weights and empirical80th-percentile inner-day margin rule unchanged. Primary
is ungated14-minus-10 soft/raw gain on the same fresh test contexts, with both
versus native required. Selective/actor comparisons, MSE and switches/harm
are secondary; margin sample size changes in the selective comparison.

Fresh stream namespace1600000000+(seed-7700)*100000: prefix environment+prefix,
prefix actor+1000+prefix, suffix environment+10000+100*probe+replicate, suffix
actor+20000+100*probe+replicate. Original co-batch seed retained for consistent
reward units. No hidden-state truncation, learned tail, root entropy, actor
updates or refitting after test starts. Test metrics use draws8--15 only;
all contexts/seeds and both temperature groups retained, seed-level mean/SE.
This is a finite-grid one-action data-increment experiment, not full-policy
or original RE-SAC validation, and reuses the three trained seeds.

Budget: six scheduler GPU tasks,32 new contexts/source (16 train+16 test),
at most18432 action continuations plus six exact-repeat smoke branches.
Expected hour-scale simulation. Atomic progress after every action branch;
skip completed independent prefixes on resume, reproduce unfinished-prefix
predictions, never refit already saved models. Full branch CSV/probe JSON and
checkpoints remain remote; scheduler syncs only small model/progress/summary
files. Stop after deployment/startup verification; no paper changes.

Deployed t99950--t99955 through scheduler to jtl110gpu GPU0/1, three/card,
CPU0--5. Four local tests (3.375s) and four remote tests (4.581s) passed:
fresh RNG separation, held-out data exclusion, second-half-only scoring with
negative outcomes retained, and exact CUDA resume during both data collection
and testing without refitting saved models. Synthetic cached predictions use
the same GPU inference path as production; production tolerances unchanged.
Startup: all six CUDA/RTX3080Ti, distinct CPU affinities0--5, base fit on40
contexts, nonempty checkpoints and first new training query reached. Formal
experiment remains running; no result claim. Frozen task/runtime root:
`bus_data_increment_20260921`.

Completed and analyzed2026-09-22: all six exit0/retry0,9.586--9.775h;
32 queries/source,16 test queries/source,3072 continuations/source plus one
repeat smoke. All18432 planned branches completed. Scheduler synced42 small
JSON files (261180 bytes); no local CSV/checkpoint retrieval. Frozen-runtime
summarizer produced `bus_data_increment_20260921/summary.json`.
Adaptive/frozen ungated14-minus-10 soft gain: -.385497 (seed SE.177190) /
+2.377619 (1.164050); raw-system gain: -2.627611 (5.157592) / +21.570424
(12.166608). Adaptive soft decreases in all three seeds; frozen increases in
all three. However,14-day gains vs native are soft+.190018 (.312693) /
-.625768 (1.188808), raw-7.786951 (9.521609) / +2.905041 (18.589324).
Ten-day soft vs native is+.575515 / -3.003387. Thus frozen improvement is
relative to a harmful base, not established native-relative usefulness.
Expanded selective chooses native in all96 test contexts, giving exactly
zero native-relative gain. Action-contrast MSE14-minus-10: -4.512437
(SE34.439639) / -19.410352 (26.787302), with one worsening seed per group.
The added data changes ungated actions in8/48 adaptive and17/48 frozen
contexts. No reliable control improvement established; do not expand this
low-dimensional readout data budget or select a temperature/seed/threshold
post hoc. Close this bounded test, retain the negative/mixed outcomes, and
next examine critic action-ranking training before further simulation.
No new jobs submitted and no manuscript edits in this analysis turn.

## 2026-09-22: Matched nonlinear critic objective screening

Freeze before execution. Reuse all six day500 SAC sources7700--7702 and both
temperature treatments. Keep the preceding14-training-day/4-test-day split:
56 fit contexts using all16-draw MC means,16 test contexts scored on draws8--15.
No new simulation. These test days are already inspected and now provide
development evidence only, not independent confirmation.

Two separate full nonlinear critic copies start from the same original
weights, including separate copies of the embedding shared with the original
actor. Keep the actor/alpha/reward mapping and original networks unchanged.
Fit the deployed clipped-min Q directly: level loss is context-averaged
action-averaged squared error to the MC mean; contrast loss subtracts each
context's mean error before squaring. Exactly, level=contrast+squared context
mean error. This is action-gap regression, not a new pairwise ranking loss.
Both arms use fresh Adam1e-5, full batches, equal context then action weight,
gradient clip1 and exactly4096 updates, with256-step atomic checkpoints.
No Bellman/target/actor updates, learning-rate sweep, gating or extra penalty.
Report the fixed endpoint, not a best training or held-out checkpoint.

Primary: contrast-minus-level and contrast-minus-native soft/raw-system
gains. Secondary: level/native, each versus original actor and the saved
14-day lowdim model, absolute and action-contrast sampled MSE, signed error,
switch/harm counts and initial/final training losses. Reproduce native
checkpoint predictions and saved lowdim choices before interpretation.
Retain all three seeds and both temperatures; average within seed then
mean/SE over seeds. Both primary comparisons must motivate any later fresh
confirmation; loss reduction alone does not justify actor training.

Six scheduler GPU workers,12 fits,49152 full-batch gradient updates; zero
additional rollouts. Source data, model/optimizer checkpoints remain remote;
only small training/progress/summary JSON returns. Finite-grid one-action
development screen, not full-policy or original RE-SAC validation. No paper
changes. Deploy and verify startup without waiting for training to finish.

Implemented independent critic copies and five tests locally/remotely for
objective decomposition, padding/shift invariance, original actor preservation,
test-label exclusion, second-half scoring, original source split, exact resume
in either objective, small-summary aggregation and source inference batch size.
Initial t100292--t100297 failed before training: combined-context inference
produced float32 differences from cached per-context predictions. Restored
the original per-context prediction batches without relaxing tolerances;
training remains full-batch. Scheduler retries t100298--t100303 all exit0
on jtl110gpu GPU0/1, three/card, CPUs0--5. Native differences are exactly0,
both4096-step fits complete, actor unchanged. These short fits had already
finished at startup verification (22.704--22.846s); no waiting for a long run.
Only205026 additional bytes of missing JSON retrieved, no CSV/checkpoints.
Aggregate: `bus_critic_ranking_20260922/summary.json`.

Contrast-minus-level soft gains adaptive/frozen: +2.258392 (seed SE1.406982)
/ +1.368944 (.664003); raw+27.553425 (28.822398) / +22.959648 (8.326976).
Contrast-minus-native soft: -3.323687 (1.344393) / +.136201 (1.752012);
raw-69.087106 (18.541747) / +20.165132 (14.443255). All adaptive seeds lose
in both metrics. Frozen soft seeds are-3.072413/+.520987/+2.960030. Level
soft vs native is-5.582079/-1.232742. Contrast/native/level held-out action
sampled MSE is814.256/589.752/856.835 adaptive,410.624/391.867/405.136 frozen.
Level fitting reduces mean signed underprediction to-5.998/-6.381 while
still worsening decisions. Contrast training losses drop to2.17--7.32 from
100.05--371.43, without reliable held-out control improvement. Training loss
and held-out sampled actor-anchored MSE have different definitions and must
not be compared as a direct train/test error ratio.
Neither arm passes the native-relative development screen. Do not promote
these fits to fresh simulation or full-policy training. Next examine
generalization using whole-day validation within training days before any
regularization/stopping change; the four inspected test days cannot select
hyperparameters or establish fresh confirmation. No manuscript changes.

## 2026-09-22: Training-day validation and stopping selection

Freeze before execution. Retain both nonlinear objectives, native checkpoints,
three seeds/two temperatures, Adam1e-5/clip1, full-batch context/action weights
and original14-fit-day/4-test-day split. No new regularizer or simulation.
For each arm/source,14 whole-day folds train on13 days/52 contexts and validate
on the excluded day's four contexts. At steps0,256,...,4096 compute centered
min-Q-minus-MC squared error against the all16-draw reference, with the same
context/action weighting as the contrast training objective. Record level
error, native-relative soft/raw gains and training loss too, but select only
by mean held-day centered error. Both arms use this identical criterion;
choose the earliest exact tie, with step0 eligible. Validation labels never
enter that fold's optimizer. No four-test-day inputs or outcomes select steps.

Save both selected steps before refitting from the original critic on all14
days. Preserve the selected model while continuing to4096 to reproduce the
prior fixed-budget predictions/actions. Score only selected and fixed4096
models on the four inspected test days, draws8--15; no test learning curve
or test-selected step. Original actor/shared embedding must remain unchanged.
Primary: each selected arm minus its fixed4096 control and native, soft/raw,
plus selected contrast-minus-level. Secondary original actor/lowdim gains,
MSE/bias, switching and full train/validation curves. Average inside seed,
then mean/SE over all three seeds separately by temperature. Overlapping
folds are not independent replicates; minimum CV error is a selection
statistic, not an unbiased estimate of performance. This remains inspected-
data development, not independent confirmation or original RE-SAC validation.

Budget: six scheduler GPU workers,168 CV fits plus12 full-data refits,
737280 gradient updates, zero new rollouts. Save model/Adam state every256
steps; preserve completed curves and selected snapshots on resume. Caches
and checkpoints stay remote, sync only small JSON. No paper changes.
Stop after deployment/startup verification, not after waiting for all fits.

Deployed t100310--t100315 via scheduler on jtl110gpu GPU0/1, three/card,
physical CPU affinities0--5. Three local tests (7.649s) and three remote tests
(9.387s) passed, covering whole-day splits, fixed selection/tie/step0 rules,
matched validation error, fold/refit exact resume, test-label exclusion,
unchanged actor, fixed-budget control reproduction and JSON-only aggregation.
Startup: all six live CUDA/RTX3080Ti, nonempty checkpoints,2/28 CV fits
complete and third fit at1280/4096. No formal result interpretation yet.
Frozen tasks/runtime: `bus_critic_day_validation_20260922`. Return after
startup verification as requested, without waiting for this campaign.

Completed all six original tasks exit0/retry0 in348.123--350.113s. All168 CV
fits and12 full-data refits complete; original actor unchanged and fixed4096
predictions reproduce exactly (maximum difference0 in every source). Retrieved
425242 bytes of missing JSON only; aggregate is
`bus_critic_day_validation_20260922/summary.json`. No new jobs or paper edits.
Selected steps7700/7701/7702: adaptive level2816/0/512, contrast2816/256/256;
frozen level256/1024/0, contrast768/512/0. All selected steps precede4096;
three of12 select no update. Curves show continued fitting with deteriorating
held-day error, supporting an overfitting component under this fixed setup.

Selected contrast-minus-fixed soft adaptive/frozen: +2.705635 (seed SE1.989007)
/ -1.182862 (.933352); raw+41.789803 (33.532517) / -20.553768 (15.131760).
Selected contrast-minus-native soft: -.618052 (1.126137) / -1.046661 (.915954);
raw-27.297303 (21.441687) / -.388635 (4.059540). Adaptive soft seeds are
+1.372119/-2.526357/-.699916; frozen-2.872025/-.267958/0. Selected level
soft vs native is-2.475573 (2.115112) / -.541741 (.473215). Selected contrast
action MSE687.532/326.853 vs fixed814.256/410.624 and native589.752/391.867.
Thus earlier stopping helps some error measures but not reliable control;
the frozen group has lower action MSE yet worse soft return than fixed4096.
Among nine nonzero selected fits, five already have negative native-relative
soft gains on their training-day CV predictions. This is descriptive evidence
of selection-objective mismatch, not proof that changing the selector will
generalize. Do not replace the frozen MSE selector after inspecting test gains.
Neither selected objective passes the native-relative screen. A next bounded
development comparison may select by training-day decision returns, with its
rule frozen before execution and the inspected test days still excluded from
selection; do not increase data/training budgets or promote to policy training.

## 2026-09-22: Training-day return-based stopping comparison

Freeze the new selector before execution. Reuse all14 existing training-day
CV curves at0,256,...,4096 for each arm/source; do not retrain the CV folds.
Separately within each original level/contrast arm, seed and temperature,
maximize mean held-day soft return minus native. CV returns use all16 draws
with equal context/day weights. Choose the earliest exact tie; step0 has
zero gain and remains eligible. Raw-system return and MSE are reported but
never break ties, gate choices or select steps. No four-test-day input or
outcome enters selection. No choosing a preferred objective or temperature.
Preserve the preceding MSE-selected results as controls, not replace them.

Save selected steps before fitting/evaluation. Reuse native at0 and stored
MSE-selected/fixed4096 models when the step matches. Otherwise refit the
same original critic on all14 training days only to the selected step, with
unchanged full-batch Adam1e-5/clip1 and original objective. Verify final
training losses match the saved full-data trajectory at that step; retain
optimizer checkpoints for exact resume. Original actor/alpha remain fixed.
Score return-selected, MSE-selected and fixed4096 models on all four inspected
test days, draws8--15. Primary: per-arm soft/raw gain vs MSE-selected and
native; retain fixed, actor, lowdim and contrast-versus-level comparisons,
all seeds, temperatures, zero-update cases and negative outcomes.

Six scheduler GPU tasks, at most12 short refits within the existing4096-step
ceiling; no new CV fits or rollouts. Reuse models instead of repeating fits.
Only small result JSON returns; checkpoints/data stay remote. Maximized CV
gain is optimistic and is not evidence of benefit. This sequential comparison
uses previously inspected test data and remains development, not independent
confirmation or original RE-SAC/policy-improvement validation. No paper edits.
Deploy, verify startup and stop without waiting for unfinished training.

Deployed t100318--t100323 through scheduler to jtl110gpu GPU0/1, three/card.
Eight local and eight remote tests passed, including exact short-fit/resume,
model reuse and test-label exclusion. All six finished during startup
verification in2.864--16.127s. Seven of12 selections retain native; five
short refits total7424 updates, with no new CV fits or rollouts. Retrieved
36 small JSON files (564730 bytes), no CSV/checkpoints. Aggregate:
`bus_critic_return_selection_20260922/summary.json`. Selected steps7700/1/2:
adaptive level3072/0/0, contrast2304/0/0; frozen level0/0/0,
contrast512/768/768.

Return-selected contrast-minus-MSE soft adaptive/frozen: +1.185062
(seed SE.679145) / +1.138270 (.642050), positive in all six paired seeds.
Contrast-minus-native soft: +.567010 (.567010) / +.091609 (.505077);
raw+2.939192 (2.939192) / -1.530027 (3.030706). Adaptive soft seeds are
+1.701031/0/0; frozen-.592477/-.210029/+1.077335. Frozen contrast-minus-actor
soft remains negative in all three seeds (mean-.737440, SE.132752).
Level-minus-native soft is-2.202766 (2.202766) / 0. Selection improves the
MSE-selected comparator, but does not establish stable native-relative benefit.
Next diagnose training-day CV-to-refit action-ranking changes separately from
cross-day transfer, without increasing training/rollout budgets. These remain
development results on inspected test days, not original RE-SAC or full-policy
validation. No additional campaign launched and no manuscript changes.

## 2026-09-22: Read-only selected-critic transfer diagnostics

Freeze before execution. Reuse all six return-selection caches and their
unchanged level/contrast models; no new updates, CV fits, selection or rollouts.
Score final models on the14 original training days with all16 draws and four
inspected test days, reporting first8/all16/second8 separately. Second8 remains
the original primary evaluation; neither half changes model selection. Compare
selected CV and refit gains on identical training days/draws. Report the exact
arithmetic split of second8-test-minus-CV into refit-train-minus-CV,
all16-test-minus-refit-train, and second8-test-minus-all16-test. Retain per-day
and per-context gains versus native/actor, switch rates and paired draw scatter.
Keep all objectives/seeds/temperatures and reproduce prior test choices/gains.

CV fold snapshots/actions were not retained, only day averages: exact fold-to-
refit action disagreement cannot be recovered without refitting. Final-model
training scores are in-sample, selected CV scores optimistic, and13-versus14
fit days differ. This is descriptive localization, not causal attribution or
independent confirmation. No confidence claim from overlapping folds. Six
short scheduler GPU inference jobs; checkpoints stay remote, only JSON returns.
No source-model/manuscript changes. Stop after startup verification if still running.

Completed t100327--t100332 on jtl110gpu GPU0/1, three/card, in3.127--3.305s;
zero updates/rollouts and all original test choices/gains reproduced. Three
local tests (1.537s) and three remote tests (3.341s) passed. Retrieved18 JSON
files,1010650 bytes; no checkpoints. Results:
`bus_critic_transfer_diagnostics_20260922/summary.json`.
Contrast soft gain CV/refit-train/test-all16/test-second8: adaptive
+.155834/+1.241782/+.960847/+.567010; frozen
+.352941/+2.489877/-.136698/+.091609. All five nonzero fits have higher
mean training-day refit gains than CV gains, not an aggregate in-sample
collapse during refitting. This does not establish unseen-state refit stability.
Frozen contrast raw train/test-all16/second8: +48.388331/-1.519248/-1.530027.
Frozen soft first8/second8 seeds: -1.005276/-.592477,
-.491205/-.210029, +.401464/+1.077335: both halves retain two negative seeds,
so the weak aggregate benefit is not explained solely by the second8 draw set.
Adaptive contrast gains remain confined to7700;7701/7702 choose no update.
Adaptive level train soft+.652753 becomes test-all16-1.519385 and second8
-2.202766. The strong in-sample gains do not persist consistently across days;
already-small selected CV gains should not be replaced by refit training gains
as expected benefit. Next target cross-day action-gap generalization rather
than more stopping-rule searches or longer fitting. An original-gap anchoring
penalty is a candidate single-factor control, not a validated remedy. No new
training campaign or manuscript edits; retain all negative results.

## 2026-09-22: Fixed native-gap anchoring control

Freeze before execution. Define C_g as subtraction of the equally weighted
candidate mean within context g. Fit copied clipped-min Q with
mean_g mean_a [C_g(Q-Y)]^2 + lambda mean_g mean_a [C_g(Q-Q_native)]^2,
lambda=1 fixed, no sweep. Y is the original16-draw soft-return mean and
Q_native the cached original critic prediction. The penalty ignores any
context-constant Q shift; it is not parameter regularization or Q-level
anchoring. Algebraically the objective is twice centered squared error
against (Y+Q_native)/2 plus a parameter-independent constant. It tests target
gap shrinkage, not a new uncertainty decomposition or absolute-Q remedy.

Keep source day500 SAC, seeds7700/7701/7702, adaptive/frozen temperature,
14 fit days/56 contexts, copied embedding, Adam1e-5/full batch/clip1,4096-step
ceiling and checkpoints every256 updates. Reuse unanchored return-selected
contrast, selected level and fixed4096 contrast snapshots; do not retrain them.
For anchored contrast only, run14 leave-one-whole-day-out fits and choose
maximum mean held-day soft gain versus native at0,256,...,4096, earliest exact
tie and step0 eligible. This is the same previous return-selection rule.
No raw-system score or test state/label chooses steps or coefficients. Save
selection before one full-data refit; keep selected snapshot and4096 endpoint.
Store CV context choices/gains with curves so later diagnosis needs no reruns.

Primary: anchored-selected minus unanchored-selected and native, soft/raw
system returns. Secondary: anchored4096 minus unanchored4096 (matched steps),
selected level/actor controls, Q bias/error, native-gap drift and switching.
Score the four inspected test days only on draws8:16; reproduce cached control
choices/gains. Retain all seeds, temperatures, failures and step0 selections.
Mean within seed then mean/seed SE, no independence claim for overlapping folds.
Same inspected development data, not independent confirmation or full-policy/
original RE-SAC validation. Source actor/alpha and manuscript stay unchanged.

Six scheduler GPU tasks,84 new CV fits plus6 full-data fits,368640 updates,
zero new rollouts. Model/Adam/active-fold/selection state resumable; cache and
checkpoints stay remote. Only small JSON results return. Fifteen local tests
passed in15.322s, including shift/gradient invariance, shrinkage identity,
ragged-action weights, criterion/step0 rules, fold/refit exact resume, test-label
exclusion and original unanchored regression tests. Stop after deployment and
startup verification; do not wait for training to finish.

Deployed t100364--t100369 via scheduler to jtl110gpu, where the source caches
remain; GPU selection left to scheduler. Four remote tests passed in59.092s.
Startup: t100364/365/366 on GPU1, t100367/368 on GPU0, all live CUDA/RTX3080Ti,
unique physical cores6--10, active first-fold steps256--768 and nonempty
model/optimizer checkpoints (~1MB each). t100369 (frozen7702, core11) remains
queued because both GPUs reached four tasks/card including three pre-existing
BAPR tasks; no packing cap override or other-task interruption. Scheduler
will dispatch it after capacity frees. Small-result auto-pull is configured;
no checkpoints downloaded, no formal outcomes read and no manuscript edits.
Return after startup verification without waiting for the campaign to finish.

Completed all six t100364--t100369 exit0/retry0, including the queued worker;
84 CV fits and6 refits, original actor unchanged and cached controls reproduced.
Elapsed516.209--677.651s/source. Automatic result-sync records still showed
zero attempts and no local outputs at inspection; manually retrieved30 small
JSON files (304618 bytes), no validation dump or checkpoints. Frozen-runtime
aggregation saved `bus_critic_gap_anchor_20260922/summary.json`.
Selected steps7700/1/2: adaptive256/3584/2560, frozen1024/768/2560.

Anchored-selected minus native soft adaptive/frozen: -.297226 (seed SE.072944)
/ +.042837 (.962963); raw-11.400169 (7.880014) / +16.356219 (12.562167).
Adaptive soft seeds-.161515/-.411443/-.318719; frozen-1.108860/-.718136/+1.955506.
Anchored-selected minus unanchored-selected soft: -.864236 (.499872) /
-.048773 (.463478); raw-14.339361 (7.606978) / +17.886246 (9.577477).
Against the original actor, anchored soft is-.428490/-.786213 and raw
-8.320483/-2.117635. Preserve the positive frozen native-relative raw result,
but it does not establish an improvement over the original actor.

Matched4096 anchoring reduces test native-gap drift135.144->41.397 adaptive
and106.679->38.557 frozen (69.4%/63.9%). Endpoint soft gain versus unanchored
is+2.403627 (1.760686) / +.588876 (1.659214), but versus native is-.920059 /
+.725077. These secondary endpoints do not replace the prespecified selected
comparison. All six selected CV gains are positive, but five test soft gains
are negative. The penalty constrains drift as intended without establishing
reliable cross-day decision benefit; this fixed-lambda repair fails the primary
screen. Do not scan lambda or promote a favorable endpoint/temperature from
these outcomes. Retain the negative result; no new jobs or manuscript edits.

## 2026-09-22 Original versus decision-time physical observation

Freeze before dispatch: `bus_observation_information_20260922/tasks.json`,
six paired GPU jobs, seeds7700--7702 crossed with adaptive/frozen temperature.
Compare original28 observation against original28 plus22 current-state fields:
within-hour time, trip age, load, current next-stop queue, and each recorded
forward/backward neighbor's presence, signed relative position, speed, load,
on-route/holding/dwelling phases and remaining holding/dwelling seconds.
Use the simulator's existing neighbor definitions. No future arrivals/RNG or
pending focal action in added inputs; these are simulator-provided diagnostics.

Same native-anchored quadratic-action ridge.01, equal context/action weights.
Keep the original-only training-fitted RMS scale for both maps; no margin or
hyperparameter sweep. Same14 training days/all16 draws,14 leave-day-out pairs
for reporting only, then full-data paired refit. Freeze models before extracting
the four inspected test days. Primary: augmented-minus-original plus each versus
native/actor soft and raw-system gains on draws8:16, all seeds/temperatures.

Replay108 existing prefixes to recover432 contexts, verifying time/bus/trip,
observation, actor/grid and native Q before joining cached returns. Zero new
suffix continuations;180 small GPU ridge solves. Checkpoint each extracted
context and fitted pair; skip complete prefix days/folds on resume. Cache stays
remote; retrieve compact JSON only. Augmentation changes model expressivity;
this is inspected-day development, not independent confirmation, sufficient-state
proof, full-policy improvement, uncertainty decomposition or original RE-SAC
validation. Paper unchanged; stop after deployment and startup checks.

Deployment:17 local tests (14 passed,3 CUDA tests skipped), five CUDA tests on
gpu2 passed, including extraction/CV/refit interruption and test-label exclusion.
First launch t100382--t100386 failed before replay: cached outcomes contain
only soft/system returns, not the older focal metrics. Corrected the metric
list to match the source; primary endpoints unchanged. Re-ran local regression
and validated all six real cache splits/metadata joins. t100387 was cancelled
before its first extracted context; its empty-progress checkpoint is resumable.
Retain these attempt records. Scheduler retries t100388--t100392 are running
on jtl110gpu CUDA with distinct physical cores6--10, first1--2 contexts verified
and nonempty ~440KB checkpoints. Replacement t100393 (frozen7702/core11) is
queued at the four-tasks/card cap, counting three pre-existing BAPR workers.
No cap override, source checkpoint download or manuscript changes. Stop here;
no formal outcomes inspected and no waiting for completion.

Completed t100388--t100393, all exit0 with no further retries beyond the
recorded startup repair. Each verified72 contexts across18 prefixes, completed
14 CV pairs and1 refit pair, and retained the original actor/critic unchanged.
Elapsed1580--2666s/source. Small results were already local; restricted rsync
of30 JSON files (125871 bytes) transferred0 files. No checkpoints retrieved.
Frozen-runtime aggregation: `bus_observation_information_20260922/summary.json`.

Augmented-minus-original soft adaptive/frozen: -1.287046 (seed SE1.657930) /
+1.399060 (2.858310); raw-system -15.112416 (15.845455) / +15.971081 (25.050321).
Augmented-minus-native soft: -5.927561 (1.868601) / -3.231114 (.701540);
raw-system -89.631812 (10.814123) / -39.213374 (10.545117).
All six augmented runs have negative soft/raw gains versus both native and
actor. Original-input correction also loses to both baselines in all six runs.
Augmentation lowers training pair MSE in all six; test pair MSE rises998.989
to1156.345 adaptive and falls1169.211 to846.818 frozen. Preserve the mixed
prediction result, but the native-relative decision-benefit screen fails.
These data do not show that observation insufficiency caused the earlier
failures, nor that physical state is sufficient. No new jobs or manuscript edits.
Next proposed diagnostic: fixed-sample-count same-day context exclusion versus
whole-day exclusion on cached training data, to test whether same-day information
helps the unchanged readout. This would be a transfer diagnostic, not prospective
policy evaluation; freeze its exclusions before fitting and keep the current test
days out of selection. Do not sweep penalties or add further state fields here.

## 2026-09-22 Same-day versus whole-day exclusion

Freeze `bus_day_context_transfer_20260922/tasks.json`: six GPU jobs, all three
seeds and both temperatures. Use only the14 training days/56 contexts and
their cached physical fields. For target slot k of day d, retain the other
three points of d; exclude d/k and the three points of donor day e at slots
other than k. Enumerate all13 donor days. Both conditions fit52 contexts,
13 per slot. Refit the unchanged original/augmented ridge.01 pair on each
retained set; keep original-only fitted scaling shared within the pair.

Reuse source whole-day CV choices and all16-draw metrics; verify their joins
against cached returns, without retraining those controls. Same-day targets
also use all16 draws; their labels never fit their own readouts. Average
donor outcomes within target, then targets/day/seed; no best donor, averaging
predictions into an ensemble, model selection or donor-level significance test.
Primary: same-day minus whole-day soft/system gain for each representation,
and each versus native/actor. Retain MSE, switches and choice disagreement.

Per source728 new fit pairs;8736 ridge solves across six jobs, zero simulator
runs or actor updates. Save every completed pair and latest models; all donor
details/cache stay remote, only compact summaries return. Four previous test
days excluded. Same-day points may occur later than the target: retrospective
conditioning, not prospective policy validation or original RE-SAC evidence.
Paper unchanged; stop after deployment and startup verification.

Four local tests passed (CUDA test skipped); all five remote CUDA tests passed
in7.901s, including exact resume, held-point exclusion and donor averaging.
All six real source joins and728 exclusion plans/source validated before
submission. Scheduler t100423--t100427 now run on jtl110gpu GPU1/GPU0 with
distinct physical cores6--10,21--45 completed pairs/source at the startup
check, nonempty ~420--446KB checkpoints and retry0. t100428 (frozen7702,
core11) is queued at four tasks/card including existing BAPR jobs. No other
tasks interrupted, cap overrides, checkpoint downloads or formal outcome reads.

Placement update: user released the GPU1-only restriction for RE-SAC. The
campaign manifest now allows jtl110gpu/jtl110gpu2, both with the frozen runtime
and required source caches staged. Streamed only18 source files (~3.1MB) from
GPU1 to GPU2, with no local checkpoint files. All six source loads passed there.
Edited the sole queued task t100428 in place; scheduler launched it on
jtl110gpu2 GPU0 with Torch2.4.1 CUDA, affinity11,5/728 completed pairs and a
408220-byte recovery checkpoint. The other five running tasks were untouched.
No CPU-server placement, experiment changes or formal outcome inspection.

Completed all six t100423--t100428 exit0/retry0, each728 pairs/1456 solves,
56 targets/14 days/13 donors,525.620--894.763s/source. Retrieved15 missing
JSON files (735135 bytes before compression); the other15 were already local.
No checkpoints downloaded. Frozen-runtime `bus_day_context_transfer_20260922/summary.json`
retains all seeds. Completion, finite values, donor counts and context-to-day
to-seed aggregation agree across all six outputs.

Same-day-minus-whole-day soft gains, adaptive/frozen mean (seed SE): original
+.116498 (.628416)/-1.109102 (.745461), augmented+.245693 (.762739)/+.428705
(.335004). Augmented raw-system changes are+5.834999 (10.764357)/+4.707931
(7.842756). Augmented same-day-minus-native soft remains-3.884445/-3.339880,
negative in five of six runs; raw means-44.816601/-38.737889. Original same-day
soft gains versus native are negative in all six. Frozen augmented-minus-original
soft is+2.412457 (.665859), but the original condition itself worsened; this
relative improvement does not establish native-relative decision benefit.
Same/whole choices differ in23--30% of donor comparisons on average by
representation/temperature, without reliable return improvement.

This fixed-readout diagnostic does not support same-day information as a
sufficient repair. Its retrospective conditioning and inspected-data limits
remain. Stop extending this offline feature/readout line. Next proposed:
matched fixed-policy, fixed-replay Bellman-backup controls to separate target
aggregation's Q-level bias from action-gap errors before actor updates.
No new jobs or manuscript edits in this analysis turn.

## 2026-09-22 Fixed-policy/replay Bellman-backup controls

Freeze `bus_frozen_backup_20260922`: seeds7700--7702, adaptive/frozen day500
SAC checkpoints. Sample32768 original replay rows without replacement using
seed*100000+26001. Cache one next action/log density per row from the unchanged
actor (stream+26002); evaluate the unchanged original target network. Reuse
512 co-batches (+12000) for the MC-compatible expected reward transform g_D,
including transformed zero terminal rewards. No new simulator trajectories.

Three independently copied full critics, including embedding: min target,
mean target, and min plus gamma*(1-d)*c, where c is the sampled nonterminal
mean head-mean-minus-min gap. Mean/min_shift target sample means agree;
terminal targets remain g_D(0). The uplift is a constant continuation bonus,
not a post-fit constant Q shift; terminals/dynamics can transmit it into
action gaps. All arms use fixed cached targets, fresh Adam1e-5, both-head MSE,
clip1,4096 minibatch2048 updates, identical minibatch streams (+26003).
No actor/alpha/target update, chaining, stopping selection or penalty sweep.

Primary fixed4096 comparisons: mean-minus-min and mean-minus-min_shift
soft/raw-system returns, plus each against original native min and actor.
All fitted critics use min-head decisions, isolating the training target.
Secondary: signed all-candidate Q error, exact squared context-offset plus
centered-MSE decomposition, native-anchor gap MSE, switches/harm and train
loss. Use the original four increment test days/16 contexts, draws8:16;
test data never fit targets/models or select steps. Average by day then seed,
report all three seeds separately per temperature. Inspected-data one-root
development diagnostic, not iterated Bellman/online or RE-SAC validation.

Budget: one scheduler GPU preparation job, then six GPU fitting jobs,
18 fits/73728 critic updates total. Preparation extracts only compact inputs
from original server checkpoints. Fit jobs may use GPU1/GPU2 after staging;
large source checkpoints stay put, local retrieval remains small JSON only.
Checkpoint model/Adam/sampling RNG every256 updates and completed prep sources;
completed arms skip fitting on resume. Paper unchanged. Stop after startup.

Seven tests passed locally (six plus the runner test separately); all seven
passed on the staged GPU1 runtime in7.731s, including CUDA optimizer/RNG and
whole-run interruption after a completed arm. Preparation t100440 exit0;
six source-native prediction joins and target-mean identities passed on real
data. Compact inputs total45423660 bytes, streamed GPU1-to-GPU2; local received
18805 bytes of preparation JSON only. GPU2 loaded all six inputs successfully.

Submitted t100442--t100447 once, eligible on GPU1/GPU2. GPU2 SSH scan timeouts
blocked the last two initially; fresh scheduler resume scans succeeded and
were committed under its state lock, without disabling checks or resubmitting.
Scheduler placed all six on GPU1 GPU1 (GPU2 later timed out in node probing).
At startup inspection t100442--445 had already completed12288 updates each,
exit0/retry0,65.721--66.829s/source. t100446/447 were live CUDA/RTX3080Ti on
distinct cores10/11, each8960/12288 updates with ~935KB checkpoints; earlier
workers used cores6--9. Original actor/critic/target unchanged in completed
markers. No formal performance results inspected. Stop without waiting for
the remaining fits; no other tasks interrupted or manuscript changes.

All six t100442--t100447 completed exit0/retry0:18 fits/73728 updates,43.616--
66.829s/source. Source networks remained unchanged. Auto-sync returned most
outputs; restricted rsync added six JSON files/64701 bytes. No input bundles
or checkpoints downloaded. Frozen-runtime `bus_frozen_backup_20260922/summary.json`
retains all seeds. Budgets, finite values, target-mean matching, day aggregation
and the exact offset-plus-centered-MSE identity passed for all outputs.

Mean-minus-min soft adaptive/frozen: +.009391 (seed SE.009391)/-.020714
(.045621); raw+.132218/-.290993. Mean-minus-min_shift soft+.061643 (.042513)
/-.079283 (.065563); raw+.683535/-.698360. Mean-minus-native soft+.074172
(.095032)/+.101910 (.227620), with only7702 positive in each temperature;
mean-minus-actor soft-.057092/-.727140, all frozen seeds negative.

Relative to fitted min, mean raises test Q by2.456218/2.393042; min_shift by
2.406762/2.318932. Mean Q-minus-MC remains-142.356196/-150.904065. Of mean's
sampled-MSE reduction702.208887/784.975094 versus min,99.175%/99.875% comes
from squared within-context offset reduction. Centered MSE falls298.575989
to292.782957 adaptive and168.561323 to167.582865 frozen, but unchanged native
scores243.767582/166.607623. Mean versus min changes only3/48 test choices
per temperature. This one-generation update does not establish reliable
native/actor-relative decision improvement. Preserve the small/mixed gains.

Limitations remain one frozen target generation and inspected finite-grid
development queries. A small one-step uplift does not rule out cumulative
bootstrap underestimation. Next proposed: fixed-policy/replay rolling-target
controls at matched gradient-update budgets, retaining these frozen-target
results as controls. No new jobs or manuscript edits in this analysis turn.

## 2026-09-22 Matched-budget rolling Bellman-backup controls

Freeze `bus_rolling_backup_20260922`: the same seeds7700--7702 and adaptive/
frozen day500 sources,32768 replay rows, rewards, cached next actions/log
densities, alpha and original online/target networks as the frozen controls.
Only enable target <- .99*target+.01*critic after each optimizer step,
including embedding parameters. Each arm starts from day500, not the fitted
frozen endpoint. No actor update, new simulator draw, target resampling or
test-based selection. Extract only missing next_state rows by stored replay
indices; verify state/action/done alignment and reproduce initial targets.

Keep min, mean and min_shift, fresh Adam1e-5, clip1, minibatch2048, paired
sampling streams and4096 updates/arm. The continuation bonus c in min_shift
remains the original sampled nonterminal head gap; do not recalibrate it.
Thus rolling mean/min_shift target sample means need not remain matched.
Use archived frozen endpoints without retraining; verify their choices and
returns against the same16 contexts/four days and outcome draws8:16.

Primary: paired rolling-minus-frozen soft/raw-system return per arm; rolling
mean-minus-min/shift and their changes from frozen contrasts. Retain every
arm versus native min and actual actor. Secondary: signed Q error, exact
offset/centered-MSE decomposition, native-gap MSE, action disagreement and
target evolution. Aggregate by day then seed, separately by temperature.
One-root inspected fixed-policy/replay development, not full-policy or
original RE-SAC/dual-uncertainty validation.

Budget: one scheduler GPU preparation job and six GPU fitting jobs,
18 fits/73728 new critic/target updates. GPU1/GPU2 eligible after compact
server-to-server staging. Original checkpoints remain GPU1; only small
result JSONs return locally. Checkpoint online/target/Adam/RNG and completed
arms every256 updates; preparation skips completed sources. Eight local
tests pass, including CUDA resume, tau0 agreement with frozen fitting,
replay alignment and whole-run/preparation recovery. Paper unchanged.

All eight tests also passed on the staged GPU1 runtime in5.299s. Preparation
t100460 completed all six sources; initial target differences were exactly0.
Streamed six sidecars/25510840 bytes GPU1-to-GPU2 and staged18 archived control
JSONs/343204 bytes. GPU2 validated every sidecar/control and CUDA operation.
Local retrieval was four preparation JSONs/28332 bytes; no checkpoint/input
download. Scheduler doctor passed without fixes.

Submitted t100465--t100470 once. Scheduler placed t100465/467/470 on GPU1 GPU1,
t100466/469 on GPU1 GPU0, and t100468 on GPU2 GPU0; both nodes remain eligible.
Startup confirmed CUDA/RTX3080Ti and distinct physical cores6/7/8/10/11 on
GPU1, core9 on GPU2. GPU1 workers had2048--2560 updates and~418KB checkpoints;
GPU2 had4864 updates and~832KB checkpoint. Initial target differences remain0.
Stop after startup without waiting or inspecting endpoint performance.
Other projects and manuscript unchanged.

All six t100465--t100470 completed exit0/retry0:18 fits/73728 updates,
176.981--203.696s/source. All36 result JSONs/1278267 bytes were already
auto-synced; restricted server rsync transferred zero result files. Frozen
runtime generated the cohort summary. Full budgets/history, finite values,
archived context/choice/value joins, day aggregation and MSE decomposition
passed. Source networks remained unchanged; no checkpoints downloaded.

Adaptive/fixed-temperature means (seed SE): rolling mean-minus-min soft
+.697303 (.226550)/+.769252 (.306961), raw+6.576394/+8.432053; both outcomes
positive in all six runs. Mean's rolling-minus-frozen soft is+.389510
(.068737)/+.787422 (.470202), positive in3/3 and2/3 seeds respectively.
Rolling mean-minus-native soft+.463683 (.149076)/+.889331 (.668644), raw
+3.575283/+10.871431. Mean-minus-actor soft+.332418 (.029975)/+.060282
(.355820), raw+6.654968 (1.215837)/-7.602423 (3.019495). Adaptive actor
comparisons improve both outcomes in3/3 seeds; fixed-temperature raw actor
comparisons worsen in3/3. Retain this temperature-dependent failure.

Rolling mean-minus-min_shift soft+.081745 (.043664)/-.039448 (.249841),
raw+4.571199/-6.618983. Fixed continuation uplift reproduces88.3%/105.1%
of mean's aggregate soft gain over min; no general advantage over this
simple control. Mean Q-minus-MC moves from min's-144.150229/-155.963591
to-66.299072/-81.711012. Of its absolute-MSE reduction,99.850%/99.924%
is offset reduction. Centered MSE311.950129->286.350459 adaptive and
169.419821->155.253916 fixed, versus native243.767582/166.607623.
Mean/min choices now differ18/48 and20/48 contexts, versus3/48 per
temperature with frozen targets. Adaptive min seed7702 training loss
increased765.778->876.082; endpoint completion is not convergence proof.

Rolling backups strengthen the Q-level/decision signal versus frozen-target
fitting, but constant-uplift controls and fixed-temperature actor failures
prevent a distinct RE-SAC or dual-uncertainty conclusion. The queries are
inspected one-action development interventions, not full-policy evaluation.
Next proposed: independent paired evaluation on new bus days/trajectory
streams, keeping the fixed4096 endpoints, both temperatures and min/mean/
min_shift/native/actor controls; no selection or further tuning here.
No new jobs or manuscript edits in this analysis turn.

## 2026-09-22 Independent bus-day backup confirmation

Freeze `bus_backup_confirmation_20260922` before new simulation. Seeds7700--7702,
both temperature groups, all six rolling/frozen min/mean/min_shift critics
at the preselected4096 endpoint, original native min and actor. No training,
new coefficient, source selection, early stopping or MC-selected controller.
Retain frozen-target controls to test whether rolling effects generalize.

Eight fresh prefix days/source, times12000/24000/36000/45000, action grid
0/15/30/45/60 plus actual actor action. Route sigma1.5 unchanged. New namespace
1900000000+(seed-7700)*100000; prefix environment/action offsets0/1000, suffix
offsets10000/20000+100*probe+replicate. Shared streams across temperature
groups; no overlap with prior diagnostic streams. The original512 reward
co-batches (+12000), alpha, terminal transform and joint actor stay fixed.
All eight fresh draws score predictions fixed before any suffix simulation.
One root intervention only; complete suffix, no root entropy or learned tail.

Primary rolling mean-minus-min/shift soft and raw-system returns; all rolling
arms versus native/actor and their own frozen endpoints. Report frozen
contrasts and rolling/frozen interactions; signed Q error, offset/centered
MSE, native-gap error and harmful switches. Mean by day, then source seed;
all three seeds separately per temperature, no pooling days as training seeds.

One short scheduler GPU preparation job, then six GPU evaluation jobs.
48 source-policy days/192 contexts/up to9216 continuations plus six exact
repeat smoke branches;0 gradient updates. Keep raw CSVs, input bundles and
per-branch recovery checkpoints remote; only progress/result JSONs sync.
Resume skips complete days and branches, regenerates incomplete prefixes
and verifies predictions. Six local tests pass, including all-draw negative
outcomes, fresh streams, no optimizer updates, CUDA recovery and preparation
resume. Limited to independent simulation confirmation of frozen one-action
decisions under reused training seeds, not full-policy or RE-SAC validation.
Paper unchanged; stop after startup, without waiting for completion.

All six tests also passed on the frozen GPU1 runtime in12.097s, including
CUDA branch/day resume and an optimizer-step prohibition. Preparation
t100487 exit0/retry0 verified every archived endpoint choice/metric and
the original reward transform. Six compact payloads total5452412 bytes,
streamed GPU1-to-GPU2; GPU2 loaded all inputs and executed CUDA successfully.
Only four preparation JSONs/14220 bytes were pulled locally; original replay
checkpoints stayed GPU1 and one compact fitted endpoint moved GPU2-to-GPU1.

Submitted t100488--t100493 once, all running/retry0 at startup. Placement:
t100488/491 GPU1 GPU1; t100489/492 GPU1 GPU0; t100490/493 GPU2 GPU0. Each
uses CUDA/RTX3080Ti and one distinct physical core:6/7/9/10 on GPU1,8/11 on
GPU2. All six passed the real first-branch exact-repeat smoke and persisted
1--3 completed branches, checkpoints~908KB. Input preparation tolerated a
transient GPU2 probe timeout; no pin restrictions or checkpoint checks were
removed. Stop after startup, without reading performance or waiting for
the remaining full-simulator continuations. No other tasks or paper changed.

## 2026-09-26 Independent confirmation results

All six t100488--t100493 completed and archived exit0/retry0. Each finished
8 days/32 contexts/1536 continuations,4.315--5.019h/source;9216 scored
branches plus six repeat-smoke branches in total. All36 result JSONs
(1808955 bytes) were already auto-synced; restricted rsync transferred zero
files. Frozen-runtime cohort summary saved. Budgets, finite values, unchanged
models, context/day/seed aggregation, paired comparison identities and exact
offset-plus-centered-MSE decomposition passed. No CSV/checkpoint downloads.

Adaptive/fixed-temperature mean (seed SE): rolling mean-minus-min soft
-.379127 (.293940)/-.343582 (.283476), raw-10.742344 (8.035986)/-.431821
(3.143927). Soft gains are negative in2/3 seeds in each group, reversing
the development result. Mean-minus-min_shift soft-.100315 (.127504)/
-.026712 (.529289), raw-4.367653/+3.490529: no consistent mean advantage.
Mean rolling-minus-frozen soft-.326911 (.309706)/-.340928 (.174165).

Rolling mean-minus-native soft-.923764 (.704953)/+.060076 (.248001),
raw-17.428122/+6.335088. Mean-minus-actor soft-.303765 (.206644)/+.119841
(.463557), raw-12.307534 (4.265954)/+7.408299 (6.966585). Adaptive actor
comparisons worsen both outcomes in all three seeds; fixed-temperature
positive averages remain mixed across seeds. No stable decision benefit.

Q-minus-MC changes from rolling min's-155.827998/-148.342094 to mean's
-77.261015/-73.838526. Offset reductions account for99.889%/99.998% of
absolute-MSE improvement. Centered MSE285.067394->263.689002 adaptive and
262.321563->261.934738 fixed; native254.358548/258.895657 remains lower.
Mean/min decisions differ39/96 and32/96 contexts. Preserve all daily gains,
including large negative days; neither training failure nor cherry-picking
explains this completed evaluation. Positive development gains did not
generalize to the new draws, although Q-level bias reduction did.

This is a negative independent confirmation for a reliable control-benefit
claim, not proof that mean backups are always worse. Only three reused
training seeds/group, one noisy route and one-root interventions were tested.
Next proposed: use existing server-side branch predictions/outcomes to
attribute harmful action switches and separate common Q shifts from action-gap
errors before any new training. No further tuning on these confirmation
queries as if they were untouched. No new jobs or manuscript edits this turn.

## 2026-09-26 Backup action-switch attribution

Use the completed confirmation CSVs/predictions in place on GPU1/GPU2;
no network inference, optimizer, new simulation, policy selection or veto.
Keep mean/min, mean/min_shift, mean/native, min_shift/min and rolling/frozen
mean choices fixed. Decompose the prediction difference into a within-query
common shift and centered action correction; verify the former keeps the
baseline ranking and the latter reproduces the changed ranking. This is a
post-fit identity, not an equivalence to the min_shift training bonus.
Compare all-action centered errors with errors on the actual switched pair;
paired MC variance cancels in model-MSE differences. Retain all eight draws,
all contexts and signed positive/negative contributions; halves0:4/4:8 only
describe noise sensitivity. Worst1/3 gross-loss shares are descriptive, not
trimmed evaluation. Report day->seed aggregation and all temperatures/seeds.
Six tests passed, including unchanged-choice denominators, source-summary
reproduction, gap/variance identities and the global-versus-selected-pair
error distinction. Detailed attribution stays remote; pull compact JSONs.
This is retrospective diagnosis, not new confirmation. Paper unchanged.

All six sources reproduced their original summaries exactly:192 contexts,
9216 existing continuations, zero new rollouts/updates. Retrieved18 JSONs,
920544 bytes uncompressed; detailed contexts/CSVs/checkpoints stay remote.
Aggregate: `bus_backup_attribution_20260926/summary.json`.

Adaptive/fixed mean-minus-min common shift+78.566983/+74.503568 versus
centered-correction RMS2.576791/1.857657. Decisions switch39/96 and32/96;
helpful17/17, harmful16/15, with6/0 zero-gain switches. All-action centered
MSE changes-21.378392/-.386826, but selected-pair MSE increases2.396153/
7.231888. Better aggregate prediction error does not imply better errors
on the competing actions. Positive soft contributions+.456577/+.677895
are outweighed by negative-.835704/-1.021477; unchanged cases remain in
the denominator. Worst three contexts account for62.260%/80.003% of gross
soft losses, with no exclusions. Overall gains remain-.379127/-.343582.

Fixed seed7701/probe28 switches holding0->60: predicted mean-versus-min
chosen-action gap changes-1.586922->+5.410622, while paired soft outcome
is-43.968375 (conditional MC SE7.278327), halves-47.061518/-40.875231.
Adaptive seed7702/probe24 switches0->15, gap-.841797->+.493042 but
soft-23.035851 (3.613746), halves-23.952865/-22.118836. These are observed
ranking errors; half agreement is descriptive, not independent confirmation.
The min_shift control also increases selected-pair MSE1.947973/6.016174
and loses soft-.278812/-.316870 versus min. A constant training bonus is
not equivalent to a harmless post-fit shift.

Next target the source of incorrect action gaps: separate immediate-reward
and bootstrapped-continuation differences and inspect replay support before
changing updates. Do not repeat the failed generic native-gap penalty or
tune a veto on these losses. Current attribution identifies where outcomes
are lost, not the causal training mechanism or a validated repair. No new
training jobs or manuscript edits.

## 2026-09-26 First-hop backup mechanism attribution

The archived confirmation stores full returns but not individual rewards or
next states. Replay all192 fixed contexts/all8 suffix streams/all original
actions only until the focal bus's next transition (9216 first hops), with
one full-suffix exact-repeat smoke per source. Keep original predictions,
choices, joint actor, reward co-batches and archived complete returns. No new
training, outcome selection, altered noise streams or full-suffix campaign.

Split soft G into g_D(first reward), discounted next-action entropy and the
remaining discounted MC continuation. For each endpoint and the same fixed
competing pair, split Q-G into Q minus its one-step target plus endpoint
bootstrap minus MC continuation. Use saved4096 rolling targets, the original
target for frozen/native controls, original min_shift c and terminal masking.
These are endpoint diagnostics, not historical optimizer errors or a causal
identification of training failure. Mean/min, mean/min_shift, mean/native,
min_shift/min and rolling/frozen mean comparisons retain all contexts and
signed harmful contributions with day->seed aggregation, separately by
temperature. First-hop replays reuse inspected outcomes, not new confirmation.

Report replay support using the actual32768 training rows: exact first four
categorical fields, remaining fields scaled by training population std, the
nearest32 states' action distances/range, and joint state/action proximity
with action/60. No learned uncertainty interpretation, threshold or veto.
Missing category matches are counted and retained in detailed outputs;
proximity aggregates are omitted if a constituent context lacks matches.

Six scheduler GPU jobs, compatible GPU1/GPU2; one physical core/worker.
Recover after each complete replicate; recheck regenerated prefix predictions
and unchanged models. Checkpoints, detailed first hops and replay remain
remote; only progress/summary JSONs return. Paper unchanged. Return after
deployment/startup verification, without waiting for the campaign to finish.

Six tests passed locally in2.133s and on GPU1 in4.111s, including real CUDA
toy-event replay/resume and prohibition of optimizer updates. GPU2 loaded all
six mirrored source cohorts and reproduced each1536-branch source summary.
Submitted once as t100580--t100585, all running/retry0. t100580/581 use GPU2
GPU0/1 and physical cores6/7; t100582/584 use GPU1 GPU0, t100583/585 GPU1
GPU1, physical cores8--11 respectively. All six passed the archived full-suffix
exact-repeat smoke and first-hop repeat;48--54 branches/source persisted at
startup, nonempty recovery checkpoints~40--45KB. No original replay or model
checkpoint downloaded locally. Small-result auto-sync configured. Stop here
without reading formal outcomes or waiting for completion; no paper changes.

## 2026-09-26 First-hop attribution results

t100580--t100585 all done exit0/retry0:192 contexts/9216 first hops plus
six exact full-suffix smokes, zero updates and unchanged models. Runtime
2032.819--3620.134s/source. All36 result JSONs (862323 bytes) already synced;
no additional CSV/checkpoint downloads. Frozen-runtime aggregate saved as
`bus_backup_mechanism_20260926/summary.json`. Completion, finite values and
shared return contrasts agree with the archived confirmation.

Adaptive/fixed mean-minus-min soft remains-.379127/-.343582. The first
reward contributes+.051769/-.177761, next-action entropy-.003133/-.003707,
and remaining discounted MC continuation-.427763/-.162115. Within observed
harmful decisions, continuation accounts for85.418%/61.948% of gross soft
loss. This is a signed retrospective decomposition, not a variance fraction.

The predicted pair gap is+.342621/+.302176, but its mean-backup endpoint
one-step target gap is-.051384/-.185392. The extra predicted gap versus min
(.733564/.599282) splits into residual differences .627629/.502904 and
target differences .105935/.096378. Thus both endpoint Bellman inconsistency
and continuation error matter; these are not historical optimizer residuals.
Mean targets favor the old action in12/16 and8/15 harmful contexts, accounting
for75.597%/90.497% of observed gross negative contribution.

Fixed7701/probe28: predicted gap+5.410622, one-step target-16.247471,
full soft difference-43.968375 (first reward-5.759033, continuation-37.633358).
Fixed7702/probe13 is different: soft-30.172339 is almost entirely immediate
reward-30.234400, entropy+.062061, continuation~0. Do not ascribe every
failure to long-horizon bootstrap bias. All cases remain in the summaries.

Exact-category replay support averages5.146/4.760 rows per query;1/96 and
3/96 have no match. Every harmful query has at least one match; categorical
counts and nearest-action distances alone do not establish a coverage cause.
Candidate next: paired first-hop supervision from independent training states,
with matched-data ordinary TD versus action-gap TD controls and held-out days.
Do not fit these inspected confirmation queries, repeat the failed generic
native-gap penalty, or treat this attribution as a successful repair. No new
jobs or paper edits in this results-analysis turn.

## 2026-09-26 Paired first-hop TD repair screen

Freeze before execution: six sources, seeds7700--7702 crossed with adaptive/
fixed temperature. Fourteen new prefix days/source, times12000/24000/36000/
45000, grid0/15/30/45/60 plus original stochastic actor action, eight paired
first-hop draws/action. Namespace2000000000+(seed-7700)*100000; prefix
environment/action offsets0/1000, suffix offsets10000/20000+100*probe+rep.
At most2688 branches/source,16128 total; no new complete suffix campaign.
Retain the original day500 joint actor/alpha, route sigma1.5, reward mapping
and terminal transformation. Save reward/next state/action/log density per
branch; first branch repeats exactly. No old confirmation states enter fitting.

Both arms independently copy rolling_mean4096, including embeddings, with
the same frozen rolling_mean endpoint target. Labels average all eight
g_D(r)+gamma*(1-d)*(mean(target heads)-alpha*logpi) samples per action.
Both heads are trained; min-head deployment is identical. Ordinary level TD
uses equally weighted within-state action MSE; contrast TD centers each
head/state error before squaring, exactly half the independently uniformly
sampled pair-difference MSE (self-pairs included). Level loss equals contrast
loss plus squared state-mean error. This reuses a centered-loss idea with
paired one-step transitions, not a new uncertainty decomposition.

Each arm uses the same56 contexts, targets,4096 full-batch Adam1e-5 updates,
clip1 and checkpoints every256. No actor/alpha/target update, replay mixing,
native-gap penalty, tuning or early stopping. Only after both fits finish,
load the inspected confirmation's32 contexts/eight days/all8 archived suffix
draws. Primary contrast-minus-level soft/raw-system gains; both versus their
unchanged rolling mean start, rolling min, native and actor. Retain centered
and offset Q errors, all signed gains, seeds, temperatures and failures.
Same-data ordinary TD controls for the newly collected data. This is an
inspected-data repair screen; success would still require fresh confirmation.

One resumable GPU worker/source, compatible GPU1/GPU2. Persist each complete
replicate and model/Adam/completed arms; resume cached immutable source and
regenerate unfinished prefixes. Collect cyclic simulator garbage after each
replicate group; old reported70GB-class peaks motivate80GB RAM reservations
until live use establishes a lower peak. Detailed transitions/checkpoints
stay remote, only compact JSON results return. Paper unchanged; stop after
startup verification, without waiting for completion.

Seven tests passed locally in10.531s and on GPU1 in9.740s: both-head pair
identity/masking, terminal/entropy target, fresh streams, all-draw evaluation,
CPU/CUDA data-and-optimizer resume, immutable source, and test-label changes
leaving fitted weights exactly unchanged. GPU2 matched all six source/target
pairs and found every evaluation artifact before dispatch.

Submitted t100620--t100625 once, all running/retry0: t100620/621 GPU2 GPU0/1,
cores6/7; t100622/624 GPU1 GPU0, t100623/625 GPU1 GPU1, cores8--11. All use
CUDA/RTX3080Ti and distinct physical cores, passed exact first-hop repeat,
and persisted48--60 branches/source with~1.03MB checkpoints. Live configs
retain14 days/8 draws/4096 updates, not unit-test budgets. Small-result
auto-sync configured; no checkpoints downloaded or formal outcomes read.
Stop after startup, without waiting for data collection/fitting to finish.

## 2026-09-27 Paired first-hop TD results

All six t100620--t100625 done exit0/retry0:56 training contexts/2688 first
hops and8192 updates/source,16128 first hops/49152 updates total. Evaluated
all32 held-out contexts/source with archived8-draw outcomes; source unchanged.
Recorded elapsed11.236--15.757h/source. t100623 was RAM-evicted on GPU1 and
resumed on GPU2; retry0 does not mean uninterrupted execution. The80GB RAM
reservation and explicit GC did not prevent observed~110GB-class task peaks.
Forty-one JSONs were already synced; retrieved only the missing80-byte smoke
record from its original GPU1 host. No checkpoint/CSV downloads. All budgets,
loss identities, finite values, day aggregation, archived baselines and paired
return identities agree. Frozen-runtime aggregate: `bus_paired_td_20260926/summary.json`.

Adaptive/fixed-temperature contrast-minus-level soft mean(seed SE):
-.840540(1.078703)/+.149828(.520462), raw-5.004307(25.881727)/
-5.343469(3.579964). Fixed-temperature raw differences are negative in all
three seeds; its small positive soft mean comes from one seed, not consistency.
Contrast-minus-starting rolling mean soft-1.147749/-.589291 and raw+8.481949/
-8.936199. Fixed-temperature contrast loses both outcomes to its start in
all three seeds. Level-minus-start soft-.307209/-.739119, raw+13.486256/
-3.592730. Retain the mixed raw-system gains; neither arm passes the screen.

Contrast-minus-native soft-2.071513/-.529215, raw-8.946173/-2.601111;
contrast-minus-actor soft-1.451514/-.469450, raw-3.825584/-1.527900.
Level-minus-actor soft-.610974/-.619278, raw+1.178723/+3.815569 (mixed seeds).
Training contrast MSE level->contrast16.806870->6.113038 adaptive and
16.100672->5.629898 fixed, while held-out centered MC MSE worsens223.602274
->229.624659 and211.631742->242.927643. Both improve centered MSE versus
the starting model263.689002/261.934738, without control improvement.

This matched-data fixed-budget centered-TD repair fails to establish benefit.
Next use saved first-hop targets/outcomes to distinguish target-ranking failure
from learned-ranking failure with disjoint suffix draws. Also check the actual
two-head deployment: per-head centering leaves relative head offsets free,
while min-head action ranking is not invariant to independent head offsets.
This is a possible mechanism, not an established cause of these results.
No sweep, new simulation/training jobs or manuscript edits this turn.

## 2026-09-27 Cached target-ranking and head-offset diagnosis

Freeze before execution: all six completed paired-TD endpoints,32 inspected
confirmation contexts/source and all eight archived first-hop/full-return
draws. No fitting or simulator branches. Select an action by mean one-step
rolling_mean target on draws0:4 and score full soft/system returns on4:8,
then reverse. Keep both directions and their within-context mean; average
contexts within day, then days within seed. Folds do not add independent seeds.

For the starting rolling_mean and both fitted critics, retain raw min,
per-head action-centered min, head mean and both individual heads. Also
restore starting per-head means to each fitted centered head before min:
this changes offsets while keeping learned action contrasts fixed. No
return labels determine offsets or choose a head/mode. Compare each changed
deployment with its own raw min; retain original level/contrast gains and
target-selected gains versus level/contrast/start/min/native/actor. Record
head gaps, head selection, switches, centered target/MC-mean errors.

Use the completed frozen7701 checkpoint on GPU2, not its pre-migration
GPU1 copy. Reuse cached mechanism targets/returns, reproduce archived raw
decision gains, persist each context for resume. Small JSONs auto-sync;
checkpoints/details stay remote. Six scheduler GPU workers, GPU1/GPU2
eligible;4GB RAM each because there are no simulator branches. This is a
retrospective diagnostic, not repaired-policy evidence. Paper unchanged.

Six tests passed locally and on GPU1, including CPU/CUDA exact resume and
disjoint-fold selection. t101005--t101010 ran on GPU2/GPU1 CUDA, all done
exit0/retry0,32 contexts/source, zero updates/branches, archived gains exactly
reproduced. These cached jobs finished during startup verification; retrieved
only30 JSONs/616192 bytes. Aggregate: `bus_td_ranking_20260927/summary.json`.

Adaptive/fixed split-target-minus-actor soft+2.416913/+3.526282, raw
+41.489122/+39.616083; all three seeds positive in each direction for both
metrics. Target-minus-level soft+3.027887/+4.145561, raw+40.310399/+35.800513.
This establishes useful cached target rankings relative to these learned
choices, not target perfection or a deployable simulator-free selector.

Contrast mean absolute head gaps150.127647/98.594355 versus start5.360790/
5.648318; centering changes34.375%/37.500% of decisions. Contrast-centered
minus-raw soft+.984137/-.108636, raw-.027478/+.734483, not a consistent repair.
Restoring start offsets gives soft+.711481/+.824143, raw-.189397/+12.316631;
retain this mixed result and every head/mean mode in the aggregate.

Level-centered minus-level soft+1.119848/+1.200563, raw+10.547391/+9.064022,
both positive in all six sources. Relative to actor the means are soft
+.508874/+.581285, raw+11.726114/+12.879591, but each temperature has one
negative seed. Versus native, adaptive soft remains-.111125. Thus investigate
ordinary level TD with action-set head alignment on fresh days, retaining raw
min/native/actor controls; no further centered-loss tuning or success claim.

## 2026-09-27 Fresh head-alignment confirmation

Freeze before new simulation: seeds7700--7702, adaptive/fixed temperature,
eight fresh days/source, times12000/24000/36000/45000, eight paired full
suffix draws. Namespace2100000000+(seed-7700)*100000, prefix environment/
action offsets0/1000+day and suffix offsets10000/20000+100*probe+rep.
These streams are disjoint from paired-TD training and inspected confirmation.
All192 contexts are retained, with at most9216 unique action branches plus
six first-branch repeat smokes. No fitting or hyperparameter selection.

Use the frozen level-TD4096 endpoint. Its aligned action maximizes the min
of per-head action-centered predictions on the full original grid plus the
actual actor action, exactly the previous diagnostic rule. Controls: its raw
min, training-start rolling_mean raw/aligned, native SAC critic min, actor.
Save all six choices before each context's suffixes; simulate their selected
action union once per paired draw. This preserves all paired utility
comparisons while omitting unused-action return estimates. No MC selector.
Original joint actor/alpha/reward transform/route sigma1.5 remain fixed.

Primary: aligned level versus raw level/native/actor, both soft and raw-system
returns. Start raw/aligned and the alignment-by-fit interaction distinguish
new TD fitting from deployment effects. Average four contexts/day, eight
days/seed, then three-seed means/SE separately by temperature; keep negative
days/seeds. This confirms a frozen single-action rule on fresh evaluation
data, not an online/full-policy or original RE-SAC/dual-uncertainty claim.

Forty-eight scheduler GPU day shards, GPU1/GPU2 eligible, each with branch
checkpoint and cached source payload. Physical-core leases avoid same-core
collisions across these day workers. One day/process limits the observed
multi-day memory accumulation without changing simulator dynamics;64GB RAM
and1.5GB VRAM reserved per shard, cyclic GC after each branch. Only small
JSON results sync locally. Paper unchanged; stop after deployment/startup.

Seven tests passed locally4.759s and on GPU1 in5.335s, including CPU/CUDA
branch resume, selected-action deduplication, seed/day aggregation, disjoint
streams and physical-core leases. Submitted t101030--t101077 exactly once:
14 running/34 queued, all retry0 at startup check. GPU1/GPU2 each have seven
CUDA/3080Ti workers on distinct cores5--11; cards hold3--4 tasks each. All
six day0 sources passed the exact full-suffix repeat and persisted completed
branches; checkpoints~1.12MB include immutable source payload. Initial peak
RAM~4GB on GPU1/~1GB on GPU2 is not an established end-of-day peak. No formal
outcomes read, source checkpoints downloaded, or manuscript edits performed.

Placement update: user also permits node007/jtl311linux. Expanded only the
19 still-queued shards to these two nodes plus GPU1/GPU2; running work and
the original submission manifest remain unchanged. Requested16GB RAM based
on completed one-day peaks1.3--4.4GB; scheduler retains its live learned
resource estimates. Staged the frozen runtime and six required source inputs
directly between servers, with no local checkpoint download. Existing user
environments only: node007 sumo-lstm-py310,311linux LSTM-RL; added the node007
Python-path mapping and passed both mapping regression tests. No system
CUDA/cuDNN changes.

All seven confirmation tests passed on each new node. Native/start critic
predictions on all32 archived probes/source agreed within rtol1e-6/atol1e-4
(maximum absolute difference6.103515625e-5), with identical argmax choices.
This is tolerance agreement, not cross-device bitwise resume equivalence.
Startup verified16 node007 RTX2080Ti workers and two311linux RTX2080 workers:
CUDA, distinct physical-core leases, completed suffix branches and~1.12MB
branch checkpoints. After rescanning a transient GPU2 SSH error, t101062
also launched on311linux GPU1: CUDA, third distinct core, first probe and
initial checkpoint verified. Snapshot:13 done/35 running/0 queued, retry0
throughout; running placement GPU1=8,GPU2=8,node007=16,311linux=3. No new
scientific outcomes inspected or manuscript changes made.

Completion readout: t101030--t101077 all done,exit0/retry0. All48 day shards,
192 contexts,4112/4112 selected-action suffix branches and six exact repeats
are complete; configs match the frozen specification, models unchanged and
training_updates=0. Filled the missing node007 small result folders via the
scheduler result-sync executor; all240 local shard JSONs total401576 bytes.
No CSV/checkpoint downloads. The frozen runtime produced
`bus_head_alignment_confirmation_20260927/summary.json`; all reported metrics
are finite. Means/SE below average days within each of three training seeds.

Adaptive/fixed level-aligned-minus-level-min soft+.252354(SE.396051)/
+.727809(SE.340828), raw+.880217(SE5.476807)/+9.937654(SE3.146118).
Fixed gains are positive in all three seeds for both metrics; adaptive raw
gains are negative in two seeds. Against actor, soft+.404045(SE.454525)/
+.017671(SE.774068), raw+9.531330(SE3.685198)/+4.173818(SE9.510640).
Adaptive raw gains are positive in all three seeds; fixed actor comparisons
are negative in two seeds for both metrics. Against native critic, soft
+.438603/+.081178 and raw+10.560018/+7.874156, with mixed seed signs except
adaptive raw. Retain every day/seed and both temperatures.

The fixed TD raw-min endpoint loses soft.938968/raw10.137022 versus its
start; alignment recovers part of that loss. Fitted aligned minus starting
aligned is adaptive soft+.143136/raw+2.569784, fixed soft-.273930/raw-2.940658.
Thus fresh data do not establish a stable improvement over actor/native or
an advantage of the additional TD fit. Alignment changes action contrasts,
not calibrated Q levels, and this does not validate Q-underestimation repair
or original RE-SAC. Next isolate target-versus-fit ranking loss using existing
paired-TD diagnostics before expanding training; do not promote the starting
aligned control to a newly confirmed winner after inspecting these outcomes.
No new jobs, manuscript edits or hyperparameter selection this turn.

## 2026-09-27 Cached TD fit-versus-ranking diagnosis

Freeze before execution: all six paired-TD endpoints,56 training contexts
and32 original inspected held-out contexts/source. Reconstruct cached targets
with the actual frozen training target, terminal mask and next-action entropy.
No training or simulator calls. Compare start/level/contrast per-head level,
offset and centered errors; min/aligned/mean/individual-head action choices,
empirical target regret and split-draw target/return gains. Independent draw
halves select then score, both directions retained. Training halves are not
independent validation because fitting used all eight draws. Separate fitted
choice gap errors into prediction-minus-target and target-minus-full-return.
Keep negative gains, all modes/temperatures/seeds, no winner selection.
Average contexts within day, days within seed, three-seed mean/SE. Cached
held-out queries are not the fresh192-context confirmation. GPU scheduler
workers checkpoint each context; only small result JSONs return locally.

Six synthetic tests passed locally and on GPU1, including CUDA/CPU resume,
terminal/entropy accounting, disjoint-half scoring and negative gains. Initial
t101113--t101118 stopped in the target check: batched float32 inference differs
from the original one-transition kernel by up to.00056648. Automatic retries
also stopped before diagnosis; their remaining queued successors were cancelled.
Preserved the initial runtime/logs. Corrected only reconstruction batch shape,
not tolerances, cached labels or scientific endpoints. Real frozen7701 GPU1
preflight then reproduced training targets exactly and held-out targets within
2.8422e-14. Resubmitted six sources as t101134--t101139 from
`runtime_single_transition`, all four GPU servers eligible; no local/CPU jobs.

Five sources completed immediately. Adaptive7700 on311linux still differed
by.00030590 under single-transition inference; cancelled its queued retry,
retained the threshold and ran only that source on3080Ti as t101142. Final
cohort t101135--t101139,t101142: all done/exit0,336 training and192 old held-out
contexts, zero updates/branches, saved training losses verified and models
unchanged. All old attempts/retries are terminal. Pulled30 JSONs/1662141 bytes,
no source checkpoints or CSVs. Aggregate: `bus_td_fit_diagnostic_20260927/summary.json`.

Adaptive/fixed per-head centered target MSE falls on training from149.881/
54.141 to16.807/16.101, and held-out from107.375/90.585 to80.503/59.359.
Yet raw-min empirical target regret improves on training3.007/2.831 to
1.442/1.350 and worsens held-out4.011/4.107 to5.114/5.171. The held-out
change is adverse in3/3 adaptive and2/3 fixed seeds. Contrast fits reduce
training regret further to.482/.497 but held-out remains5.709/4.378.
Held-out half-target argmax agreement95.83%/87.50%; centered mean-label
sampling variance.817/.510 versus cross-half contrast signal112.463/90.287.
For level raw-min versus start raw-min, predicted gaps+1.430/+1.341 oppose
cached target gaps-1.104/-1.064; fit gap errors+2.534/+2.405. Lower global
MSE did not preserve held-out action ranking. Alignment/mean aggregation
does not consistently remove this defect across temperatures/seeds.

Prioritize action-ranking generalization, not another uniform value shift
or merely more draws/optimizer steps. Next use day-blocked folds of original
training data to distinguish overfitting from loss/decision mismatch before
choosing a training change. These are retrospective diagnostic findings;
the mixed fresh confirmation remains authoritative for utility claims.
No further jobs or manuscript changes this turn.

## 2026-09-27 Training-only paired-TD day folds

Freeze before fitting: original paired-TD14 days/56 contexts/source only,
seeds7700--7702 and both temperatures. Seven folds hold out days(2f,2f+1),
fit the other12 days/48 contexts. Same pre-fit rolling_mean4096 initialization,
not either fitted all-day endpoint. Same cached8-draw labels, network,
Adam1e-5/clip1, full-batch4096 updates and256-update checkpoints in each arm.

Compare level MSE, centered-contrast MSE, and a decision-gap candidate.
For each head/context, let a_star maximize the mean training target y;
v=max_a[Q(a)-Q(a_star)+y(a_star)-y(a)]_+. Decision loss=mean(v^2)+offset_MSE.
The offset term/coefficient1 matches level MSE exactly; only its contrast
term is replaced. Valid-action masks exclude padding in both maximizers.
Keep raw-min deployment primary and centered-min/head-mean secondary.

Read training/held-day target regret, gain over the identical initial rule,
switches and head level/centered MSE at every checkpoint including0/4096.
Retain every curve; no best-step selection, refit, new simulator draws or
access to either external confirmation set. Curves address overfitting and
loss/decision mismatch; they do not measure true environment return.
Average held days/folds within seed, then three-seed means/SE separately by
temperature; overlapping folds are not independent replicates.42 scheduler
GPU fold tasks, each with three paired arms;126 fits/516096 total updates.
Full data/model/optimizer/progress checkpoints stay remote; small JSONs sync.
After deployment/startup verification, return without waiting for training.

Deployment: t101145--t101186,42 tasks, zero retries at startup audit.
Placement: jtl110gpu8, jtl110gpu2 8, jtl311linux4, node00722; no local/CPU
training. Seven tests passed on every frozen remote runtime, including
baseline optimizer equivalence, held-label isolation and CPU/CUDA resume.
All six real sources loaded on every node; deployed budget4096/save256.
Remote runtime/progress inspection confirmed CUDA and nonempty latest.pt
for every task. At that inspection40 folds had completed all three arms;
the other two were at decision3584/3840 after completing level/contrast.
The subsequent scheduler snapshot was8 done/34 running with40 exit0
records; result synchronization was still catching up. No scientific
aggregate was read this turn. Auto-sync contains compact JSONs only;
checkpoints and per-day curves remain remote. Manuscript unchanged.

Completed t101145--t101186:42 done/exit0/retry0,126 fits,516096 updates;
all expected curves/configurations complete and finite, all CUDA. Automatic
sync was incomplete; fetched160 missing small JSONs/3057108 bytes. No
checkpoints or per-day details downloaded. Aggregate:
`bus_td_day_folds_20260927/summary.json`.

Held-day raw-min target regret (lower better), adaptive/frozen: start
3.006992/2.831283; level3.292426/3.837024; contrast3.596167/4.384989;
decision3.240066/3.883931. Decision-minus-level is-.052360 (seed SE.098933)
/+.046907 (.132514): mixed across seeds, no consistent improvement.
All three train-regret means improve, but all six held-regret means worsen
versus start. Frozen-temperature start-relative gains are negative in every
seed/arm. Adaptive decision improves early (256--1280), then loses that
advantage; every recorded nonzero frozen-temperature checkpoint is worse
than start for all three arms at the across-seed mean. Do not select an
early checkpoint retrospectively or treat folds as independent seeds.

Changing only the fitting objective did not repair cross-day ranking here.
Next proposed control: on these same training-only TD labels, compare
full-critic versus last-layer-only updates from the identical initial model
under matched objectives/budgets, to isolate representation-update effects.
No new jobs or manuscript changes in this result-analysis turn.

## 2026-09-27 Paired full-critic versus readout TD day folds

Freeze `bus_td_readout_folds_20260927/tasks.json`: the same3 seeds,2
temperatures,7 two-day folds and original14 training days/56 contexts.
Each GPU worker pairs full-critic and last-layer-only updates for level,
contrast and decision losses. Both start from the same pre-fit rolling_mean
critic, with fresh Adam1e-5/clip1, the same48 training contexts and cached
eight-draw targets,4096 steps and256-step saves. Rerun full controls on the
same worker to avoid cross-device comparisons.42 jobs,252 fits,1032192 updates.

Readout mode updates only critic.6.weight/bias across both heads; embeddings
and all hidden layers stay fixed, asserted at every save. Keep original actor
and source critic unchanged. Primary: readout-minus-full held-day raw-min
target regret at4096 for each loss, plus each versus its starting rule.
Retain all train/held curves and secondary metrics; no best-step selection,
refit, new simulation or external confirmation access. Average folds within
seed, then three-seed means/SE separately by temperature. Freezing also
changes function class and optimization; it does not isolate drift alone.

Full model/optimizer/data checkpoints and per-day details stay remote;
only compact JSONs auto-sync. All four GPU servers eligible, no local/CPU
training. Paper unchanged; return after deployment/startup verification.

Deployment t101198--t101239:11 tests passed locally and on every frozen
remote runtime, including full-control equivalence, last-layer Adam/frozen
parameters, held-label isolation and CPU/CUDA resume. All six real source
caches loaded on each node; formal4096/save256 budgets verified intact.
Startup snapshot:41 running,1 queued (t101238),retry0. Running placement:
jtl110gpu8,jtl110gpu2 8,jtl311linux3,node00722. All41 inspected workers use
CUDA, have nonempty full-state checkpoints and advancing optimizer steps;
full_level completed everywhere and readout_level reached2048--4096 steps.
Runtime trainable counts:24715 full versus130 readout. Remaining task is
capacity-gated, with no placement override. No outcome curves inspected,
checkpoint downloads or manuscript edits; stop without waiting for results.
Final scheduler snapshot: t101238 also launched;1 done/41 running,0 queued,
0 failed/retries. No completion wait or scientific outcome inspection.

Completed t101198--t101239:42 done/exit0/retry0,252 fits,1032192 updates.
All configurations/budgets/curves complete and finite, feature-freeze and
source-preservation assertions passed. All210 compact JSONs auto-synced
(7813353 bytes); no additional downloads. Frozen-runtime aggregate:
`bus_td_readout_folds_20260927/summary.json`. Runtime71.969--136.859s/fold.

Held raw-min target regret, full -> readout, adaptive/frozen:
level3.292426 ->3.599268 /3.837024 ->2.655651;
contrast3.596167 ->3.024722 /4.384989 ->2.864657;
decision3.240066 ->3.153668 /3.883931 ->2.863040.
Starting regret is3.006992/2.831283. Only frozen-temperature readout-level
has positive across-seed mean gain versus start:+.175632 (seed SE.047142),
with seeds7700/1/2:+.266044/+.107266/+.153585, a6.203% regret reduction.
Its gain versus matched full-level is+1.181373 (SE.339998), positive3/3.
All three frozen-temperature readout arms beat their full controls3/3, but
contrast/decision remain slightly worse than start at the mean. Adaptive
readout-level is worse than start3/3; preserve this negative boundary.

Freezing did not solve fitting: readout-level training head MSE remains
624.693/499.561 versus full-level15.269/14.654. Restriction changes capacity
and optimization as well as feature updates, so drift alone is not identified.
These are cached-TD development outcomes, not independent bus-return gains.
Next candidate: lock frozen-temperature/readout/level/4096, refit on all
original training days and compare with the unchanged starting rule and
matched full-level on fresh bus-day continuations under a frozen paired
return protocol. No further loss/epoch selection on these results. No new
jobs or manuscript changes in this result-analysis turn.

## 2026-09-27 Locked readout fresh-day confirmation

Freeze `bus_readout_confirmation_20260927/{refit_tasks,tasks}.json` before
execution. Candidate: frozen temperature, level loss, readout only,4096
updates. Three refit jobs (seeds7700--7702) use all14 original training
days/56 contexts/eight-draw targets. Pair full/readout from the identical
pre-fit rolling_mean4096, fresh Adam1e-5/clip1; every256 updates save data,
models and optimizer. Freeze original actor/source and assert fixed features.
Export one immutable model pair/seed before any new evaluation starts.

Twenty-four GPU day shards:8 new days/seed, times12000/24000/36000/45000,
8 paired full suffixes/action. Fresh namespace2120000000+(seed-7700)*100000,
prefix offsets0/1000+day, suffix offsets10000/20000+100*probe+replicate,
disjoint from paired-TD training and both prior confirmation cohorts.
Precommit readout/full/start/native raw-min choices and original stochastic
actor, using the same grid plus actor action; deduplicate their union before
suffixes.96 contexts, at most3840 distinct branches plus3 exact-repeat smokes.

Primary: readout minus start/full soft and raw-system returns, retaining
native/actor comparisons; secondary switching/harm and selected-Q signed
error. Average draws/contexts/days within seed, then three-seed means/SE.
No new loss/epoch selection or adaptive-temperature success claim. This is
fresh single-action utility evidence with unchanged future joint actor, not
full-policy or original RE-SAC/dual-uncertainty validation.

GPU1/GPU2/node007/311linux eligible; day workers use1 core,1.5GB VRAM,
16GB RAM (previous48 one-day peaks max4375.789MB). One process/day and
GC after each branch; checkpoint completed branches, never discard tails.
Only compact JSONs auto-sync; frozen model transfer is for server deployment,
not local archival. Paper unchanged; stop after deployment/startup checks.

Deployment: refits t101254--t101256 completed, full/readout4096 each for
all three seeds. Before confirmation, fixed NumPy2 reward-array pickles
unreadable on GPU1/NumPy1 by exporting means/m2 as numeric lists. Re-exported
from completed checkpoints with identical weights and no optimizer updates;
reward-transform equivalence tested. Explicitly staged the two changed files
after the staging TTL cached older code. Eight targeted tests passed on
each of the four nodes; all three bundles loaded on every node. Transfers
used in-memory tar streams, with no local checkpoint archival.

Confirmation t101262--t101285:24 running,0 queued/failed/training retries;
GPU1:7, GPU2:7, node007:8,311linux:2. One SSH-handshake launch failure
(t101269) recovered before evaluation began. All workers verified CUDA,
distinct physical-core affinity per node and branch checkpoints. All three
day0 workers passed the first exact-repeat branch; startup counts1--4
completed continuations, not completion. No utility outcomes inspected;
compact JSON auto-sync only. Stop without waiting for evaluations.

Completed t101262--t101285:24 done/exit0/retry0,96 contexts,1840/1840
deduplicated full continuations plus3 exact-repeat smokes. All models stayed
fixed, zero evaluation updates; configurations/budgets passed the frozen
aggregator and all day metrics are finite. Three refits also done/exit0/retry0.
Local135 compact JSONs total231366 bytes; explicitly synced9 missing result
folders while auto-sync caught up. No CSV/checkpoint downloads. Aggregate:
`bus_readout_confirmation_20260927/summary.json`. Day runtimes634--1370s.

Readout-minus-start soft return:-.037016 (seed SE.065605), seeds7700/1/2:
-.009223/+.060141/-.161966. Raw-system return:+.832816 (SE2.550711),
only1/3 seeds positive on both metrics. Readout-minus-full soft:+.852969
(SE.122105), system:+10.264243 (SE4.064448), both positive3/3. Full-minus-start
soft:-.889985 (SE.056514), system:-9.431428 (SE3.744099), both negative3/3.
Readout-minus-native soft:+.008571 (SE.307255), system:+5.946545 (SE5.891929);
readout-minus-actor soft:-.114919 (SE.275961), system:+1.652517 (SE2.153496).
No stable improvement over unchanged start, native selection or actor.

Readout changes only8/96 start choices; full changes45/96. Full-data training
level MSE averages532.122080 initially,502.547658 readout,17.488986 full.
Selected-Q signed MC errors: start-72.301900, readout-73.003504,
full-77.022301, native-143.214999; these compare each rule's own chosen
actions, not a same-action causal decomposition of bias. The fresh-return
test supports protection from the full-fit degradation, not a positive
repair of starting utility or Q underestimation. Preserve this failed
development-to-utility transfer; no confirmation-based epoch/loss selection.
Next proposed training-only diagnostic: sufficiently solve the fixed-feature
readout level objective and evaluate on original day folds, to distinguish
optimization limits from fixed-representation limits. Confirmation days stay
outside fitting/selection. No new jobs or manuscript edits this turn.

## 2026-09-27 Fixed-feature readout least-squares diagnostic

Freeze `bus_readout_least_squares_20260927/tasks.json`: frozen temperature
only,3 original seeds and7 two-day folds,21 GPU tasks. Each task uses48
training/8 held contexts from the original14 paired-TD days, identical pre-fit
rolling_mean initialization and cached eight-draw labels. Rerun full/readout
level Adam4096/lr1e-5/clip1 on the same worker, saving every256 steps.
Add a fixed-feature least-squares optimization benchmark, not a matched-budget
algorithm: unchanged float32 hidden features, float64 last-layer solve/score,
including bias, weight1/(contexts*actions_in_context). Per-head SVD retains
singular values above max(rows,columns)*eps32*s_max; minimum-norm change
from initial weights preserves their numerical nullspace. No ridge/cutoff sweep.

Read training level MSE and held-day raw-min target regret/gain for all four
methods, plus existing secondary head metrics. Record rank, conditioning,
normal residual, parameter change and float32-cast training loss. Average
seven folds within each seed, then three-seed mean/SE. No confirmation data,
new simulations, actor updates, best-step selection or environment-return claim.
Save all Adam curves and full optimizer/data/solver checkpoints remotely;
only compact JSONs sync locally. GPU1/GPU2/node007/311linux eligible,1 core,
1.5GB VRAM/4GB RAM per task. Paper unchanged; stop after startup checks.

Pre-submission numerical correction:311linux CPU test exposed5.96e-8
float32 feature batch-roundoff. An eps64 rank cutoff promoted it to extra
directions, weights reached1.05e7, and design/forward MSE differed1.413/2.294.
Set the rank rule to the feature precision eps32 before any formal jobs;
keep the solve/last layer float64. Test both rank deficiency and near-null
float32 noise against weighted NumPy least squares, retaining the actual
per-context forward check. This is a precision-resolved subspace benchmark;
its training residual is not an unrestricted feature-class lower bound.

Deployment t101300--t101320:21 GPU tasks, placement GPU1:6,GPU2:6,
node007:7,311linux:2; no local/CPU-server training. Five tests passed locally
and on all four frozen remote runtimes, including weighted NumPy equivalence,
rank/near-null precision, actual forward predictions, CPU/CUDA resume,
held-label isolation, unchanged features and matched Adam controls. All three
real training sources loaded on each node; formal4096/save256 budgets intact.
Startup queue21 running/retry0. Remote inspection found19 already complete
with full/readout4096 and the SVD solve;311linux had full4096/readout256 and
full3840 at its earlier snapshot. All21 CUDA, distinct per-node physical-core
affinity and nonempty checkpoints. No scientific outcomes inspected or
checkpoint downloads; compact JSON auto-sync only. Stop without waiting.

Completed t101300--t101320:21 done/exit0/retry0,172032 Adam updates,
21 SVD solves, zero simulator branches/external contexts. All frozen
configurations/budgets/curves complete and finite. Explicitly synced17 missing
small result folders while automatic sync progressed;147 local JSONs total
1507495 bytes, no CSV/checkpoint downloads. Frozen-runtime aggregate:
`bus_readout_least_squares_20260927/summary.json`. Runtime23.75--67.20s/fold.

Train head-level MSE, start/full/readout/least_squares:
532.121723/14.619481/499.560902/113.172696. Held raw-min cached-target
regret, same order:2.831283/3.880970/2.655651/5.327631. LS-minus-start
held regret:+2.496348 (seed SE.496542), seeds7700/1/2:
+1.692415/+2.393408/+3.403222; LS-minus-readout:+2.671980 (SE.469286).
LS training regret is also worse than start3/3:3.394905 versus2.831283.
Held LS level MSE3628.363708, centered MSE179.483403; centered-min and
mean-head held regrets4.943598/5.813851 also exceed their starting rules.

Of the418.949027 mean training-MSE reduction versus start, only.123524 is
action-centered error reduction (54.141001 ->54.017476);99.9705% comes
from reducing the within-state common offset error (477.980722 ->59.155219).
Adam under-optimization is present, but more complete level fitting neither
repairs training action ranking nor generalizes across days. Not merely a
raw-min head-selection effect; retain this negative result, not a utility claim.
Numerics are consistent: retained ranks45--58/65, maximum retained normal
residual4.80e-10, design-versus-forward MSE difference<=.001102 and
float32-cast difference<=.000858. Residual-floor scope remains the retained
precision-resolved subspace, not an unrestricted representation impossibility.
Next proposed training-only comparison: solve action-centered versus level
objectives in the same fixed feature space before further optimizer-budget
increases or new confirmation runs. No new tasks or manuscript edits this turn.

## 2026-09-27 Paired level versus action-centered readout solves

Freeze `bus_readout_contrast_solve_20260927/tasks.json`:3 frozen-temperature
GPU seed workers, each7 original two-day folds and14 paired level/contrast
solves;42 solves total. Same48 training/8 held contexts, pre-fit rolling_mean
features and eight-draw labels. No repeated Adam fits or new simulations.
Level reruns the previous least-squares control on the same worker. Contrast
centers valid-action features and labels separately within each context,
preserves equal context/action weights, excludes the bias column and holds
the output bias fixed. Subtract a within-context reference before the mean
to reduce cancellation of common levels; no held labels enter a solve.

Both objectives use the absolute rank cutoff max(rows,65)*eps32*s_max of
the uncentered weighted design including bias. Keep float32 features and
float64 solve/final layer; minimum-norm parameter change, no ridge/sweep.
The shared solver now accepts that explicit cutoff and returns unchanged
weights/undefined retained condition for rank-zero centered heads. Its
default level solution is unchanged; previous frozen runtimes stay intact.
Different objectives may retain different numerical subspaces. Fixing bias
does not fix state-dependent head means; record head-mean/gap shifts and
min-head switches alongside train/held MSE and raw-min target regret/gain.
Centered-min and mean-head decisions remain secondary, with no post-fit
alignment or best-mode selection. Seven folds average within each seed,
then three-seed mean/SE; no external confirmation/utility claim.

Persist each solved arm/fold and source data/models remotely; small JSONs
only auto-sync. Eligible GPU1/GPU2/node007/311linux,1 core/1.5GB VRAM/4GB
RAM per seed worker. Paper unchanged; stop after deployment/startup checks.

Deployment t101322--t101324:seed7700 on311linux/GPU0;7701/7702 on
node007/GPU0/GPU2. Ten tests passed locally and on all four frozen remote
runtimes: level-control equivalence, pairwise/centered objective identity,
weighted NumPy solution, reference-centered label-shift invariance, fixed
bias/features, rank-zero action-insensitive heads, CPU/CUDA resume and held
label isolation. All three training caches loaded on every eligible node;
formal7 folds/14 solves per worker intact. Startup queue3 running/retry0;
remote inspection already found all42 solves completed (3.35--6.19s/seed),
CUDA and distinct physical-core affinity with nonempty checkpoints. No
gradient updates, simulator branches or confirmation inputs; no scientific
metrics read, checkpoint downloads or manuscript edits. Auto-sync JSON only.

Completed t101322--t101324:3 done/exit0/retry0, all42 paired solves and
7 folds/seed complete. All15 result JSONs auto-synced (172269 bytes); no
additional downloads. Frozen-runtime aggregate:
`bus_readout_contrast_solve_20260927/summary.json`. Source/features/bias
assertions passed; zero gradient updates, simulator branches or external contexts.

Start/level/contrast training head-centered MSE:54.140999/54.017476/18.210390;
training raw-min cached-target regret:2.831283/3.394905/1.489374. Held centered
MSE:54.140999/179.483403/161.693836; held raw-min regret:
2.831283/5.327631/5.523938. Contrast improves training regret in3/3 seeds but
worsens held regret in3/3: contrast-minus-start +2.692655 (seed SE.694038),
per seed +1.558872/+2.566029/+3.953065. Contrast-minus-level held regret
is+.196307 (SE.197632), mixed across seeds. Held centered-min/mean regrets
4.815346/5.465306 also exceed their starting rules2.861689/2.688900 in3/3.

Contrast held head-level MSE rises to7468048.631194, head-mean shift MSE
7462139.179497 and head-gap shift MSE19910075.044856 despite fixed bias.
This exposes uncontrolled state-dependent levels, but is not a sufficient
explanation of all ranking degradation: centered/mean decisions also worsen.
Contrast ranks35--50/64; maximum retained normal residual5.23e-13,
design-forward objective difference2.37e-8, float32-cast difference.0003872.
The negative result is not a failed solve or a design/forward mismatch.

Direct contrast fitting repairs training action differences but not cross-day
generalization. Next candidate: a readout update regularized toward the
original critic, with any strength selection confined to inner whole-day
training folds and unchanged outer evaluation. Do not substitute another
unconstrained fit or extra optimizer steps. These remain reused-seed,
cached-TD development diagnostics, not bus-return or Q-underestimation
repair evidence. No new tasks or manuscript changes in this result turn.

## 2026-09-27 Nested-day output-anchored readout update

Freeze `bus_readout_trust_folds_20260927/tasks.json`:3 frozen-temperature
seeds x7 outer two-day folds,21 GPU workers. Reuse original14 days/56
contexts and cached eight-draw TD targets, pre-fit rolling_mean features,
actor and output biases. No simulations, external contexts or Adam updates.

Fit centered target MSE + lambda * mean[(Q_new-Q_start)^2], with equal
context/action/head weights. Unlike the previous centered-gap anchor, this
penalty includes state-dependent Q means. Anchor to the original readout
in float64 on unchanged float32 features. Solve minimum-change delta weights
by augmented SVD, normalizing both blocks by sqrt(1+lambda). Keep the prior
uncentered-design absolute precision cutoff; lambda0 exactly reruns the
previous unconstrained contrast solver. No numerical-cutoff search.

Fix strengths0,.01,.1,1,10,100 and unchanged-start option. For each outer
fold's12 training days, six inner folds hold out consecutive pairs from the
sorted training days, fitting40 contexts/scoring8. Select mean inner raw-min
cached-target regret; exact ties prefer start, then strongest penalty. Persist
selection before accessing outer outcomes. Refit all six registered strengths
on48 contexts and retain paired level-SVD control; selected uses the frozen
inner choice, including no update.36 inner +7 outer solves/fold,903 total.

Primary: selected-minus-start/contrast0/level held raw-min target regret and
gain. Retain every strength, centered/level MSE, head drift and secondary
head modes; no outer winner selection. Fold-average within seed before
three-seed mean/SE. These reused development days cannot independently
confirm policy utility or Q-underestimation repair, even with nested exclusion.

Save source data, inner scores/diagnostics and each outer model remotely;
checkpoint after every fit, resume without repeating completed solves.
Only config/runtime/progress/summary/completion JSONs auto-sync. Eligible
GPU1/GPU2/node007/311linux,1 physical core/1.5GB VRAM/4GB RAM per worker.
Paper unchanged; stop after deployment/startup checks, not completion.

Deployment t101330--t101350:21 tasks, placement GPU1:5,GPU2:5,node007:9,
311linux:2; no local/CPU-server jobs. Eight tests passed locally and on all
four frozen remote runtimes: weighted NumPy/zero-control equivalence,
full-output rather than centered-only anchoring, fixed features/bias,
nested-day/outer-label isolation, no-update/tie selection, CPU/CUDA exact
resume and seed-based aggregation. All three real sources loaded on each
node with56 contexts; formal six strengths/43 solves per outer fold intact.
Fixed a missing remote-details mkdir during tests, before formal submission.

Initial SSH handshake failures held t101345's resume scan and t101349's
launch. The latter launched on node007 through normal scheduling; forced
one fresh checkpoint scan for t101345 instead of bypassing checks or waiting
for its15-minute cached error. It then launched on311linux. Final startup
snapshot8 done/13 running,0 queued/failed,0 execution retries. Runtime files
confirmed CUDA and physical-core affinity for all21; checkpoints nonempty.
Twenty already had all43 solves completed during startup inspection; last
t101345 was42/43 with checkpoint1721134 bytes. No scientific outcomes read,
completion wait, checkpoint download or manuscript change. JSON auto-sync only.

Completed:21 unique seed/fold units done/exit0,903 solves, all nested budgets
and frozen configurations valid. Eight original launches were mislabeled failed
despite exit0 and complete43/43 logs: their terminal phrase was absent from
scheduleurm's success patterns. Retries t101352--t101359 resumed at43/43
and did no further solves; all sixteen original/retry logs have the terminal
message and no traceback. Preserve attempt history. Added a recognized DONE
prefix to the working entrypoint only, with complete/interrupted output tests;
the frozen runtime and scientific results remain unchanged. Synced13 missing
small result folders; all105 JSONs total592781 bytes. No checkpoint downloads.
Frozen-runtime aggregate: `bus_readout_trust_folds_20260927/summary.json`.

Start/level/unconstrained/inner-selected held raw-min cached-target regret:
2.831283/5.327631/5.523938/3.331852. Selected-minus-start:+.500570
(seed SE.455315), seeds7700/1/2:+1.408547/+.106725/-.013563.
Selected training regret2.534225; held centered MSE57.039180 versus
start54.140998. Selected centered-min/mean held regrets2.989548/3.026003
also exceed their starting means2.861689/2.688900. Selection counts:
start1,lambda0:1,.01:1,.1:0,1:4,10:6,100:8. Seed7700 folds3/5
choose0/.01; inner gains+.596465/+.367447 become held gains-7.435630/
-2.604351. The selected rule's held level MSE93740.094758 is dominated by
fold3 (1957190.424). This is not a consistently successful constrained update.

Fixed lambda100 changes only2/168 held-context actions: mean gain+.023252
(SE.016930); do not promote this inspected secondary endpoint to a confirmed
winner. Positive-strength solves are numerically consistent: outer maximum
design-forward objective gap1.13e-12 and float32-cast gap.000269; every
regularized training objective improves. Output anchoring limits drift, but
inner regret selection does not ensure cross-day improvement. Next diagnostic:
relate frozen-feature support/extrapolation and per-day action-switch losses
using the existing training caches, before another strength search or fresh
return confirmation. Reused cached TD targets are not actual utility evidence.
No new experiment tasks or manuscript edits this turn.

## 2026-09-27 Frozen-feature support and switch-loss diagnosis

Freeze `bus_readout_support_20260927/tasks.json`:21 GPU jobs use the completed
nested-readout checkpoints in place,3 seeds x7 folds. Diagnose56 contexts/fold
(48 training/8 held),1176 fold-context records,168 unique held contexts total.
All prior arms, selected rule and start stay fixed. No refits, inner-model
reconstruction, simulator calls, external confirmation or new action filters.

On each fold's training features, compute weighted SVD geometry for full
features including bias and action-centered features excluding bias. Both
retain the previous uncentered-design eps32 absolute cutoff. Leverage is
sum((z V_r/s_r)^2); null fraction is squared residual outside that retained
span divided by squared feature norm (zero vector ->0). Average heads/actions;
also retain switched-action-minus-start feature-gap leverage/null fraction.
These precision-resolved geometric measures are not calibrated uncertainty.

Use training-context midrank percentiles and strict training95th-percentile
flags separately for full/centered leverage/null fraction. Held labels and
features never fit these references. Per seed, relate held percentiles to
positive switch harm and absolute predicted-minus-target action-gap error
using Spearman correlations; constants remain undefined. Report all signed
gains, action choices, harmful-switch counts and additive harm contributions
above/below the fixed reference, including unchanged choices and adverse days.
Aggregate days within seed, then three-seed mean/SE, never folds as seeds.

Reproduce each saved arm's train/held regret, gain and level/centered MSE;
assert all actor/critic weights unchanged. Checkpoint after every context;
all-context details and source weights remain remote. Only small JSONs,
including8 held-context diagnostic records/fold, auto-sync. Jobs use the
GPU node owning each source checkpoint (GPU1:5,GPU2:5,node007:9,311linux:2),
1 physical core/1.5GB VRAM/4GB RAM each. No checkpoint movement to local.
This is retrospective mechanism diagnosis, not causal or utility confirmation.
Paper unchanged; return after startup checks without waiting for completion.

Deployment t101375--t101395:21 scheduler GPU tasks at the source owners,
GPU1:5,GPU2:5,node007:9,311linux:2. Eight tests passed locally and on each
frozen remote runtime: SVD trace/unseen directions, strict training references,
label-free geometry, signed/undefined associations, source metric reproduction,
CPU/CUDA resume without repeated contexts/refits, and seed-based aggregation.
All21 real completed source checkpoints loaded in place before submission.
Startup snapshot21 running/0 queued/0 retries. One remote startup pass already
found all56 contexts completed for every task, CUDA, distinct per-node core
affinities, nonempty1958666-byte checkpoints and source-metric checks passed.
No diagnostic outcome metrics inspected or checkpoint transfers; zero refits
and no manuscript edits. Only compact JSON auto-sync; stop at deployment.

Completed t101375--t101395:21 done/exit0/retry0,1176 fold-context records,
168 unique held contexts. Frozen-runtime aggregation passed the complete
coverage/configuration checks; every source-metric reproduction and unchanged
model assertion passed, zero refits/branches. Synced16 missing result folders;
all105 small JSONs total1418585 bytes. No checkpoint/details downloads.
Aggregate: `bus_readout_support_20260927/summary.json`.

Selected update switches22/168 actions,10 harmful: mean target benefit.113643
versus harm.614213, net gain-.500570 (seed SE.455315). Held target regret
remains3.331852 versus start2.831283. Geometry does not reliably isolate these
losses: level/contrast leverage flags143/89 of168 contexts, catching7/4 of10
harmful switches. Contrast-null flags60 contexts but catches only1 harmful
switch and6.6163% of total harm. The selected harm Spearman means for level
leverage/contrast leverage/level null/contrast null are-.065534/-.048493/
-.039867/+.010940, each defined for only2 seeds (7702 has no harmful switches).
Gap-error correlations are also weak and not consistently positive by seed.

Seed7700/context47 loses18.905753 despite all four high flags being false;
its predicted gain is+.655377. Level-null median7.42e-31 is numerical-scale,
not substantive unknown-state evidence. Thus these frozen geometry scores do
not justify a new action gate; this does not exclude every coverage mechanism.
Pause further fixed-feature readout/threshold sweeps on these reused samples:
neither this diagnostic nor the earlier fresh-day confirmation establishes a
stable repair. Retain this branch as negative evidence before choosing a new
online-training intervention. Cached-label associations are not actual bus
utility or calibrated epistemic uncertainty. No new tasks or manuscript edits.

## 2026-09-27 Online target/actor aggregation factorial

Freeze `bus_online_aggregation_20260927/tasks.json`: fresh seeds8200--8202,
four K2 arms min_min/headwise_min/min_mean/headwise_mean (target_actor).
Earlier500-day diagonal controls changed both factors; this adds the missing
crossed interventions in online training, not another cached-readout repair.
Identical initial networks and paired environment/action/replay RNG streams
within seed; each seed's four arms share a GPU host to avoid hardware-block
confounding. GPU1/GPU2/node007 each host one complete seed block;311linux is
currently CPU-saturated. No local or CPU-server production work.

Reuse the corrected high-noise bus recipe (route_sigma1.5), shared embedding,
batch2048, replay1000000, Adam1e-5, tau.01, clip1, one update/5 transitions,
1024 critic/512 actor updates/day for500 days. Only temperature additionally
changes: freeze its optimizer from initialization, so all arms use alpha.6
throughout. This coefficient is the existing cap, not newly tuned. No norm,
BC, disagreement or fitted correction. Twelve runs:6144000 critic/3072000
actor updates and6516 full simulator days including warmup and evaluation.

Primary: headwise-minus-min target raw bus return with the min actor fixed,
averaged over days425/450/475/500 and two fixed unseen scenarios within seed,
then three-seed paired mean/SE. Secondary: target effect with mean actor,
both conditional actor effects, factorial interaction, joint contrast, all
checkpoints and late-minus-day100. No peak/strength/seed selection or budget
extension. Log gamma*(1-done)*(mean(next heads)-min(next heads)) and its
actually applied deduction. These measure the aggregation operation, not
calibrated underestimation; return gains would still require a separate
matched-objective Q calibration before claiming a Q-bias repair.

Reuse complete replay/network/optimizer/RNG day-boundary resume, checkpoints
at days0/1 and every10 days (at most10 days replayed after interruption).
Only compact JSON results auto-sync; full CSVs/checkpoints stay remote.
Eight tests pass locally, including both existing diagonal update identities,
factor isolation, CPU/CUDA exact resume, interrupted-day recovery, JSON-only
sync scope and seed-based interaction accounting with negative/missing cells.
Paper unchanged; deploy, verify startup and stop without waiting for results.

Deployment t101402--t101413:12 scheduler GPU jobs, all running/retry0.
Seeds8200/8201/8202 use GPU1/GPU2/node007 respectively, four arms/host,
1 physical core/1GB reserved VRAM/4GB RAM per job. All eight tests also pass
on each frozen remote runtime; real environment initialization confirms28
state fields, CUDA and formal500/1024/2048 budgets. Startup inspection finds
distinct physical-core affinities, live CUDA processes and day0 checkpoints
1762746--1767802 bytes for all12. Each seed's four day0 evaluations match
exactly. Runtime versions differ across hosts but not within a seed block.
Compact startup evidence: `bus_online_aggregation_20260927/startup.json`.
No result interpretation, checkpoint download or manuscript edits; stop now.

Completed/analyzed2026-09-28:12 original tasks done/exit0/retry0; server logs
confirm day500/512000 updates and the terminal message, no traceback/resume.
All6144000 critic/3072000 actor updates and504 evaluations accounted for;
frozen configs, warmups and within-seed initial evaluations match. All72
compact JSONs already auto-synced,668562 bytes; no additional data downloads.
Frozen-runtime aggregate: `bus_online_aggregation_20260927/summary.json`.

Late raw-return means for min_min/headwise_min/min_mean/headwise_mean:
-452407.698/-439218.347/-456685.597/-450422.810. The prespecified target
effect with min actor fixed is+13189.352 (seed SE3243.983), seed8200/1/2:
+6922.439/+17776.985/+14868.631. Relative to the baseline magnitude this
is2.9154%; approximate df2 t95% interval[-768.380,27147.083]. This is a
positive three-seed direction, not a conventional95% significance claim.
With mean actor, the target effect is+6262.787 (SE4549.641), one seed adverse.
Mean-minus-min actor effects are-4277.898/-11204.463 at min/headwise target;
the interaction is-6926.565 (SE1312.409), negative in all three seeds.
Changing both factors jointly gives only+1984.889 (SE9282.984), mixed seeds.

The min actor/headwise target arm's late-minus-day100 return improves in
two seeds but declines16322.803 in8202; do not claim universal prevention
of degradation. Its late training Q mean is+103.085 versus min_min-72.692;
this alone is not Q-minus-return calibration. The positive primary isolates
a useful target-aggregation intervention under this fixed-temperature recipe,
not a validation of the norm bonus, statistical uncertainty decomposition,
or the full historical RE-SAC bundle.

Consolidation rather than another sweep: historical K10 bus evaluations on
seeds32/40 retain gains+62182.525/+9629.960 over their SAC controls, verified
against local operational JSONs. Keep the earlier archive/protocol limitations.
The manuscript's B0 is a separate five-seed configuration; v5/v5b/v6b are
single-seed, environment-specific references. Do not combine these and this
K2 factorial into one purportedly identical winning algorithm. Existing
positive application results and this narrower mechanism result can support
main-text evidence, while negative repairs remain recorded. No new jobs,
best-checkpoint selection, manuscript edits or claim of submission readiness.

## 2026-09-28 Underestimation comparison proof and evidence consolidation

Added `UNDERESTIMATION_BRIDGE.md` and external Lean target `Underestimation`:
finite weighted twin-min bias identity, differential-noise scaling, exact
over-penalty from confusing identified total/epistemic variance, conditional
repair without overshoot, and stationary scalar bootstrap propagation.
The existing counterexample now requires distinct inputs and no longer calls
fixed penalties universally necessary. These do not identify the legacy
norm/disagreement proxies as a valid decomposition.

`bus_underestimation_evidence.py` consolidates all three seeds and both
temperatures from existing backup-confirmation/calibration JSONs, retaining
the negative decision results. Output: `bus_underestimation_evidence_20260928/summary.json`.
No training, new simulator trajectories, remote retrieval or manuscript edits.
The MuJoCo design separates iid from persistent dynamics noise at matched
marginals; implementation and numerical preregistration precede deployment.

## 2026-09-28 Observed persistent-actuation pilot

Freeze `mujoco_persistent_bias_20260928/tasks.json`: HalfCheetah-v2/Hopper-v2
on the existing Brax spring backend, seeds8300--8302, clean/iid/persistent,
K2 min-target/min-actor versus headwise-target/min-actor. This is36 runs.
Each seed's12 cells share one host: GPU1/GPU2/node007 respectively; backend
versions differ across hosts, not within a paired seed block. No local or
CPU-server production work. No legacy norm, BC, LCB, EMA or anchor components.

Noise is additive sigma*clip(N(0,1),-2,2) per actuator, sigma=.3 for iid and
persistent, with refresh periods1 and25. Clean has sigma0, period25. The
current shock, remaining refresh phase and remaining episode time are observed
in every cell; phase and shock belong to full simulator/checkpoint state.
Apply clip(command+shock,-1,1), including actuator cost in raw simulator reward.
Replay and policy entropy use command actions. All APIs use the same law.
The1000-step horizon is a true observed terminal condition, not a continuing
task truncation. These are modified finite-horizon tasks, not official MuJoCo
benchmark scores. Persistent noise need not worsen conditional uncertainty:
current shocks are visible, so the hypothesis may fail or reverse.

Per run:131072 collected transitions,8192 uniform warmup,32 collection and
32 updates/block thereafter, batch256, replay131072, hidden128, three critic
layers/two actor layers, Adam3e-4, gamma.99, tau.005, fixed alpha.2. Both
critics and actor receive122880 updates; totals4718592 transitions and4423680
updates of each type. Carry simulator state across blocks, with resets only
at physical termination or the stated horizon. Seed namespaces separate
initialization/collection/replay/evaluation/MC; evaluation never advances
training state or RNG. Same seeds pair treatments but policy-induced states
can diverge. No tuning, replacement seeds, best endpoint or budget extension.

Evaluate at65536/131072, with final endpoint primary: eight complete raw-return
episodes and16 Q queries per run/checkpoint. Queries follow64-step stochastic
policy collection blocks, independently of Q or return values. Compare policy
mean/sample root actions,16 paired full soft-return continuations/action,
no root entropy, no learned bootstrap and no omitted terminal tail. Primary
min-head signed error and raw return; retain mean-head error, absolute/sampled
squared error, action-gap error, head deduction, MC variance and MC SE.
Average queries inside run, then paired differences over the three training
seeds. Prespecify persistent-minus-iid interaction of those algorithm contrasts.
These are each checkpoint's own-policy occupancy errors, not identical-query
causal comparisons between learned policies. Sampled MSE includes MC noise.

Full model/optimizer/RNG/replay/simulator checkpoints every8192 steps and
at evaluation query boundaries; resume the next training block or unfinished
query. Keep only latest.pkl remotely. Auto-sync JSONs only. Smoke mode uses
production network/batch/replay shapes with128 steps and horizon8, in separate
directories and seed8999; its scores are not scientific results. Deploy only
after synthetic resume/target tests and real-physics CUDA smoke checks, then
verify production startup and stop without waiting for completion.

Deployment binds each worker to a distinct physical core within its host block:
GPU1/GPU2 use logical CPUs0--11 (one sibling per physical core), node007 uses
CPUs16--27 (twelve different physical cores). Other projects remain untouched;
this is process affinity, not exclusive reservation of the whole host.
Resource estimates use a separate formal-run family from the short smoke jobs.

Deployment: eight focused tests passed locally and in all three production
interpreters; CUDA smoke tasks t101772--t101774 completed with exit0. Formal
tasks t101776--t101811 are submitted via scheduleurm. The startup receipt is
`diagnostics/mujoco_persistent_bias_20260928/startup.json`: GPU1/GPU2 each have
eight running and four queued, node007 has twelve running. All28 running jobs
have CUDA runtime records, distinct within-host physical-core affinities and
nonempty checkpoints at16384--32768 transitions. GPU1/GPU2 pack four per card;
node007 has two/two/four/four of this cohort across GPUs0--3. A transient SSH
launch/readback failure recovered; no training retry was recorded. This is
deployment evidence only; await the complete36-run final-endpoint readout.

## 2026-09-28 Fresh-day calibration of the online bus factorial

Freeze `bus_aggregation_calibration_20260928/tasks.json`: all four completed
online-aggregation arms, seeds8200--8202, day500 only, eight fresh stochastic
bus days per source. This is96 one-day GPU evaluation jobs,24576 uniformly
selected decision queries and zero optimizer updates. Day500 is the retained
endpoint, not a reconstruction of the previous425/450/475/500 primary window.
Each source stays on its original host; read checkpoints without modifying
them. New code/runtime, result and recovery paths are separate from both old
bus training and the currently running persistent-actuation cohort.

Fix alpha.6 and the actual source joint policy. Compute complete per-vehicle
soft returns without root entropy or a learned tail, including transformed
terminal zero rewards. Freeze each checkpoint's replay distribution and use512
independent co-batches of2047 rewards with the focal reward included in the
sample-std normalization. Retain two256-batch halves as normalizer sensitivity.
No fitted correction, best query/day selection, new training or tuning.
Use seed namespace2300000000+(seed-8200)*100000, with day offsets0/1000/2000
for environment/action/query streams and3000 for shared within-run co-batches.

Primary: headwise-minus-min target with min actor, signed native Q-minus-MC,
absolute seed-level mean bias change and fresh-day raw return. Retain all
conditional actor/target contrasts and interaction, head min/mean errors,
sampled MSE and mean-minus-min deduction. Average queries within day, days
within source, then contrast three paired training seeds. These are own-policy
occupancy errors in checkpoint-native reward units, not a common raw-reward Q
objective, repeated-state conditional means, action-gap calibration or causal
mediation. A less-negative error that overshoots zero is not a successful repair.

One day/process bounds known simulator memory growth. Save immutable lightweight
model/normalizer state before simulation; restarting an unfinished day replays
the same streams. Source replay and query details stay remote; only compact
JSONs sync. Every new task waits for all twelve persistent-actuation completion
JSONs for its host, then normal scheduler capacity gates. Thus it cannot compete
with that host's current cohort, including its queued tail. No paper edits.

Deployed t101813--t101908:96 queued,32 per host, all blocked by the actual
predecessor completion-file gate at the recorded inspection. Local frozen
runtime:8 tests pass including CUDA evaluation/resume. Each remote production
interpreter:7 CPU tests pass,1 CUDA test deliberately skipped while the GPU
cohort runs. All12 source checkpoint files exist remotely; no checkpoint was
downloaded. Queue receipt records previous cohort25 running/6 queued/5 done
and unchanged commands. New production evaluation has not started yet.

## 2026-09-28 Completed aggregation and persistent-actuation readout

All36 training and96 evaluation jobs have exit0, no retries, CUDA records and
successful server log endings. Retrieved only missing compact JSONs (~255KB),
not replay, query CSVs or checkpoints. Bus summary covers96 days/24576 queries.
The new `mujoco_persistent_bias_readout.py` reconstructs all1152 query roots and
36864 continuations, verifies both frozen endpoints and budgets, and retains
every seed. Two focused readout tests pass. Post-hoc MC precision corrections
are explicitly secondary and preserve negative unbiased estimates.

Bus primary return gain+12252.11 (all three seeds positive), absolute seed-level
mean bias170.47 ->11.43. HalfCheetah calibration improves across all nine pairs
without reliable utility gain; Hopper clean/persistent return improves with
overshoot, and iid is mixed in utility. No persistent-specific return advantage
is established. See `AGGREGATION_SYNTHESIS.md` and both cohort summary JSONs.
Keep n=3, native reward scales, old negative evidence and historical full RE-SAC
distinct. No new jobs, selected checkpoint, manuscript edit or extended budget.

## 2026-09-28 Integration and exact repair boundary

`aggregation_evidence.py` reads the three completed summary JSONs, retaining
13 distinct task/endpoints and39 paired seed records. Its tables separate
online late-window bus returns, fresh-day500 calibration and both Brax endpoints.
Bias changes are decomposed exactly into common head-bias change and clipping
deduction reduction. Eleven focused tests pass; no new sampling or retrieval.

External `Underestimation.lean` adds the necessary-and-sufficient absolute-bias
repair interval, action-gap error identity, common-offset invariance and a
reference-margin ranking condition. `lake build Underestimation` passes.
These clarify beneficial small overshoot versus harmful overcorrection and
why value-level gains alone do not establish decision gains. No paper edits.

## 2026-09-28 Algorithm and baseline lineage

`ALGORITHM_ALIGNMENT.md` records actual bus/JAX target and actor operations,
B0 per-environment launch settings and the K2 intervention. The actor sign is
correct; positive bus norm bonus and negative JAX norm shift are different
operations. Historical bus SAC already uses headwise/mean aggregation. The
integration tool now exposes its counterpart in the new factorial separately:
headwise_min minus headwise_mean is+11425.51+/-9445.14 on fresh-day500, with
two positive seeds and one negative. This is not the primary target contrast
or a reproduction of the old trainer. Twelve analysis tests pass. Existing
full-agent reward gains and adverse/mixed component results are retained.
No new simulation, training, remote retrieval, source-trainer or paper edits.

## 2026-09-28 Reproducible main evidence blocks

The existing integration entry point now generates `main_results.md`: all16
historical bus operational runs (eight variants, seeds32/40, ten days each),
the current four-arm K2 calibration, and all six final Brax transfer cells.
Historical service and reward metrics are recomputed from the160 stored daily
rows, then averaged within training seed. Legacy seed8 is not mixed into that
operational protocol. Paired seed SE is distinct from rollout variation.
Earlier32-day component negatives remain in the same output. Fifteen analysis
tests pass. Only local compact JSONs were read; no training or manuscript edits.
The conditional12-run K10 interaction design is described in
`ALGORITHM_ALIGNMENT.md`; no seeds were registered and no jobs submitted.
