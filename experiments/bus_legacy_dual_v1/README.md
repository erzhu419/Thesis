# RE-SAC and CS-BAPR Dual Actions

The original holding studies and their fixed-budget endpoints are preserved.
This branch fills the missing holding-plus-speed cells for both the native
headway reward and the cooperative passenger-time reward. All operating
statistics use independent passenger accounting.

RE-SAC and native CS-BAPR use five paired training seeds and their original
nine test cases. The repaired symbolic CS-BAPR cohort uses its two original
seeds and separate nine test cases. Matched ensemble, MLP and non-symbolic
controls receive the same dual actions. These cohorts are not pooled.

Forty-eight new 100-episode training jobs are independent. Each policy releases
only its own nine evaluations. Contract and short-run gates are separate for
the three cohorts. All CPU jobs can run on node001 through node007; none is
pinned to node311 or requires a GPU.

The legacy RE-SAC policy retains its 0..60 actor coordinates. The second one is
decoded into a 0.8..1.2 speed ratio. CS-BAPR retains normalized actions and stores
both coordinates in replay. The frozen SINDy data identify only holding effects:
symbolic consistency therefore applies only to holding, not an invented speed
reference. The speed output is trained by the RL objective.

Zero-holding fixed-speed controls are diagnostics, not learned dual policies.
The original holding endpoints are references, not newly rerun experiments.
Analysis stays on the server; only the compact summary is retrieved. Existing
code, policies, data and thesis text are untouched. All 48 formal policies and
459 evaluations are complete, with 432 holding evaluations reused.

The recorded reward field describes the evaluator's return, not a universal
headway score. The RE-SAC evaluator retains its original headway reward even
for passenger-trained policies; CS-BAPR evaluation follows the selected reward.
Passenger comparisons use independent operating cost, not overlapping returns.

Dual passenger-trained RE-SAC reduces cost versus its matched ensemble control
by 20.09%, 5.46%, and 14.27% in normal, demand-surge, and fast-switch scenes.
Each scene improves in four of five paired training seeds. Headway-trained
dual RE-SAC reduces plan error by 11.32%, 3.57%, and 5.93% versus that control.
CS-BAPR passenger-trained dual policies improve upon their holding parents,
but the NAU and symbolic components do not show consistent additional cost
benefits over their respective dual controls. The two CS cohorts stay separate.
