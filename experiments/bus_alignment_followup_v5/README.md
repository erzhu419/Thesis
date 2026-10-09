# Bus alignment follow-up v5

## Diagnosis

The existing CS comparisons did not reproduce the principal comparison in the
current transit paper. That paper compares the stabilization package against
pristine BAPR, with symbolic losses disabled. NAU versus an already stabilized
MLP is a smaller, different comparison. Its selected checkpoints follow 500
episodes; this thesis campaign uses the fixed 100-episode endpoint. The paper
reports static and sustained disturbances, not just the former short pulses.

The recent centered/no-NAU/SINDy gains are from RandomLoad, not the bus benchmark.
Its fixed linear dynamics and observed load admit a strong fitted reference.
This result cannot establish the same improvement for bus holding.

The previous bus adaptation also had a specific representation conflict:
`constant + mean(state) - mean(projected_state)` makes the complete deterministic
mean constant at projected headways, regardless of waiting load, dwell or speed.
On 192 resident identification holdout states per seed, the fitted reference at
the same projection spans about 19.36--48.60 seconds and 14.34--49.66 seconds of
holding. Its waiting/dwell/speed derivatives are nonzero. A constant projected
actor cannot match those directions. The new actor learns a contextual baseline
at the projection while retaining a zero neural residual. The projection is an
architectural anchor, not a claim of a unique physical bus equilibrium.

For MC-WM, the previous headway run's two joint75 seeds choose mean speeds about
1.186 and 0.820. Actor states already mix source-online and target-offline data;
that suspected mismatch is absent. This follow-up tests a speed-distribution
prior and a gradient-norm cap separately, equally across all three learners.

## Experiment

- CS original recipe: eight new pristine-BAPR trainings; sixteen stabilized
  holding/dual endpoints reused. Both rewards, two seeds, 100 episodes.
- CS context repair: eight new dual-action trainings with/without symbolic
  penalties; eight constant-baseline endpoints reused. Both rewards, two seeds.
- CS evaluation: six demand schedules and three fresh paired test days, 720
  learned-policy evaluations plus 36 zero-holding speed diagnostics.
- MC headway: speed KL weight 0.1 or gradient clip 10, three learners, two source
  scenes and two seeds: 24 trainings and 480 direct/guarded SUMO evaluations.
  Reuse each seed's 50,000-update offline initializer and compact resident data;
  each new run receives the same 25,000 source events.
- Fourteen short trainings/evaluations and three interface contracts precede
  independent family gates. Total scheduled jobs: 1,305 including gate handoff.

All formal training seeds remain in the summary. Existing CS and MC protocol
families, rewards, and direct/guarded executors are analyzed separately. Static
5x/10x and a 600-second 50x burst supplement former pulses; the latter is not a
static-50x replication. MC test days were used previously, so this is an
exploratory repair comparison. Newly demonstrated improvements are not assumed.

## Execution

All experiment changes are confined to this directory. Existing projects,
checkpoints, completed campaigns and thesis text are untouched. Jobs can use
node001--node007; all request zero VRAM. Separate training seeds and all completed
policy evaluations are independent; a policy's evaluations wait only for its
training and its family's short-test gate. Analyses run on server storage; only
the compact scalar summary returns locally.

Commands: `python3 launch.py stage`, `python3 launch.py contracts`,
`python3 launch.py submit`, `python3 launch.py collect`.

## Completed Results

All 40 formal trainings and 1,236 evaluations are complete. See `RESULTS.md`
for the separate CS recipe, CS context and MC headway conclusions. Retain the
CS stabilization-versus-pristine comparison and MC gradient-clip candidate;
retain negative context and speed-prior evidence without pooling protocols.
The server-generated `summary.json` includes every seed, matched-case contrasts,
seed-averaged endpoints and final training diagnostics. No new training or thesis
edits were performed during completion analysis.
