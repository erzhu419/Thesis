"""Small local-only background, support and rank audit for completed v15."""
import json
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
SOURCE = HERE.parent/'mcwm_closed_loop_training_pairs_v15_20261021/training_pair_summary.json'
v = json.loads(SOURCE.read_text())
unique, seen = [], []
for pair in v['pairs']:
    key = dict(selected=pair['selected'], prefix=pair['prefix'])
    if key not in seen:
        seen.append(key)
        unique.append(pair)

def moments(values):
    x = np.asarray(values, float)
    return dict(n=len(x), mean=float(x.mean()), mae=float(abs(x).mean()), rms=float(np.sqrt(np.mean(x*x))), max_abs=float(abs(x).max()))

rows = []
for pair in unique:
    lo, hi = pair['background_residuals']['fixed300']
    gain = pair['truth']['fixed300']['gain']
    current_line = ('7X','102X','705X').index(pair['target_line'])
    fixed_source = pair['source_vs_frozen_nominal']['endpoints']['fixed300']
    rows.append(dict(state_id=pair['state_id'], true_gain=gain,
        physical_baseline_low_high=[lo[0],hi[0]], saved_m_low_high=[lo[1],hi[1]],
        raw_level_residual_low_high=[lo[2],hi[2]], centered_residual_low_high=[lo[3],hi[3]],
        baseline_cancels=lo[0] == hi[0], saved_background_cancels=lo[1] == hi[1],
        raw_contrast_error=(lo[2]-hi[2])-gain,
        centered_contrast_error=(lo[3]-hi[3])-gain,
        source_nominal_action=pair['source_vs_frozen_nominal']['nominal_action'],
        source_assignment_position=pair['assignment']['assignment_position'],
        innovation_first_coverage_current_line=pair['coverage']['innovation']['fixed300'][0][current_line],
        innovation_action_design=pair['coverage']['innovation']['fixed300'][2],
        frozen_nominal_minus_continuing_source=fixed_source.get('frozen_minus_source')))
X=np.asarray([v['action_design']['state_vectors'][p['state_id']] for p in unique])
s=np.linalg.svd(X, compute_uv=False)
source_deltas=[row['frozen_nominal_minus_continuing_source'][3] for row in rows]
summary=dict(source_summary_bytes=SOURCE.stat().st_size, unique_prefix_n=len(unique),
    paired_background_identity=dict(all_baselines_equal=all(row['baseline_cancels'] for row in rows),
        all_saved_m_equal=all(row['saved_background_cancels'] for row in rows),
        raw_contrast_reconstruction_max_abs=max(abs(row['raw_contrast_error']) for row in rows),
        centered_contrast_reconstruction_max_abs=max(abs(row['centered_contrast_error']) for row in rows),
        precision='Derived from8-significant-digit compact result; exact shared baseline/m identity holds, tiny residual reconstruction differences reflect rounding.'),
    fixed300_signal=v['signal']['unique_prefixes'],
    full_action_design=dict(rows=len(X), columns=X.shape[1], rank=int(np.linalg.matrix_rank(X)),
        singular_values=s.tolist(), condition_2=float(s[0]/s[-1]),
        note='Label-free design from rounded issue-state covariates.9unique states/9slopes is saturated and very ill-conditioned; no outcome fit.'),
    route_only_design=dict(rows=len(X),columns=3,rank=int(np.linalg.matrix_rank(X[:,:3])),rows_per_line=3),
    action_coverage=v['action_design']['unique_prefix_source_coverage'],
    source_vs_frozen_nominal_cost=moments(source_deltas),
    rows=rows,
    confirmed=['Native low/high pairs cover all9 unique states,3 per route; all paired fixed300truths complete.',
      'The physical baseline and saved pre-issue background are common within each fixed300pair and cancel algebraically.',
      'Current innovation readout first-position705Xcoverage is1low/0high at every sampled prefix; newly collected diagnostic high outcomes never entered those fits.'],
    empirical=['On these9states, saved-background centering increases single-action residual RMS8921.12 to14033.64 while paired mean absolute gain is286.78.',
      'Paired matrix is rank9 but condition about1.06e6; rank alone cannot support9-parameter effect learning or held-future performance.'],
    next_design=dict(decision='Proceed only to a small budgeted paired-difference development comparison, not full9slopes or independent confirmation.',
      primary='Prespecified3route action slopes from paired targets; preserve analytic clock and physical baseline. Do not select features or lambda using these9targets.',
      budget='57packets×5labels=285; each admitted pair uses2slots and displaces2ordinary slots, leaving3ordinary slots in that packet. No old285plus extra paired training labels.',
      trajectory='Changed label allocation requires a new common prefix; all comparison readouts share this new flow. Keep initializer/24000horizon/trust equation/eightupdates unchanged.',
      evaluation='Use separately selected post-freeze native paired states and report stable no-harm, effect direction/regret and support; do not certify on these9development states.',
      continuation_target='Paired frozen-policy cost and source continuing-learning/calibration outcome are different estimands. Store both; do not assert equality or silently train one endpoint against the other.'),
    limitations=['Same seed6201; only9unique development states. New branch action coverage is not the old learner support.',
      'The source-vs-frozen nominal difference includes the effects of future learning and calibration, not solely parameter updates. No mediator isolated.',
      'A future three-slope model is a low-dimensional test, not a claim that route-constant gains explain shock-dependent sign changes.'])
(HERE/'background_metrics.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2)+'\n')
print(json.dumps(dict(bytes=(HERE/'background_metrics.json').stat().st_size,
    unique_prefixes=len(unique),condition=summary['full_action_design']['condition_2'],
    source_nominal_shift=summary['source_vs_frozen_nominal_cost'])))
