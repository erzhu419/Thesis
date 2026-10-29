"""Frozen, event-indexed candidate traces; no learner or native environment.

The recurrence preserves each supplied native float32 one-step target Y_i:
    G_i = Y_i + .9 * lambda_trace * c_(i+1) * (G_(i+1) - Qbar_(i+1)).
Qbar must be scored on the next recorded executed action using exactly the
target-head subset and min statistic used by Y_i.  The caller proves that
contract and supplies F-pushforward execution support.  This is a diagnostic
candidate for the existing mixed backup, not a convergence claim for Retrace.

Inputs are real closed transitions, including dt=0 events.  A fixed-clock
partial ledger is not a transition and must never be passed as an endpoint.
"""

from __future__ import annotations

import math

import numpy as np


def construct_trace(native_targets, recorded_target_q_for_link, c, eligibility,
                    lambda_trace, *, real_endpoints=None, dt=None):
    """Construct one trace for one predeclared eligibility/support scope.

    ``eligibility[i]`` authorizes use of reward/feedback row i.  Run separately
    with all real diagnostic rows and with the original 285-slot membership;
    the first scope measures support and never authorizes training.  Link i
    links row i to i+1, so Qbar and c have n-1 entries.  Missing Qbar/support is
    represented by None and truncates the link.  No baseline is substituted.

    ``reach_weights[j]`` is the coefficient of reward j in the candidate
    target at row 0.  Its index counts actual events, not elapsed seconds.
    Thus a zero-duration event retains its .9 continuation factor.  Rows not
    eligible for this scope return no target instead of using a diagnostic
    label as an authorized training label.
    """
    native = np.asarray(native_targets, dtype=np.float32)
    if native.ndim != 1 or not np.isfinite(native).all():
        raise ValueError('native_targets must be a finite float32 event vector')
    count = len(native)
    if len(eligibility) != count:
        raise ValueError('eligibility must have one value per real transition')
    if len(recorded_target_q_for_link) != max(count-1, 0) or len(c) != max(count-1, 0):
        raise ValueError('Qbar and c must have one value per successor link')
    lam = float(lambda_trace)
    if not math.isfinite(lam) or not 0. <= lam <= 1.:
        raise ValueError('lambda_trace must be in [0, 1]')
    actual = [True]*count if real_endpoints is None else list(real_endpoints)
    if len(actual) != count:
        raise ValueError('real_endpoints must have one value per transition')
    if dt is not None and (len(dt) != count or any(not math.isfinite(float(v)) or float(v) < 0. for v in dt)):
        raise ValueError('dt must contain the actual nonnegative event durations')

    allowed = [bool(eligibility[i]) and bool(actual[i]) for i in range(count)]
    links = []
    for i in range(max(count-1, 0)):
        reason = None
        if not actual[i] or not actual[i+1]:
            reason = 'not_a_real_closed_event'
        elif not eligibility[i]:
            reason = 'current_feedback_not_eligible'
        elif not eligibility[i+1]:
            reason = 'successor_feedback_not_eligible'
        elif c[i] is None:
            reason = 'executed_action_support_unavailable'
        elif not math.isfinite(float(c[i])) or not 0. <= float(c[i]) <= 1.:
            raise ValueError('supplied truncated execution ratio must be in [0, 1]')
        elif float(c[i]) == 0.:
            reason = 'zero_execution_support'
        elif lam == 0.:
            # In particular, do not inspect or multiply an absent/NaN Qbar.
            reason = 'lambda_zero'
        elif recorded_target_q_for_link[i] is None or not math.isfinite(float(recorded_target_q_for_link[i])):
            reason = 'matching_subset_target_q_unavailable'
        weight = (np.float32(np.float32(.9)*np.float32(lam)*np.float32(c[i]))
                  if reason is None else np.float32(0.))
        links.append(dict(from_event=i, to_event=i+1, active=reason is None,
                          reason=reason, coefficient=float(weight)))

    # Copies, rather than recomputed reward+bootstrap, preserve original native
    # addition rounding and signed zero exactly when lambda=0 or a link stops.
    values = [np.float32(native[i]) if allowed[i] else None for i in range(count)]
    for i in range(count-2, -1, -1):
        if links[i]['active']:
            correction = np.float32(np.float32(links[i]['coefficient'])*
                                   np.float32(values[i+1]-np.float32(recorded_target_q_for_link[i])))
            values[i] = np.float32(native[i]+correction)

    reach = [0.]*count
    if count and allowed[0]:
        reach[0] = 1.
        for i, link in enumerate(links):
            reach[i+1] = reach[i]*link['coefficient'] if link['active'] else 0.
    first_stop = next((dict(event=i+1, reason=link['reason'])
                       for i, link in enumerate(links) if not link['active']), None)
    if count and not allowed[0]:
        first_stop = dict(event=0, reason=('not_a_real_closed_event' if not actual[0]
                                          else 'current_feedback_not_eligible'))
    return dict(targets=[None if value is None else float(value) for value in values],
                native_targets=native.tolist(), lambda_trace=lam, links=links,
                reach_weights=reach, reached_event_indices=[i for i, weight in enumerate(reach) if weight != 0.],
                first_stop=first_stop, eligible_events=sum(allowed),
                zero_duration_events=None if dt is None else sum(float(value) == 0. for value in dt),
                event_count=count, terminal_rule='retain_last_real_native_one_step_target',
                synthetic300_state=False, new_training_updates=0)
