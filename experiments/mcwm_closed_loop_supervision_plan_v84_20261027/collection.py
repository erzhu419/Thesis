"""Pre-issue action rule for a future reference-anchored collection campaign."""
import numpy as np


def choose_pair(transform, state, reference_proposal, *, simulation_step, action_step):
    """Use the deployed reference action and the farthest legal F endpoint.

    Ties choose the smaller hold. No prediction, outcome or future state is
    accepted by this interface. Returned actions are already executed-space.
    """
    reference = transform.numpy(state, reference_proposal)
    endpoints = [transform.numpy(state, np.asarray([u], np.float32)) for u in (-1., 1.)]

    def hold(action):
        return .5*(float(action[0])+1)*transform.max_hold

    ref_hold = hold(reference)
    candidate = max(endpoints, key=lambda action: (abs(hold(action)-ref_hold), -hold(action)))
    gap = abs(hold(candidate)-ref_hold)
    return dict(reference_action=reference.tolist(), candidate_action=candidate.tolist(),
        reference_hold_seconds=ref_hold, candidate_hold_seconds=hold(candidate),
        gap_seconds=gap, supported=gap >= 2*max(simulation_step, action_step),
        target='remaining_cost(candidate)-remaining_cost(reference)',
        single_root_intervention=True, factual_action_remapped=False)
