"""One differentiable canonical-state execution rule for deployment and learning.

The actor samples a latent holding proposal u.  Every policy-dependent Q,
model, and trust call uses a=F(s,u); entropy stays the entropy of u.  This is
not a density correction for executed actions: F has flat reserve/floor
regions, so its pushforward can have atoms.

The formula follows v19 quarter-load-sixty using the *canonical* headways.
Canonical ratios are clipped by the existing v7 representation, hence native
raw-state v19 can differ when clipping loses a gap imbalance.  New deployment
uses this same F; the runner records native-v19 disagreements separately.
"""

from dataclasses import asdict

import numpy as np
import torch


class ExecutionAction:
    def __init__(self, contract):
        self.contract = asdict(contract) if not isinstance(contract, dict) else dict(contract)
        self.max_hold = float(self.contract["max_hold_seconds"])
        self.headway_scale = float(self.contract["scheduled_headway_scale_seconds"])

    def __call__(self, states, proposal):
        """Transform an (N, 1) proposal; preserve autograd through active pieces."""
        current = torch.round(states[:, 2] * 15).long().clamp(0, 15)
        slots = states[:, 5:].reshape(-1, 16, 14)
        event = slots[torch.arange(len(states), device=states.device), current]
        # event: active,line,station,fwd_ratio,bwd_ratio,target_fwd,target_bwd,
        # queue_ratio,load_ratio,demand_ratio,dwell,fwd_present,bwd_present,prior.
        forward = event[:, 11] > .5
        backward = event[:, 12] > .5
        forward_error = (event[:, 3]-1.) * event[:, 5] * self.headway_scale
        backward_error = (event[:, 4]-1.) * event[:, 6] * self.headway_scale
        reserve = (.5 * (backward_error * backward - forward_error * forward)).clamp(0., self.max_hold)
        parent = self.hold_seconds(proposal).reshape(-1)
        base = .125 * parent
        hold = torch.where((states[:, 0] < .75) & (parent > 0.),
                           torch.minimum(parent, torch.maximum(base, reserve)), base)
        # v19's 60 s completion floor covers v17's 20 s tail floor entirely.
        completion = ((states[:, 0] >= .15) & (states[:, 0] < .75)
                      & forward & ~backward & (event[:, 9] > 0.) & (event[:, 8] <= .25))
        hold = torch.where(completion, torch.maximum(hold, hold.new_tensor(60.)), hold)
        hold = hold.clamp(0., self.max_hold)
        return (2. * hold / self.max_hold - 1.).reshape(-1, 1)

    def hold_seconds(self, actions):
        return .5 * (actions.clamp(-1., 1.) + 1.) * self.max_hold

    @torch.no_grad()
    def numpy(self, states, proposal):
        vector = np.asarray(states).ndim == 1
        states = np.asarray(states, dtype=np.float32).reshape(-1, 229)
        proposal = np.asarray(proposal, dtype=np.float32).reshape(-1, 1)
        result = self(torch.as_tensor(states), torch.as_tensor(proposal)).numpy()
        return result[0] if vector else result
