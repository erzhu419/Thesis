"""Switch execution only after the complete native fork event batch."""
import numpy as np
import torch

from experiments.mcwm_closed_loop_response_residual_v11_20261021.paired_stream import PairedControlledStream


@torch.no_grad()
def tail_action(transform, proposal):
    """Use the native tail formula without changing any actor/state clock."""
    value = torch.as_tensor(np.asarray(proposal, np.float32).reshape(-1, 1))
    hold = .125*transform.hold_seconds(value)
    return (2.*hold/transform.max_hold-1.).numpy().reshape(-1)


class PhasePolicy:
    """Run the unchanged actor on the true state, then select its execution rule."""
    def __init__(self, learner, transform, stream, horizon, evidence):
        self.learner, self.transform, self.stream = learner, transform, stream
        self.horizon, self.evidence, self.last_proposal = float(horizon), evidence, None

    def __call__(self, state):
        clock = np.float32(self.stream.now/self.horizon)
        self.evidence['actor_real_clock'] &= bool(state[0] == clock)
        self.evidence['actor_clock_max_abs_error'] = max(self.evidence['actor_clock_max_abs_error'],
            abs(float(state[0])*self.horizon-self.stream.now))
        self.last_proposal = self.learner.act(state[None],deterministic=True)[0]
        if self.stream.tail_policy_active:
            self.evidence['tail_policy_issues'] += 1
            return tail_action(self.transform,self.last_proposal)
        self.evidence['native_policy_issues'] += 1
        return self.transform.numpy(state,self.last_proposal)


class PhaseControlledStream(PairedControlledStream):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.tail_policy_active = False
        self.tail_batches = 0

    def step(self, policy):
        # Freeze this choice for the entire simultaneous event batch. A fork
        # created halfway through it must not change siblings' normal actions.
        self.tail_policy_active = self.fork is not None
        self.tail_batches += int(self.tail_policy_active)
        return super().step(policy)
