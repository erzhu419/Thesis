"""Same-draw local model counterfactuals without changing the live update."""
import torch
from experiments.mcwm_closed_loop_action_local_v3_20261020.learning import ActionConsistentTrainer
from experiments.mcwm_closed_loop_action_local_v3_20261020.local_world import LocalWorld
from experiments.mcwm_closed_loop_v1.learning import ModelAugmentation
from experiments.mcwm_closed_loop_diagnostic_v2_20261020.td_diagnostics import TransitH2OTrainerV7


class ProbedWorld(LocalWorld):
    capture_counterfactual = False
    counterfactual_batch = None

    @torch.no_grad()
    def generate(self, actor, size):
        self.counterfactual_batch = None
        before = self.generator.get_state() if self.capture_counterfactual else None
        result = super().generate(actor, size)
        if result is None or not self.capture_counterfactual:
            return result
        batch, _ = result
        after = self.generator.get_state()
        try:
            self.generator.set_state(before)
            ns, reward, dt = ModelAugmentation.decode(self, batch['observations'],
                self.predict(batch['observations'], batch['actions'], calibrated=False))
        finally:
            self.generator.set_state(after)
        self.counterfactual_batch = dict(batch, next_observations=ns, rewards=reward,
                                         delta_t_seconds=dt)
        return result


class ProbedTrainer(ActionConsistentTrainer):
    local_counterfactual = None

    def _target(self, batch, *, backup):
        probe = (self._collect_td and self._making_model and self.world is not None
                 and self.world.counterfactual_batch is not None)
        before = torch.get_rng_state() if probe else None
        actual = super()._target(batch, backup=backup)
        if not probe:
            return actual
        base = self.world.counterfactual_batch
        # Fork preserves the random state AFTER the actual target above.
        with torch.random.fork_rng(devices=[]), torch.no_grad():
            torch.set_rng_state(before)
            base_target = TransitH2OTrainerV7._target(self, base, backup=backup)
            q = self.critic(batch['observations'], batch['actions'])
            actual_loss = self._elementwise_td_loss(q, actual)
            base_loss = self._elementwise_td_loss(q, base_target)
            actual_u = self.actor(batch['next_observations'], deterministic=True)[0]
            base_u = self.actor(base['next_observations'], deterministic=True)[0]
            actual_a = self.action_transform(batch['next_observations'], actual_u)
            base_a = self.action_transform(base['next_observations'], base_u)
            h = self.action_transform.hold_seconds
            dt_delta = batch['delta_t_seconds']-base['delta_t_seconds']
            cost_delta = base['rewards']-batch['rewards']
            self.local_counterfactual = dict(rows=len(actual),
                applied_rows=int(((dt_delta.abs()+cost_delta.abs()) > 1e-6).sum()),
                mean_duration_delta_seconds=float(dt_delta.mean()),
                mean_absolute_duration_delta_seconds=float(dt_delta.abs().mean()),
                max_absolute_duration_delta_seconds=float(dt_delta.abs().max()),
                mean_cost_delta_passenger_seconds=float(cost_delta.mean()),
                mean_absolute_cost_delta_passenger_seconds=float(cost_delta.abs().mean()),
                mean_absolute_target_q_delta=float((actual-base_target).abs().mean()),
                mean_absolute_td_loss_delta=float((actual_loss-base_loss).abs().mean()),
                next_proposed_hold_change_seconds=float((h(actual_u)-h(base_u)).abs().mean()),
                next_executed_hold_change_seconds=float((h(actual_a)-h(base_a)).abs().mean()),
                interpretation='Same generated states, executed actions, current critic and random draws; only calibrated global model outputs differ. Next actions respond to predicted next-state changes, not a shadow optimizer update.')
        return actual
