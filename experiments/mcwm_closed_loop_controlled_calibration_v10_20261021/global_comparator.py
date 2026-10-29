"""Retain the 58-column global residual on the new common selected facts."""
from types import SimpleNamespace

import torch

from experiments.mcwm_closed_loop_action_response_v8_20261021.response_world import ResponseWorld
from experiments.mcwm_closed_loop_action_response_v8_20261021.action_response import (
    FEATURE_NAMES, response_features, fit_response_readout, decode_event_step,
)
from experiments.mcwm_closed_loop_response_ablation_v9_20261021.ablation_world import ARM_FEATURE_INDICES
from experiments.mcwm_closed_loop_v1.learning import targets, row_table, ModelAugmentation


class Global58(ResponseWorld):
    """No same-vehicle or 300-second outcome enters this global MDP model."""
    columns = ARM_FEATURE_INDICES['memory_level']

    @torch.no_grad()
    def refit(self):
        if not self.records:
            return
        records = list(self.records)
        y = targets(row_table([r['row'] for r in records]), self.world.stations, self.world.device)
        x = torch.cat([response_features(y['s'][i:i+1], y['a'][i:i+1],
            r['memory_features'], r['nonholding_seconds'], self.numeric_lines)
            for i, r in enumerate(records)])
        target = (y['g']-self.world.norm.g_mean)/self.world.norm.g_std
        for family, model in self.world.models.items():
            weight = x.new_zeros(len(FEATURE_NAMES), 4)
            weight[list(self.columns)] = fit_response_readout(x[:, list(self.columns)],
                target-model(y['s'], y['a'])[:, :4], ridge=self.ridge)
            self.readouts[family] = weight
        self.refits += 1

    @torch.no_grad()
    def predict_issue(self, issued, context, *, seed):
        world = self.world
        state = torch.as_tensor(issued['s'][None], device=world.device)
        family = world.family
        result = dict(family=family, trained_global_labels=self.labels, arms={})
        for arm, weight in (('base_analytic', None), ('global58', self.readouts[family])):
            candidates = {}
            for name, hold in (('low', 0.), ('high', 7.5)):
                action = state.new_tensor([[hold/30.-1.]])
                output = world.models[family](state, action).detach().clone()
                if weight is not None:
                    output[:, :4] += response_features(state, action, context['memory_features'],
                        context['nonholding_seconds'], self.numeric_lines) @ weight
                decoder = SimpleNamespace(norm=world.norm, stations=world.stations, device=world.device,
                    generator=torch.Generator(device=world.device).manual_seed(int(seed)))
                ns, reward, duration = decode_event_step(state, action, output,
                    lambda s, v: ModelAugmentation.decode(decoder, s, v))
                raw = output[0, :4]*world.norm.g_std+world.norm.g_mean
                raw = raw.clone()
                raw[:2] = raw[:2].expm1()
                candidates[name] = dict(dt=float(duration.item()), cost=-float(reward.item()),
                    raw_dt=float(raw[0]), raw_cost=float(raw[1]),
                    zero_duration_analytic=bool(duration.item() == 0))
            result['arms'][arm] = dict(**candidates,
                cost_gain_low_minus_high=candidates['low']['cost']-candidates['high']['cost'],
                raw_cost_gain_low_minus_high=candidates['low']['raw_cost']-candidates['high']['raw_cost'])
        return result

    def summary(self, now):
        result = super().summary(now)
        result.update(columns=58, feature_names=[FEATURE_NAMES[i] for i in self.columns],
            physical_target_labels=0, context_physical='same fixed resident source calibration; no extra target intervals')
        return result
