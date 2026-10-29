"""Packet-entry versus packet-end actor on the same five paid visible states."""
from collections import defaultdict

import numpy as np
import torch


class PacketPolicyProbe:
    def __init__(self):
        self.packets = 0
        self.states = 0
        self.groups = {}

    def record(self, learner, packet, *, processed_at):
        states = torch.as_tensor(np.stack([row.s for row in packet]),
                                 dtype=torch.float32, device=learner.device)
        values = {key: value.detach().cpu().numpy()
                  for key, value in learner.actor_comparison(states).items()}
        window = ('pre_early' if processed_at < 9000 else
                  'early' if processed_at < 12000 else 'late_training')
        grouped = defaultdict(list)
        for index, row in enumerate(packet):
            grouped[str(row.line_id)].append(index)
        rows = []
        for line, indices in sorted(grouped.items()):
            delta = values['holding_delta_seconds'][indices]
            row = dict(line=line, window=window, count=len(indices),
                gate_crossings=int(values['gate_flip'][indices].sum()),
                gate_kl_sum=float(values['gate_kl'][indices].sum()),
                weighted_positive_kl_sum=float(values['weighted_positive_kl'][indices].sum()),
                total_kl_sum=float(values['total_kl'][indices].sum()),
                holding_delta_sum_seconds=float(delta.sum()),
                holding_abs_delta_sum_seconds=float(np.abs(delta).sum()),
                holding_abs_delta_max_seconds=float(np.abs(delta).max()),
                probability_abs_delta_sum=float(np.abs(values['hold_probability_current'][indices]
                    - values['hold_probability_reference'][indices]).sum()),
                positive_mean_abs_delta_sum=float(np.abs(values['positive_mean_current'][indices]
                    - values['positive_mean_reference'][indices]).sum()),
                positive_log_std_abs_delta_sum=float(np.abs(values['positive_log_std_current'][indices]
                    - values['positive_log_std_reference'][indices]).sum()))
            rows.append(row)
            key = line, window
            if key not in self.groups:
                self.groups[key] = dict(row)
            else:
                total = self.groups[key]
                for field, value in row.items():
                    if field in ('line', 'window'):
                        continue
                    total[field] = (max(total[field], value) if field.endswith('_max_seconds')
                                    else total[field]+value)
        self.packets += 1
        self.states += len(packet)
        return dict(packet=self.packets, processed_at=float(processed_at), states=len(packet),
                    groups=rows, actor=learner.packet_actor_audit())

    def summary(self):
        return dict(same_states=True, same_F=True, additional_random_draws=0,
            packets=self.packets, states=self.states,
            groups=[self.groups[key] for key in sorted(self.groups)])


def native_preflight(learner, replay, contract, initializer, packet, receipt,
                     clock_audit, action_audit, *, seed, scenario, arm, runtime_contract):
    """Receipt from the first real packet; these same full native days continue."""
    actor = learner.actor_constraint_audit()
    targets = learner.target_audit()
    fresh = arm == 'fresh'
    checks = dict(first_packet_five_slots=replay.target_received == len(packet) == 5,
        eight_actor_updates=actor['updates'] == actor['actor_updates'] == actor['alpha_updates'] == 8,
        one_fixed_reference=actor['packets'] == actor['reference_refreshes'] == 1
            and actor['packet_step_counts'] == [8] and actor['packet_reference_fixed'],
        fixed_full_kl=actor['coefficient'] == 1.0 and actor['full_hurdle_kl_scope']
            and actor['kl_orientation'] == 'new||packet_reference',
        no_extra_sampling=actor['additional_random_draws'] == 0,
        original_target_draws=targets['parent_mixed_target_calls'] == 8
            and targets['parent_mixed_target_rows'] == 8*256,
        original_fresh_target=targets['fresh_value_replacement_calls'] == (8 if fresh else 0)
            and targets['fresh_value_replacement_rows'] == (8*64 if fresh else 0)
            and targets['no_fresh_unchanged_returns'] == (0 if fresh else 8),
        no_fresh_outcome_isolation=fresh or all(set(row) == {'s', 'a'} for row in replay.recent),
        real_actor_clock=clock_audit['actor_real_clock'],
        shared_F=action_audit['shared_rule_exact'])
    return dict(technical_gate_pass=all(checks.values()), checks=checks, seed=seed,
        scenario=scenario, arm=arm,
        initializer=dict(path=str(initializer), initial_updates=learner.total_updates-8,
            history_rows=len(replay.history['s']), source_rows=len(replay.source['s'])),
        first_feedback=dict(processed_at=receipt['processed_at'],
                            sequence_ids=receipt['sequence_ids']),
        execution_contract=runtime_contract, feedback_slots=replay.target_received,
        policy_updates=8, actor_constraint=actor, target_audit=targets,
        reused_full_native_day=True)
