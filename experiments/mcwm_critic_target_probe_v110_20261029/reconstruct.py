"""Recover the first feedback packet from consecutive saved control issues."""
from types import SimpleNamespace

import numpy as np


def first_packet(trace, receipt, protocol, contract):
    from bus_h2o.mdp_contract_v7 import transition_reward_v7
    from experiments.mcwm_full_mcwm_dripping_pilot_v107_20261029.control import PacketBudget

    transitions = []
    for before, after in zip(trace, trace[1:]):
        if after['sequence'] != before['sequence']+1:
            raise ValueError('Missing next chronological issue; cannot reconstruct feedback')
        start, end = before['issue_ledger'], after['issue_ledger']
        dt = end['time_seconds']-start['time_seconds']
        if dt < 0:
            raise ValueError('Chronological ledger time decreased')
        transitions.append(SimpleNamespace(
            s=np.asarray(before['s'], dtype=np.float32),
            a=np.asarray(before['a'], dtype=np.float32),
            ns=np.asarray(after['s'], dtype=np.float32),
            r=transition_reward_v7(start, end, contract), dt=dt, terminal=False,
            available_at=end['time_seconds']+protocol['feedback_delay_seconds'],
            sequence=before['sequence']))
    budget = PacketBudget(protocol)
    budget.ingest(transitions)
    selected = budget.take(receipt['processed_at'])
    if budget.receipts[0] != receipt:
        raise ValueError('Reconstructed selection differs from the actual first receipt')
    return selected
