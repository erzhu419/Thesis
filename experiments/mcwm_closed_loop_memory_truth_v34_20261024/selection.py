"""Predeclared first eligible issues; no prediction or outcome enters selection."""
from collections import Counter

import numpy as np

from experiments.mcwm_closed_loop_action_fork_v7_20261020.fork_stream import ForkEventStream


class MemoryForkStream(ForkEventStream):
    def __init__(self, *args, selector, **kwargs):
        self.memory = None
        self._declared_selector = selector
        super().__init__(*args, selector=self._snapshot_and_select, **kwargs)

    def _snapshot_and_select(self, event):
        if self.memory is None:
            raise RuntimeError('Install execution memory before the first issue')
        event['memory_s'] = self.memory.snapshot(event['started_at'])
        return self._declared_selector(event)


class FirstEligible:
    def __init__(self, protocol):
        self.protocol = protocol
        self.slots = {line+'/'+window: dict(line_id=line, window=window, bounds=bounds,
            selected=None, attempted=0, reasons=Counter())
            for line in protocol['focus_lines'] for window,bounds in protocol['windows'].items()}

    def active(self, event):
        for key, slot in self.slots.items():
            if slot['selected'] is None and event['line_id']==slot['line_id'] and \
                    slot['bounds'][0] <= event['started_at'] < slot['bounds'][1]:
                return key
        return None

    def consider(self, key, event, holds, *, at_stop, simulation_step, action_step):
        slot = self.slots[key]
        slot['attempted'] += 1
        if at_stop is not True or simulation_step is None or action_step is None:
            slot['reasons']['native_stop_or_clock_unavailable'] += 1
            return False
        if holds[1]-holds[0] < self.protocol['minimum_gap_steps']*max(simulation_step,action_step):
            slot['reasons']['F_alias_or_insufficient_hold_gap'] += 1
            return False
        slot['selected'] = dict(sequence=event['sequence'], issue=event['started_at'],
            action_holds=list(holds), simulation_step=simulation_step, action_step=action_step)
        return True

    def summary(self):
        return {key:dict(**{k:v for k,v in slot.items() if k!='reasons'},
                         reasons=dict(slot['reasons']), status='selected' if slot['selected'] else 'missing')
                for key,slot in self.slots.items()}
