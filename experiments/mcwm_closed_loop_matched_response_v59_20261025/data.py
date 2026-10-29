"""Load the new paid feedback, keeping no-fresh files outcome-free."""
from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np

COVARIATES = {'s', 'a', 'memory_s', 'sequence', 'line_id', 'started_at'}


def load_dataset(path, *, arm, scenario, seed):
    payload = json.loads(Path(path).read_text())
    if not payload['proof']['all_pass']:
        raise ValueError('New native collection failed integrity')
    meta = payload['metadata']
    if meta['scenario'] != scenario or meta['initialization_seed'] != seed:
        raise ValueError('Feedback belongs to another scenario or initializer')
    packets, ids, previous = [], [], -float('inf')
    for index, packet in enumerate(payload['packets'], 1):
        receipt = packet['receipt']
        now = float(receipt['processed_at'])
        if packet['index'] != index or now < previous or len(packet['rows']) != 5:
            raise ValueError('Expected chronological five-slot packets')
        rows = []
        for value in packet['rows']:
            if arm == 'no_fresh' and set(value) != COVARIATES:
                raise ValueError('No-fresh input must contain only current covariates')
            values = dict(value)
            for field, shape in (('s', (229,)), ('a', (1,))):
                values[field] = np.asarray(values[field], np.float32).reshape(shape)
            if values['memory_s']['observed_at'] != values['started_at']:
                raise ValueError('Current memory belongs to a different issue')
            if arm == 'fresh':
                values['ns'] = np.asarray(values['ns'], np.float32).reshape(229)
                if (values['available_at'] > now or values['completed_at'] > now
                        or values['memory_ns']['observed_at'] != values['completed_at']):
                    raise ValueError('Outcome unavailable or its next memory is not causal')
                if values['dt'] < 0 or values['completed_at']-values['started_at'] != values['dt']:
                    raise ValueError('Physical next-event duration is inconsistent')
            rows.append(SimpleNamespace(**values))
            ids.append(int(values['sequence']))
        if set(receipt['sequence_ids']) != {row.sequence for row in rows}:
            raise ValueError('Packet receipt and row identities differ')
        packets.append(dict(index=index, receipt=receipt, rows=rows))
        previous = now
    paired = sorted(value for value in ids if value < 0)
    if len(packets) != 57 or len(ids) != 285 or len(set(ids)) != 285 or paired != list(range(-18, 0)):
        raise ValueError('Require267 ordinary+18 paired unique slots,57 packets')
    for index in range(9):
        wanted = {-(2*index+1), -(2*index+2)}
        if sum(wanted <= {row.sequence for row in packet['rows']} for packet in packets) != 1:
            raise ValueError('Each paid pair must be admitted atomically in one packet')
    return dict(metadata=meta, proof=payload['proof'], packets=packets)
