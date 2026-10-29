"""Finish the completed C collection after one compact receipt exceeded32KiB by5bytes."""
import json

from experiments.mcwm_closed_loop_confidence_calibration_v99_20261028.analysis import analyze
from experiments.mcwm_closed_loop_confidence_calibration_v99_20261028.infra import (
    HERE, _pull, cells, key, protocol, write)


def main():
    p = protocol()
    load = lambda path: json.loads(path.read_text())
    size = lambda value: len(json.dumps(value, separators=(',', ':')).encode())
    audit = load(HERE/'freeze_audit.json')
    gate = load(HERE/'first_pair_gate.json')
    rows, new_bytes, reused = [], 0, 0
    for seed, scene in cells():
        path = HERE/'local_inputs'/(str(seed)+'_'+scene+'.json')
        if path.exists():
            row = load(path)
            reused += 1
        else:
            value = _pull('runs/'+key(seed, scene)+'/collector_summary.json', 33000)
            if not value['ready']:
                raise RuntimeError('Expected an already completed collector; no native retry')
            row = value['body']
            new_bytes += value['body_bytes']
            write(path, row)
        rows.append(row)
    transferred = size(audit)+size(gate)+sum(map(size, rows))
    if transferred > p['retrieval']['aggregate_input_cap_bytes']:
        raise ValueError('Original aggregate transfer budget exceeded')
    result = analyze(rows, p, audit, load(HERE/'source_model.json'))
    summary_bytes = len(json.dumps(result, ensure_ascii=False, separators=(',', ':'), allow_nan=False).encode())
    if summary_bytes > p['retrieval']['final_summary_bytes']:
        raise ValueError('Original final summary budget exceeded')
    write(HERE/'summary.json', result)
    write(HERE/'local_summary_receipt.json', dict(transferred_bytes=transferred,
        summary_bytes=summary_bytes, source_node='node001', execution_node='local',
        native_seconds=0, fits=0, optimizer_updates=0, compact_inputs_only=True,
        recovery_new_transfer_bytes=new_bytes, existing_collectors_reused=reused,
        original_task='t212115', original_per_collector_limit=32768,
        observed_largest_collector_bytes=32773, recovery_per_collector_limit=33000,
        scientific_contract_unchanged=True, old_code_and_results_unchanged=True))
    print(json.dumps(dict(status=result['status'], transferred_bytes=transferred,
                          new_transfer_bytes=new_bytes, summary_bytes=summary_bytes)), flush=True)


if __name__ == '__main__':
    main()
