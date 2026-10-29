"""A single resident, zero-optimizer-update probe of the frozen MC-WM endpoint."""
import argparse
import json
from pathlib import Path
import sys
import time

HERE = Path(__file__).resolve().parent


def run(root, out):
    out.mkdir(parents=True, exist_ok=False)
    p = json.loads((HERE/'protocol.json').read_text())
    original = json.loads((root/p['reference_protocol']).read_text())
    sys.path.insert(0, str(root))
    sys.path.insert(0, str(root/'experiments/mcwm_full_mcwm_dripping_pilot_v107_20261029'))
    from learning import load_learner
    started = time.monotonic()
    result = dict(status='incomplete', budget=dict(new_native_seconds=0,
        new_optimizer_updates=0, new_feedback_labels=0, model_refits=0),
        scope=p['scope'])
    try:
        print('Loading the resident 70000 endpoint; no optimizer or environment calls', flush=True)
        learner, adapter, contract, _, initialization = load_learner(root, out/'restore', 'fresh', original)
        from reconstruct import first_packet
        from kernel import probe
        folder = root/p['resident_prefix']
        trace = [json.loads(line) for line in (folder/'prefix_trace.jsonl').read_text().splitlines()]
        receipt = json.loads((folder/'packet_receipts.json').read_text())[0]
        packet = first_packet(trace, receipt, original, contract)
        selected_ids = {row.sequence for row in packet}
        for row in trace:
            row['feedback_selected'] = row['sequence'] in selected_ids
        saved = adapter._original_sample_mixed
        captured = {}

        def capture(*args, **kwargs):
            offline, source = saved(*args, **kwargs)
            captured['offline'] = {key: value.clone() for key, value in offline.items()}
            return offline, source

        adapter._original_sample_mixed = capture
        adapter.add_target(packet)
        fresh, source = learner.replay_buffer.sample_mixed(256, .5, rng=learner.rng)
        print('Recovered actual first-packet rows; computing target and contrast derivatives', flush=True)
        result['probe'] = probe(learner, captured['offline'], source, fresh, trace, contract)
        if result['probe']['status'] != 'complete':
            raise RuntimeError(result['probe']['reason'])
        result['first_packet'] = dict(receipt=receipt, exact_selection=True,
            rows=[dict(sequence=row.sequence, dt=row.dt, raw_reward=row.r,
                available_at=row.available_at) for row in packet])
        result['replay'] = adapter.summary()
        result['initialization'] = {key: initialization[key] for key in (
            'actor_exact', 'total_updates', 'online_critic_updates', 'runtime')}
        result['terminal_counters'] = dict(total_updates=learner.total_updates,
            online_critic_updates=learner.online_critic_updates)
        result['status'] = 'complete'
    except Exception as error:
        result['status'] = 'HOLD_DIAGNOSTIC_FAILURE'
        result['error'] = dict(type=type(error).__name__, message=str(error))
        raise
    finally:
        result['wall_seconds'] = time.monotonic()-started
        raw = json.dumps(result, ensure_ascii=False, separators=(',', ':'), allow_nan=False)
        if len(raw.encode()) > p['summary_limit_bytes']:
            raise ValueError('Probe summary exceeds the declared small-transfer limit')
        (out/'summary.json').write_text(raw+'\n')
        print(json.dumps(dict(status=result['status'], budget=result['budget'])), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--thesis-root', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    run(args.thesis_root.resolve(), args.out.resolve())
