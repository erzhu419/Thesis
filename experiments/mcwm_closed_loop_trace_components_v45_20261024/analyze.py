"""Combine only the bounded resident summaries; do not load weights."""
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent


def main():
    protocol = json.loads((HERE/'protocol.json').read_text())
    cases = {}
    for scenario in protocol['scenarios']:
        root = HERE/'runs'/scenario
        source = root/'summary.json'
        cases[scenario] = json.loads((source if source.exists() else root/'failure.json').read_text())
    complete = all(case.get('status') == 'complete' and case['integrity']['all_pass'] for case in cases.values())
    value = dict(status='complete' if complete else 'failed', priority='high', cases=cases,
        integrity=dict(all_pass=complete),
        preserved_contract=dict(initializer_updates=50000, feedback_slots_per_case=285,
            optimizer_updates_per_case=456, horizon_seconds=24000, F_unchanged=True, trust_unchanged=True),
        new_budget=dict(training_updates=0, feedback_slots=0, native_branches=0,
            native_seconds=0, checkpoint_downloads=0),
        decision=dict(training_started=False, target_change_installed=False,
            next_gate='Review705X component and actor/head-order attribution, then name one target-change candidate. No automatic new stage.'))
    encoded = json.dumps(value, separators=(',', ':'), allow_nan=False)+'\n'
    if len(encoded.encode()) > protocol['summary_max_bytes']:
        raise RuntimeError('Reduce resident component summary before pulling it')
    out = HERE/'analysis'
    out.mkdir(exist_ok=False)
    with (out/'summary.json').open('x') as handle:
        handle.write(encoded)
    print(json.dumps(dict(status=value['status'], integrity=value['integrity'], bytes=len(encoded.encode()))), flush=True)


if __name__ == '__main__':
    main()
