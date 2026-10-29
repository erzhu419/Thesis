"""Combine bounded precheck summaries, without loading a model or simulator."""
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent


def main():
    protocol = json.loads((HERE/'protocol.json').read_text())
    cases = {}
    for scenario in protocol['scenarios']:
        root = HERE/'runs'/scenario
        path = root/'summary.json'
        cases[scenario] = json.loads((path if path.exists() else root/'failure.json').read_text())
    complete = all(case.get('status') == 'complete' and case['integrity']['all_pass']
                   for case in cases.values())
    value = dict(status='complete' if complete else 'failed', priority='high', cases=cases,
        integrity=dict(all_pass=complete),
        source_contract=dict(initializer_updates=50000, feedback_slots_per_case=285,
            optimizer_updates_per_case=456, horizon_seconds=24000, F_unchanged=True, trust_unchanged=True),
        new_budget=dict(training_updates=0, feedback_slots=0, native_branches=0,
            native_seconds=0, checkpoint_downloads=0),
        decision=dict(shadow_training_started=False,
            next_gate='Read-only trace target precheck complete only if integrity passes; inspect paired reward/bootstrap/correction and CRN spread before deciding a new same-flow training version.'))
    encoded = json.dumps(value, separators=(',', ':'), allow_nan=False)+'\n'
    if len(encoded.encode()) > protocol['summary_max_bytes']:
        raise RuntimeError('Reduce summary on server before the bounded pull')
    out = HERE/'analysis'
    out.mkdir(exist_ok=False)
    with (out/'summary.json').open('x') as handle:
        handle.write(encoded)
    print(json.dumps(dict(status=value['status'], integrity=value['integrity'],
                          bytes=len(encoded.encode()))), flush=True)


if __name__ == '__main__':
    main()
