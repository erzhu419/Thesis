"""SUMO lane RNG intervention checks; full native snapshots remain on server."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import sys
import traceback
import xml.etree.ElementTree as ET

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from experiments.mcwm_closed_loop_future_rng_v102_20261028.rng import LaneRNG, build_adapter
from experiments.mcwm_closed_loop_memory_truth_v34_20261024.fork_state import _rng_state, _same_rng


def isolated_config(source, destination):
    source, destination = Path(source), Path(destination)
    tree = ET.parse(source)
    root = tree.getroot()
    for node in list(root):
        if node.tag in ('output', 'fcd_device'):
            root.remove(node)
    report = root.find('report')
    if report is not None:
        for node in list(report):
            if node.tag in ('log', 'duration-log.statistics'):
                report.remove(node)
    inputs = root.find('input')
    for node in inputs:
        if node.tag.endswith(('file', 'files')):
            node.set('value', ','.join(str((source.parent / v.strip()).resolve())
                     for v in node.get('value', '').split(',') if v.strip()))
    for section in root:
        for node in list(section):
            if node.tag.startswith('save-state.'):
                section.remove(node)
    # SUMO configuration section names do not alter option semantics. Keep
    # these manual-snapshot options outside the bridge's stripped output group.
    processing = root.find('processing')
    if processing is None:
        processing = ET.SubElement(root, 'processing')
    ET.SubElement(processing, 'save-state.rng', value='true')
    ET.SubElement(processing, 'save-state.precision', value='17')
    destination.parent.mkdir(parents=True, exist_ok=True)
    tree.write(destination, encoding='utf-8', xml_declaration=True)
    return destination


def _canonical(node):
    return (node.tag, tuple(sorted(node.attrib.items())), (node.text or '').strip(),
            tuple(_canonical(child) for child in node))


def read_snapshot(path):
    root = ET.parse(path).getroot()
    lanes, other = {}, []
    for parent in reversed(list(root.iter())):
        for node in list(parent):
            if 'rng' not in node.tag.lower():
                continue
            if node.tag.lower() == 'rnglane':
                lanes[int(node.attrib['index'])] = node.attrib['state']
            else:
                other.append(_canonical(node))
            parent.remove(node)
    if sorted(lanes) != list(range(64)):
        raise ValueError(f'Expected resident 64 lane RNG pools, observed {sorted(lanes)}')
    return dict(physical=_canonical(root), lane=lanes, other_rng=other)


def save_snapshot(api, path):
    path = Path(path).resolve()
    api.simulation.saveState(str(path))
    return read_snapshot(path)


def apply_checked(api, adapter, seed, out):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    python_before = _rng_state()
    clock = float(api.simulation.getTime())
    before = save_snapshot(api, out/'native_before.xml')
    if adapter.lane_count != len(before['lane']):
        raise ValueError('Adapter count differs from actual native pools')
    result = adapter.apply(seed)
    after = save_snapshot(api, out/'native_after.xml')
    changed = [i for i in before['lane'] if before['lane'][i] != after['lane'][i]]
    checks = dict(native_physical_serialization_unchanged=before['physical'] == after['physical'],
                  other_native_rng_unchanged=before['other_rng'] == after['other_rng'],
                  python_numpy_torch_rng_unchanged=_same_rng(python_before),
                  native_clock_unchanged=float(api.simulation.getTime()) == clock,
                  lane_pools_changed_as_declared=(len(changed) == (0 if seed is None else 64)))
    result.update(all_pass=all(checks.values()), checks=checks, clock=clock,
                  changed_lane_pools=changed,
                  before_lane_counts=[int(before['lane'][i].split()[0]) for i in range(64)],
                  after_lane_counts=[int(after['lane'][i].split()[0]) for i in range(64)])
    (out/'rng_intervention.json').write_text(json.dumps(result, indent=2)+'\n')
    if not result['all_pass']:
        raise RuntimeError(f'RNG-only intervention failed: {checks}')
    return result


def suffix_rng_usage(api, out):
    out = Path(out)
    before = read_snapshot(out/'native_after.xml')
    after = save_snapshot(api, out/'native_terminal.xml')
    counts_before = [int(before['lane'][i].split()[0]) for i in range(64)]
    counts_after = [int(after['lane'][i].split()[0]) for i in range(64)]
    draws = [b-a for a, b in zip(counts_before, counts_after)]
    return dict(all_pass=all(v >= 0 for v in draws) and any(v > 0 for v in draws),
                pools_advanced=sum(v > 0 for v in draws), draws_per_pool=draws,
                total_draws=sum(draws), native_clock=float(api.simulation.getTime()))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--sumo-cfg', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    import libsumo
    result = dict(status='failed', native_seconds=0, train_updates=0)
    started = False
    try:
        cfg = isolated_config(args.sumo_cfg, args.out/'native.sumocfg')
        adapter_path = build_adapter(args.out/'native_rng.so')
        library = Path(libsumo.__file__).with_name('_libsumo.so')
        libsumo.start(['sumo', '-c', str(cfg), '--seed', '1476154684',
                      '--no-step-log', 'true', '--duration-log.disable', 'true',
                      '--no-warnings', 'true', '--log', '/dev/null'])
        started = True
        libsumo.simulationStep(20)
        adapter = LaneRNG(adapter_path, library)
        receipts = {}
        for name, seed in [('F0', None), ('F1', 102001), ('F2', 202001)]:
            receipts[name] = apply_checked(libsumo, adapter, seed, args.out/name)
        f1 = read_snapshot(args.out/'F1/native_after.xml')['lane']
        f2 = read_snapshot(args.out/'F2/native_after.xml')['lane']
        libsumo.simulationStep(40)
        usage = suffix_rng_usage(libsumo, args.out/'F2')
        checks = dict(interventions_valid=all(r['all_pass'] for r in receipts.values()),
                      futures_differ=f1 != f2, changed_streams_are_used=usage['all_pass'],
                      fixed_budget=float(libsumo.simulation.getTime()) == 40)
        result.update(status='complete', all_pass=all(checks.values()), checks=checks,
                      version=list(libsumo.getVersion()), interventions=receipts, usage=usage)
    except BaseException:
        result['error'] = traceback.format_exc()
    finally:
        if started:
            result['native_seconds'] = float(libsumo.simulation.getTime())
            libsumo.close()
        (args.out/'summary.json').write_text(json.dumps(result, indent=2)+'\n')
        print(json.dumps(result, separators=(',', ':')), flush=True)
    return 0 if result.get('all_pass') else 1


if __name__ == '__main__':
    raise SystemExit(main())
