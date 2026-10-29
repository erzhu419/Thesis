"""The observed real case: same-tick passive ordering must not alter provenance."""
import importlib.util
import json
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
spec = importlib.util.spec_from_file_location('v77_analyze_test', HERE/'analyze.py')
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def rows():
    return [dict(record='service', bus_id=bus, line_id='7X', station_id='s', kind='departure',
        occurrence=0, _ordinal=i, observation_order=i, observed_at=12010, clock_seconds=12010,
        onboard_passenger_ids=['p'], queued_passenger_ids=[], capacity=50)
        for i, bus in enumerate(('trip1', 'trip2'))]


def dump(path, values):
    path.write_text(''.join(json.dumps({k:v for k,v in row.items() if k not in ('occurrence','_ordinal')})+'\n'
        for row in values))


def test_passive_order_can_differ_without_physical_difference(tmp_path):
    original = rows()
    replay = [dict(original[1], observation_order=0), dict(original[0], observation_order=1)]
    path = tmp_path/'service.jsonl'
    dump(path, replay)
    result = module.service_replay(original, path)
    assert result['exact'] and result['passive_writer_order_differences'] == 2


def test_actual_service_clock_or_inventory_difference_is_retained(tmp_path):
    original = rows()
    replay = [dict(original[0], clock_seconds=12011, onboard_passenger_ids=['p','q']), original[1]]
    path = tmp_path/'service.jsonl'
    dump(path, replay)
    result = module.service_replay(original, path)
    assert not result['exact']
    assert result['first_difference']['fields'] == ['clock_seconds','onboard_passenger_ids']
