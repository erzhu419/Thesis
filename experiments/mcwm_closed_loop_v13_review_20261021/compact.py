"""Reaggregate completed v13 results; share action-check column names once."""
import json
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
SOURCE = HERE.parent / 'mcwm_closed_loop_separated_contrast_v13_20261021'
sys.path.insert(0, str(SOURCE))
import analyze


def reduce_result(value):
    # Preserve every scalar; repeated column names caused the original 202829-byte output.
    first = next(iter(value['cells'][0]['action_difference'].values()))
    checks = list(first['checks'])
    fields = [key for key in first if key != 'checks']
    value['schema']['cell_action_difference_check_order'] = checks
    value['schema']['cell_action_difference_value_order'] = fields
    for cell in value['cells']:
        for name, row in cell['action_difference'].items():
            assert list(row['checks']) == checks
            assert [key for key in row if key != 'checks'] == fields
            cell['action_difference'][name] = [[row['checks'][key] for key in checks],
                                             [row[key] for key in fields]]
    return value


if __name__ == '__main__':
    value = reduce_result(analyze.analyze(SOURCE))
    payload = json.dumps(value, ensure_ascii=False, separators=(',', ':'), allow_nan=False) + '\n'
    assert len(payload.encode()) <= 200000
    (HERE / 'summary.json').write_text(payload)
    print(json.dumps(dict(status=value['status'], bytes=len(payload.encode()), integrity=value['integrity'])))
