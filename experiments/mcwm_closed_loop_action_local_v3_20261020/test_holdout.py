"""Missing post-update endpoints must stay missing, not become a zero effect."""
import importlib.util
from pathlib import Path
import sys

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE))
from holdout import window_cost


def test_actual_accrual_and_missing_endpoint():
    rows=[dict(time_seconds=18300,accrued_generalized_passenger_seconds=100.),
          dict(time_seconds=24000,accrued_generalized_passenger_seconds=370.)]
    assert window_cost(rows)['generalized_passenger_seconds'] == 270.
    missing=window_cost(rows[:1])
    assert missing['status'] == 'missing_endpoint'
    assert missing['generalized_passenger_seconds'] is None
    assert missing['missing_times'] == [24000.]
