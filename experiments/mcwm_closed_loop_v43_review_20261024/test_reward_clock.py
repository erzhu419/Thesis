import math

from experiments.mcwm_closed_loop_v43_review_20261024.reward_clock import summarize_pair


def fixture(diverge=False):
    # Arm partitions differ at ordinal1 but reach the same cumulative cost
    # at clocks2 and3; forty-one later atomic closures advance no physical time.
    def rows(clocks, costs):
        starts = [0.]+clocks[:-1]
        return [dict(started_at=a, completed_at=b, dt=b-a, r=-c, line_id='7X')
            for a, b, c in zip(starts, clocks, costs)]
    low = rows([1., 2., 2., 3.]+[3.]*41, [2., 2., 0., 2.]+[0.]*41)
    high = rows([1., 1., 2., 3.]+[3.]*41, [2., 0., 2., 2.+int(diverge)]+[0.]*41)
    def trace(rows):
        total, result = 0., []
        for row in rows:
            result.append(dict(issue_ledger=dict(time_seconds=row['started_at'],
                accrued_wait_seconds=total/2, accrued_in_vehicle_seconds=0.)))
            total -= row['r']
        return result
    packet = dict(updates=456, chains=[dict(line_id='7X', arm=arm,
        rows=[dict(log_c=-.1, expected_log_c=-.2, log_overlap=-.15)]*44)
        for arm in ('low', 'high')])
    return low, high, trace(low), trace(high), [packet]


def test_ordinal_partition_difference_is_not_physical_difference():
    result = summarize_pair(*fixture())
    assert result['first_ordinal_reward_difference'] == 1
    assert result['first_common_cost_difference'] is None
    assert result['last_common_closure']['high_minus_low_cost'] == 0.
    assert result['ledger_checks']['low']['reward_ledger_consistent']


def test_real_clock_and_credit_exclude_trailing_atomic_closures():
    result = summarize_pair(*fixture(True))
    first = result['first_common_cost_difference']
    assert first['clock'] == 3. and first['high_minus_low_cost'] == 1.
    assert first['low_links'] == first['high_links'] == 3
    assert first['low_closed_rows'] == first['high_closed_rows'] == 45
    assert first['ledger_high_minus_low']['cost'] == 1.
    actual = result['first_difference_credit'][0]
    assert actual[2] == 3 and math.isclose(actual[3], -.3)
    assert math.isclose(actual[6], 3*math.log(.9))
    assert math.isclose(actual[7], math.exp(-.3)*.9**3)
    assert result['missing_300s']
