"""Prevent log timing errors and changing-state comparisons becoming drift claims."""
import importlib.util
import json
from pathlib import Path

import pytest


spec = importlib.util.spec_from_file_location('v42_journal', Path(__file__).with_name('journal.py'))
journal = importlib.util.module_from_spec(spec)
spec.loader.exec_module(journal)


def packet(index, p):
    return dict(packet=index, updates=8*index, receipt=dict(processed_at=900+300*index),
        metrics=dict(zip(journal.METRICS, [p, .4, -.3, -.5, -.8]),
            **{'train/update':50000+8*(index-1)+1,
               'train/online_critic_update':8*(index-1)+1}))


def score(index, phase, proposal):
    return dict(packet=index, phase=phase, processed_at=900+300*index,
        rows=[dict(sequence=index, line_id='7X', issue=900+300*index,
                   context_observed_at=900+300*index, proposal_seconds=proposal)])


def test_current_metrics_use_pre_actor_step_weight_position():
    row = journal.compact_log(packet(3, .27))
    assert row['policy_updates_before'] == 16
    assert row['metric_update_completed'] == 17
    assert row['packet_updates_completed'] == 24
    bad = packet(3, .27)
    bad['metrics']['train/update'] = 50024
    with pytest.raises(ValueError, match='first of eight'):
        journal.compact_log(bad)


def test_only_paired_same_state_proposals_certify_threshold_crossing():
    before, after = score(3, 'packet_before', 0.), score(3, 'packet_after', .001)
    value = journal.fixed_state_crossings([before, after])
    assert value['first_crossing']['policy_update_interval'] == [16, 24]
    assert value['counts']['zero_to_positive'] == 1
    assert value['probability_values_recoverable'] is False
    after['rows'][0]['issue'] += 1
    with pytest.raises(ValueError, match='different scored states'):
        journal.fixed_state_crossings([before, after])


def test_changed_batch_probability_never_becomes_same_state_drift(tmp_path):
    records = [packet(1, .2), packet(2, .8), packet(3, .95)]
    scores = [score(i, phase, 0.) for i in range(1, 4)
              for phase in ('packet_before', 'packet_after')]
    for filename, rows in (('packets.jsonl', records), ('scores.jsonl', scores)):
        (tmp_path/filename).write_text(''.join(json.dumps(row)+'\n' for row in rows))
    value = journal.summarize_model(tmp_path, expected_packets=3)
    assert value['sampled_minibatch_first_threshold']['0.8']['policy_updates_before'] == 8
    assert value['fixed_state_gate_threshold']['first_crossing'] is None
    assert value['sampled_mean_is_fixed_state_drift'] is False
    assert value['fixed_state_probability_series_available'] is False
    assert value['updates'] == 24


def test_missing_source_packet_cannot_certify_complete_calendar(tmp_path):
    (tmp_path/'packets.jsonl').write_text(json.dumps(packet(2, .2))+'\n')
    with pytest.raises(ValueError, match='Missing or reordered'):
        journal.summarize_model(tmp_path, expected_packets=2)
