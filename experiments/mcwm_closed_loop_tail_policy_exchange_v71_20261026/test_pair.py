"""Couple both actor-producing prefixes in one reservation without reruns."""
import json
from pathlib import Path

import pytest
from experiments.mcwm_closed_loop_tail_policy_exchange_v71_20261026 import pair


def fake_children(monkeypatch, *, failed=None):
    calls=[]
    def create(command,**kwargs):
        arm=command[command.index('--arm')+1]
        out=Path(command[-1]);out.mkdir(parents=True)
        (out/'prefix_summary.json').write_text(json.dumps(dict(status='complete',integrity=dict(all_pass=True))))
        for branch in ('low','high'):
            p=out/'suffixes'/branch;p.mkdir(parents=True)
            (p/'summary.json').write_text('{}')
        class Child:
            pid=100+len(calls)
            returncode=1 if arm==failed else 0
            def poll(self):return self.returncode
        calls.append(command)
        return Child()
    monkeypatch.setattr(pair.subprocess,'Popen',create)
    return calls


def test_both_prefixes_share_reserved_seed_and_four_tail_paths(tmp_path,monkeypatch):
    calls=fake_children(monkeypatch)
    result=pair.run(6201,tmp_path/'seed6201')
    assert len(calls)==2
    assert [cmd[cmd.index('--arm')+1] for cmd in calls]==['fresh','no_fresh']
    assert all(cmd[cmd.index('--seed')+1]=='6201' for cmd in calls)
    assert result['status']=='complete' and result['integrity']['all_pass']


def test_failed_prefix_is_retained_as_failure_not_success_or_retried(tmp_path,monkeypatch):
    calls=fake_children(monkeypatch,failed='fresh')
    out=tmp_path/'seed6203'
    with pytest.raises(RuntimeError,match='no rerun'):pair.run(6203,out)
    assert len(calls)==2
    result=json.loads((out/'seed_summary.json').read_text())
    assert result['status']=='failed' and not result['integrity']['all_pass']
    assert result['children']['fresh']['returncode']==1


def test_existing_seed_output_never_starts_another_pair(tmp_path,monkeypatch):
    calls=fake_children(monkeypatch)
    out=tmp_path/'seed6201';out.mkdir()
    with pytest.raises(FileExistsError):pair.run(6201,out)
    assert calls==[]
