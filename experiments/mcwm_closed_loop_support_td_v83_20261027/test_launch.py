"""Do not pay for later days before the reused first packet passes."""
import json
from pathlib import Path
from types import SimpleNamespace
import pytest

from experiments.mcwm_closed_loop_support_td_v83_20261027 import launch


def setup(monkeypatch, tmp_path, *, ready, context_pass=True):
    protocol = json.loads((launch.HERE/'protocol.json').read_text())
    (tmp_path/'protocol.json').write_text(json.dumps(protocol))
    (tmp_path/'execution_ready.json').write_text('{"ready":true}')
    monkeypatch.setattr(launch, 'HERE', tmp_path)
    batches = []
    def submit(jobs, batch):
        batches.append((batch,len(jobs)))
        return {'submitted':[dict(id=str(index),signature=job['signature'])
                              for index,job in enumerate(jobs)]}
    infra = SimpleNamespace(remote=str,
        task=lambda label,runner,args:dict(signature='study/'+label),
        submit=submit, presence=lambda paths:{path:ready for path in paths},
        collect=lambda path:dict(context_first_packet={'all_pass':context_pass},
            td_gradient_gate={'all_pass':context_pass}),
        common_prefix=lambda rows:dict(all_pass=True))
    monkeypatch.setattr(launch, 'module', lambda *args:infra)
    monkeypatch.setattr(launch, 'preflight', lambda rows,p:dict(all_pass=True,failed_checks=[]))
    monkeypatch.setattr(launch, 'watch', lambda *args:dict(completed=0,failures=[],
        pending=[{'id':'current'}],all_terminal=False))
    return batches


def test_missing_first_packet_never_launches_later_days(monkeypatch,tmp_path):
    batches=setup(monkeypatch,tmp_path,ready=False)
    result=launch._advance(lambda fn,*args:fn(*args))
    assert batches == [('context_preflight',2)]
    assert result['stage']=='first_pair_native_preflight' and not result['done']


def test_failed_context_gate_stops_before_later_six_days(monkeypatch,tmp_path):
    batches=setup(monkeypatch,tmp_path,ready=True,context_pass=False)
    result=launch._advance(lambda fn,*args:fn(*args))
    assert batches == [('context_preflight',2)]
    assert result['done'] and result['stage']=='preflight_attention'


def test_reused_first_packet_pass_launches_remaining_without_waiting_first_day(monkeypatch,tmp_path):
    batches=setup(monkeypatch,tmp_path,ready=True)
    result=launch._advance(lambda fn,*args:fn(*args))
    assert batches == [('context_preflight',2),('context_remaining',6)]
    assert not result['done'] and result['stage']=='eight_fixed_native_days'
