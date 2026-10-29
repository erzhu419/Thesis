"""The preflight seed is reused; continuation never expands the fixed design."""
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from experiments.mcwm_closed_loop_tail_policy_exchange_v71_20261026 import launch as mod


@pytest.fixture
def campaign(tmp_path, monkeypatch):
    old = mod.module("v71_test_actual_preflight", mod.V70/"launch.py")
    helper = mod.module("v71_test_actual_readonly", mod.V60/"supervisor.py")
    protocol = dict(seeds=[6201, 6203], native_preflight=dict(seed=6201, scenario="stable",
        reuse_full_day=True, feedback_slots=5, policy_updates=8, critic_updates=8,
        value_updates=8, actor_updates=8, alpha_updates=8))
    (tmp_path/"protocol.json").write_text(json.dumps(protocol))
    (tmp_path/"execution_ready.json").write_text('{"ready":true}')
    outputs, submissions, states = {}, [], {}

    def task(label, runner, argv):
        return dict(signature="v71/"+label, runner=str(runner), argv=argv, priority="high")

    def submit(rows, batch):
        path = tmp_path/(batch+"_submission.json")
        assert not path.exists(), "The launcher must reuse existing submission receipts"
        value = dict(submitted=[dict(id=f"t{len(submissions)}", signature=rows[0]["signature"])])
        path.write_text(json.dumps(value))
        submissions.append(dict(batch=batch, rows=rows))
        return value

    infra = SimpleNamespace(task=task, remote=str, submit=submit,
        presence=lambda paths: {path: path in outputs for path in paths}, collect=lambda path: outputs[path],
        status=lambda ids: [dict(id=identity, status=states.get(identity, "running")) for identity in ids])
    monkeypatch.setattr(mod, "HERE", tmp_path)
    monkeypatch.setattr(mod, "module", lambda name, path: helper if path.name == "supervisor.py"
        else old if path.parent == mod.V70 else infra)

    def preflight():
        for arm in ("fresh", "no_fresh"):
            outputs[f"runs/seed6201/{arm}/preflight.json"] = dict(technical_gate_pass=True,
                seed=6201, scenario="stable", arm=arm, feedback_slots=5, policy_updates=8,
                critic_updates=8, value_updates=8, actor_updates=8, alpha_updates=8,
                packet_schedule=[8, 8, 8, 8], initializer=dict(updates=50000),
                first_feedback=dict(time=1205), execution_contract=dict(F="unchanged"))

    def native_done():
        preflight()
        for seed, identity in ((6201, "t0"), (6203, "t1")):
            outputs[f"runs/seed{seed}/seed_summary.json"] = dict(status="complete",
                integrity=dict(all_pass=True, checks=dict(native=True)))
            states[identity] = "done"

    return SimpleNamespace(root=tmp_path, infra=infra, outputs=outputs, submissions=submissions,
        states=states, preflight=preflight, native_done=native_done)


def test_first_seed_reused_and_second_seed_submitted_exactly_once(campaign):
    first = mod.advance()
    mod.advance()
    assert first["stage"] == "tail_exchange_native_preflight"
    assert len(campaign.submissions) == 1
    campaign.preflight()
    assert mod.advance()["stage"] == "tail_exchange_native"
    mod.advance()
    assert [batch["batch"] for batch in campaign.submissions] == ["tail_exchange_preflight", "tail_exchange_remaining"]
    jobs = [batch["rows"][0] for batch in campaign.submissions]
    assert [job["argv"][1] for job in jobs] == [6201, 6203]
    assert all(Path(job["runner"]).name == "pair.py" and job["priority"] == "high" for job in jobs)
    assert all("--stop" not in job["argv"] for job in jobs)


def test_failed_preflight_stops_remaining_seed(campaign):
    campaign.preflight()
    campaign.outputs["runs/seed6201/no_fresh/preflight.json"]["initializer"] = dict(updates=49999)
    result = mod.advance()
    assert result["done"] is True
    assert result["preflight"]["all_pass"] is False
    assert result["native_tasks_may_still_run"] is True
    assert len(campaign.submissions) == 1


def test_native_summary_missing_is_failure_not_a_completed_zero_cost_day(campaign):
    campaign.preflight()
    campaign.states["t0"] = "done"
    result = mod.advance()
    assert result["done"] is True
    assert result["completed"] == 0
    assert result["failures"][0]["reason"] == "terminal_seed_summary_missing"
    assert len(campaign.submissions) == 1


def test_summary_waits_for_both_seed_jobs_and_is_not_duplicated(campaign):
    campaign.preflight()
    mod.advance()
    campaign.native_done()
    assert mod.advance()["stage"] == "tail_exchange_summary"
    mod.advance()
    assert len(campaign.submissions) == 3
    assert campaign.submissions[-1]["batch"] == "tail_exchange_analysis"
    assert campaign.submissions[-1]["rows"][0]["argv"] == ["--out", str(campaign.root/"results/diagnostic.json")]


@pytest.mark.parametrize("supported", [True, False])
def test_either_mechanistic_outcome_closes_without_new_recipe(campaign, supported):
    campaign.native_done()
    campaign.outputs["results/diagnostic.json"] = dict(status="complete", integrity=dict(all_pass=True),
        decision=dict(feedback_redistribution_may_be_considered=supported))
    campaign.outputs["results/service.json"] = dict(status="complete")
    campaign.states["t2"] = "done"
    result = mod.advance()
    mod.advance()
    assert result["done"] is True
    assert result["stage"] == "tail_exchange_complete_requires_root_review"
    assert result["new_recipe_started"] is False
    assert result["decision"]["feedback_redistribution_may_be_considered"] is supported
    assert len(campaign.submissions) == 3


def test_terminal_partial_summary_never_counts_as_complete(campaign):
    campaign.native_done()
    campaign.outputs["results/diagnostic.json"] = dict(status="complete", integrity=dict(all_pass=True))
    campaign.states["t2"] = "done"
    result = mod.advance()
    assert result["stage"] == "tail_exchange_summary_attention"
    assert result["missing_summary"] == ["results/service.json"]
    assert len(campaign.submissions) == 3


def test_transient_status_error_reuses_existing_receipt(campaign):
    original = campaign.infra.status
    campaign.infra.status = lambda ids: (_ for _ in ()).throw(RuntimeError("waiting for scheduler state lock"))
    assert mod.advance()["stage"] == "waiting_transient_tail_exchange_read"
    campaign.infra.status = original
    assert mod.advance()["stage"] == "tail_exchange_native_preflight"
    assert len(campaign.submissions) == 1
