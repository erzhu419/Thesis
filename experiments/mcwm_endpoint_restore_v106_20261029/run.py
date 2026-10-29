"""Audit the saved historical endpoint; two diagnostic updates, no collection."""
import argparse
import gc
import importlib
import importlib.util
import json
import os
from pathlib import Path
import platform
import sys
import time

from adjudicate import assess

HERE = Path(__file__).resolve().parent


def learning_state(payload):
    """Exclude only receipt metadata; preserve every saved learning field."""
    return dict(payload, extra={"mcwm_state": payload["extra"]["mcwm_state"]})


def bootstrap(root, original_protocol):
    for key in ("h2o_root", "dataset", "source_env", "mcwm_root"):
        os.environ["THESIS_" + key.upper()] = str(root / original_protocol[key])
    os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
    for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        os.environ[key] = "1"
    original = root / "experiments/mcwm_seed_confirmation_v13"
    sys.path.insert(0, str(original))
    spec = importlib.util.spec_from_file_location("v106_original_mcwm", original / "run.py")
    implementation = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(implementation)
    sys.path.insert(0, str(root / "experiments/mcwm_full_state_recovery_v105_20261029"))
    state = importlib.import_module("state")
    import torch
    torch.set_num_threads(1)
    return implementation, state, torch


def restore_learner(checkpoint, payload, implementation, state, out):
    base = implementation.base
    saved = payload["extra"]["mcwm_state"]
    binding = saved["replay_binding"]
    replay = base.TransitReplayBufferV7(binding["dataset_path"], device="cpu",
        split=binding["split"], online_capacity=payload["online_replay"]["capacity"],
        max_offline_rows=0, device_cache_observation_dtype=binding["device_cache_observation_dtype"])
    learner = implementation.MappingTrainer(replay,
        config=base.TransitTrainerConfig(**payload["config"]), device="cpu",
        seed=saved["context"]["protocol"]["training_seed"])
    learner.configure_context(saved["context"]["protocol"], "calibrated", None)
    learner.costs = base.Costs("cpu")
    model_class = implementation.previous.previous.ContextModel
    state.load_final(learner, checkpoint, model_class=model_class, out=out)
    return learner


def audit(args):
    root, out = args.thesis_root.resolve(), args.out.resolve()
    p = json.loads((HERE / "protocol.json").read_text())
    out.mkdir(parents=True, exist_ok=False)
    checkpoint, bridge = root / p["input_checkpoint"], out / "bridge.pt"
    result = dict(status="verification_failed", decision="HOLD_TECHNICAL_FAILURE_NO_RETRY",
        input_checkpoint=str(checkpoint), bridge_checkpoint=str(bridge),
        budget=dict(diagnostic_attempts=0, diagnostic_updates=0,
            new_offline_updates=0, source_collection_events=0, source_retraining_updates=0,
            target_native_seconds=0, target_feedback_labels=0, model_refits=0),
        comparisons={}, original_live_continuation_available=False,
        evidence_scope="Saved v105 state and save/load continuation from its restored endpoint",
        next_stage_authorized=False)
    learner = None
    state = None
    started = time.monotonic()

    def progress(stage):
        print(json.dumps(dict(stage=stage, budget=result["budget"])), flush=True)

    def check(name, expected, actual):
        comparison = state.compare_states(expected, actual)
        result["comparisons"][name] = comparison
        if not comparison["all_pass"]:
            result["decision"] = "HOLD_" + name.upper() + "_MISMATCH_NO_EXPANSION"
            return False
        return True

    def update_once(branch):
        if result["budget"]["diagnostic_attempts"] >= p["diagnostic_update_cap"]:
            raise RuntimeError("The two diagnostic update attempts are exhausted")
        result["budget"]["diagnostic_attempts"] += 1
        progress(branch + "_diagnostic_update")
        learner.update(online=True, collect_metrics=False)
        result["budget"]["diagnostic_updates"] += 1

    try:
        original_protocol = json.loads((root / p["v105_protocol"]).read_text())
        original_summary = json.loads((root / p["input_summary"]).read_text())
        result["identity_adjudication"] = assess(original_summary, original_protocol)
        if not result["identity_adjudication"]["all_pass"]:
            result["decision"] = "HOLD_IDENTITY_ADJUDICATION_NO_EXPANSION"
            return result
        implementation, state, torch = bootstrap(root, original_protocol)
        runtime = dict(host=platform.node(), python=platform.python_version(),
            torch=torch.__version__, device="cpu", threads=torch.get_num_threads())
        result["runtime"] = runtime
        declared = original_protocol["training_runtime"]
        if any(runtime[key] != declared[key] for key in ("python", "torch", "device", "threads")):
            raise RuntimeError("The original training runtime is required")
        payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
        result["receipt_metadata"] = {key: value for key, value in payload["extra"].items()
                                      if key != "mcwm_state"}
        result["input_checkpoint_bytes"] = checkpoint.stat().st_size
        reference = torch.load(root / p["reference_actor"], map_location="cpu", weights_only=False)
        if not check("checkpoint_actor_identity", reference["actor"], payload["actor"]):
            return result
        del reference
        if (payload["total_updates"] != 70000 or payload["online_critic_updates"] != 20000
                or payload["extra"]["seed"] != p["training_seed"]
                or p["diagnostic_update_cap"] != 2):
            raise ValueError("Expected the fixed 6201 historical endpoint and two-update audit")
        endpoint = learning_state(payload)
        progress("restore_A")
        learner = restore_learner(checkpoint, payload, implementation, state, out / "A")
        if not check("saved_to_restored_A", endpoint, learning_state(state.export_state(learner))):
            return result
        state.save_final(learner, bridge, extra=dict(
            input_checkpoint=str(checkpoint), training_seed=p["training_seed"],
            origin="Restored v105 endpoint; original pre-load live branch unavailable"))
        result["bridge_checkpoint_bytes"] = bridge.stat().st_size
        update_once("A_live")
        expected_next = learning_state(state.export_state(learner))
        # Release the large resident offline replay before constructing B.
        del learner
        learner = None
        gc.collect()
        progress("restore_B_from_bridge")
        learner = restore_learner(bridge, payload, implementation, state, out / "B")
        if not check("bridge_to_restored_B", endpoint, learning_state(state.export_state(learner))):
            return result
        update_once("B_restored")
        if not check("restored_endpoint_next_update", expected_next,
                     learning_state(state.export_state(learner))):
            return result
        result["status"] = "complete"
        result["decision"] = "SAVED_ENDPOINT_AND_RESTORED_CONTINUATION_VERIFIED"
        return result
    except Exception as error:
        result["error"] = dict(type=type(error).__name__, message=str(error))
        raise
    finally:
        if learner is not None and state is not None:
            try:
                recovery_path = bridge if bridge.exists() else checkpoint
                state.load_final(learner, recovery_path,
                    model_class=type(learner.augmentation), out=out / "terminal")
                result["terminal_updates"] = learner.total_updates
                if not check("terminal_70000_restoration", endpoint,
                             learning_state(state.export_state(learner))):
                    result["status"] = "verification_failed"
            except Exception as error:
                result["status"] = "verification_failed"
                result["decision"] = "HOLD_TERMINAL_RESTORE_FAILURE_NO_EXPANSION"
                result["terminal_restore_error"] = dict(type=type(error).__name__, message=str(error))
        result["budget"]["partial_diagnostic_update_possible"] = (
            result["budget"]["diagnostic_attempts"] != result["budget"]["diagnostic_updates"])
        result["wall_seconds"] = time.monotonic() - started
        with (out / "summary.json").open("x") as handle:
            json.dump(result, handle, ensure_ascii=False, indent=2, allow_nan=False)
            handle.write("\n")
        print(json.dumps(dict(stage="finished", status=result["status"],
            decision=result["decision"], budget=result["budget"])), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--thesis-root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    result = audit(parser.parse_args())
    if result["status"] != "complete":
        raise SystemExit(1)
