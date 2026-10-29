"""No native simulation or real data updates; includes four synthetic Adam steps."""
import json
import os
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
import time
import traceback
from types import SimpleNamespace
import unittest

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "experiments/mcwm_bus/runtime/H2Oplus"))

import numpy as np
import torch
from support import (child_detach, compare_prefix_actions, flush_writers,
                     fork_children, json_value, root_actions, trace_adapter)


def write_json(path, value):
    with Path(path).open("x") as handle:
        json.dump(value, handle, allow_nan=False)


class SupportTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Same fixed CPU threading used by the actual learning runtime.
        torch.set_num_threads(1)
        torch.set_num_interop_threads(1)

    def test_detach_preserves_accounts_and_full_prefix_comparison(self):
        with TemporaryDirectory() as folder:
            parent, child = Path(folder)/"parent", Path(folder)/"low"
            parent.mkdir()
            names = ("actions", "commands", "ticks", "exchanges", "waiting_changes", "visit_events")
            observer = SimpleNamespace(out=parent, writers={name: (parent/(name+".jsonl")).open("x") for name in names},
                people={"p": {"ride": 7}}, visits=[{"trip_id": "b"}], current={"b": "same"},
                previous_waiting={"p"}, issues=1, ticks=10, commands=1, end=10.)
            people, visits, current = observer.people, observer.visits, observer.current
            updates = (parent/"updates.jsonl").open("x")
            observer.writers["actions"].write('{"sequence":0,"a":[0.2]}\n')
            updates.write('{"packet":1}\n')
            flush_writers(observer, updates)
            reference = Path(folder)/"reference.jsonl"
            reference.write_text('{"a":[0.2],"sequence":0}\n{"sequence":1,"a":[0.3]}\n')
            self.assertTrue(compare_prefix_actions(parent/"actions.jsonl", reference, 0)["all_pass"])
            self.assertFalse(compare_prefix_actions(parent/"actions.jsonl", reference, 1)["all_pass"])
            with (parent/"actions.jsonl").open("w") as handle:
                handle.write('{"sequence":0,"a":[0.9]}\n')
            self.assertFalse(compare_prefix_actions(parent/"actions.jsonl", reference, 0)["all_pass"])
            updates, manifest = child_detach(observer, updates, child)
            self.assertIs(observer.people, people)
            self.assertIs(observer.visits, visits)
            self.assertIs(observer.current, current)
            self.assertEqual(observer.previous_waiting, {"p"})
            observer.writers["actions"].write('{"sequence":1}\n')
            updates.write('{"packet":2}\n')
            flush_writers(observer, updates)
            self.assertEqual(manifest["prefix_ticks"], 10)
            self.assertEqual((parent/"updates.jsonl").read_text(), '{"packet":1}\n')
            self.assertEqual((child/"updates.jsonl").read_text(), '{"packet":2}\n')
            for writer in observer.writers.values():
                writer.close()
            updates.close()

    def test_declared_anchor_doses_use_original_F_and_reject_changed_state(self):
        from bus_h2o.mdp_contract_v7 import TransitMDPConfigV7
        binding = json.loads((HERE/"anchor_binding.json").read_text())
        contract = TransitMDPConfigV7(**binding["contract"])
        for item in binding["anchors"]:
            saved = item["records"]["fresh"]
            anchor = dict(action=saved["actions"], command=saved["commands"], visit=saved["visits"])
            action = anchor["action"]
            result = root_actions(anchor, action["raw_s"], action["proposed_a"], contract, item["low"])
            self.assertTrue(result["all_pass"])
            self.assertEqual(result["low"]["hold"], item["low"])
            self.assertEqual(result["high"]["hold"], item["high"])
            changed = np.asarray(action["raw_s"], np.float32).copy()
            changed[0] += 1
            self.assertFalse(root_actions(anchor, changed, action["proposed_a"], contract, item["low"])["all_pass"])

    def test_adam_after_live_fork_matches_unforked_continuation(self):
        """Detect a post-training fork deadlock or copied optimizer/RNG mismatch."""
        with TemporaryDirectory() as folder:
            out = Path(folder)
            observer = SimpleNamespace(writers={"actions": (out/"actions.jsonl").open("x")})
            trace = trace_adapter(observer)
            torch.manual_seed(114)
            model = torch.nn.Sequential(torch.nn.Linear(8, 16), torch.nn.Tanh(), torch.nn.Linear(16, 1))
            optimizer = torch.optim.Adam(model.parameters(), lr=.001)
            private_rng = np.random.default_rng(114)
            completed = 0

            def step():
                nonlocal completed
                optimizer.zero_grad(set_to_none=True)
                x = torch.randn(32, 8)
                y = torch.as_tensor(private_rng.normal(size=(32, 1)), dtype=torch.float32)
                (model(x)-y).square().mean().backward()
                optimizer.step()
                completed += 1

            def snapshot():
                return json_value(dict(model=model.state_dict(), optimizer=optimizer.state_dict(),
                    rng=torch.get_rng_state(), private_rng=private_rng.bit_generator.state, completed=completed))

            step()  # Adam moment tensors exist before the fork.
            before = snapshot()

            def prepare(arm):
                trace.writer.close()
                trace.writer = (out/arm/"actions.jsonl").open("x")

            try:
                branch = fork_children(out, trace, lambda: 1000., lambda: dict(updates=completed),
                                       prepare, timeout_seconds=20)
            except BaseException as error:
                trace.writer.close()
                details = "\n".join(path.read_text() for path in out.glob("*.log"))
                raise RuntimeError(str(error)+"\n"+details) from error
            if isinstance(branch, str):
                try:
                    started = time.time()
                    (out/branch/"ready").touch()
                    deadline = time.monotonic()+5
                    while not all((out/arm/"ready").exists() for arm in ("low", "high")):
                        if time.monotonic() >= deadline:
                            raise TimeoutError("Synthetic sibling did not enter fork continuation")
                        time.sleep(.001)
                    step()
                    write_json(out/branch/"state.json", snapshot())
                    write_json(out/branch/"summary.json", dict(status="complete", integrity=dict(all_pass=True),
                        execution_wall=dict(native_started_wall=started, native_finished_wall=time.time())))
                    trace.writer.close()
                    os._exit(0)
                except BaseException:
                    traceback.print_exc()
                    sys.stderr.flush()
                    os._exit(1)
            self.assertTrue(branch["all_pass"])
            self.assertEqual(snapshot(), before)
            step()
            expected = snapshot()
            for arm in ("low", "high"):
                self.assertEqual(json.loads((out/arm/"state.json").read_text()), expected)
            trace.writer.close()


if __name__ == "__main__":
    unittest.main()
