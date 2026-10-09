"""Mapping and routing interventions without changing the v10 training loop."""

import argparse
import json
from pathlib import Path
import platform
import torch
import numpy as np

from mapping_common import HERE, P, FIXED, CONTEXTS, module, write, bundle_path, smoke_protocol
from routing import window, GatedWindowFilter

previous = module("mapping_training", HERE.parent / "mcwm_context_control_v10/run.py")
previous.HERE, previous.FIXED, previous.bundle_path = HERE, FIXED, bundle_path


class MappingTrainer(previous.AdaptiveTrainer):
    def configure_context(self, p, scenario, bundle):
        super().configure_context(p, scenario, bundle)
        self.coefficient_count, self.coefficient_sum, self.changed_updates = 0, 0., 0
        self.coefficient_min, self.coefficient_max = None, None
        self.output = HERE / p["relative_out"] / "results"
        if self.arm in FIXED:
            return
        mapping = bundle["coefficients"] if self.arm == "legacy_window" else p["shared_coefficients"]
        model = bundle["physical_recognizer"]
        values = [mapping[c] for c in model["classes"]]
        options = dict(adjusted_context=p["adjusted_context"]) if self.arm == "shared_gated" else {}
        filter_type = GatedWindowFilter if self.arm == "shared_gated" else window.WindowFilter
        self.context_filter = filter_type(model, values, p["window_blocks"],
            p["global_coefficient"], p["fallback_score_margin"], **options)
        self.coefficient = self.context_filter.coefficient
        self.bundle = dict(bundle, coefficients=dict(mapping), selection_scope=p["selection"],
            mapping_source="v7_development" if self.arm == "legacy_window" else "v11_prespecified_hypothesis")

    def observe_completed(self):
        def observe(block):
            if self.context_filter is not None:
                self.coefficient = self.context_filter.observe(np.asarray(block["features"])[:3])
                classes = self.context_filter.model["classes"]
                self.trace.append(dict(end_source_row=block["end_source_row"], episode_index=self.episode_index,
                    belief=self.context_filter.belief.tolist(), coefficient=self.coefficient,
                    context_accepted=self.context_filter.decision[1],
                    predicted_context=classes[int(np.argmax(self.context_filter.belief))]))

        def reset():
            self.episode_index += 1
            if self.context_filter is not None:
                self.context_filter.reset()
                self.coefficient = self.context_filter.coefficient

        self.blocks.consume(self.replay_buffer, observe, reset)

    def update(self, *, online, online_only=False, collect_metrics=True):
        coefficient = self.coefficient
        result = super().update(online=online, online_only=online_only, collect_metrics=collect_metrics)
        if online:
            self.coefficient_count += 1
            self.coefficient_sum += coefficient
            self.changed_updates += coefficient != self.protocol["global_coefficient"]
            self.coefficient_min = coefficient if self.coefficient_min is None else min(self.coefficient_min, coefficient)
            self.coefficient_max = coefficient if self.coefficient_max is None else max(self.coefficient_max, coefficient)
        return result

    def save_context(self):
        super().save_context()
        write(self.output / "coefficient_usage.json", dict(arm=self.arm,
            updates=self.coefficient_count, update_mean_coefficient=self.coefficient_sum/self.coefficient_count,
            minimum_coefficient=self.coefficient_min, maximum_coefficient=self.coefficient_max,
            changed_update_fraction=self.changed_updates/self.coefficient_count,
            applied_mapping=None if self.arm in FIXED else self.bundle["coefficients"],
            scope="Every continuation gradient update, not a block or diagnostic-snapshot average"))


previous.ContextTrainer = MappingTrainer
train, evaluate = previous.train, previous.previous.evaluate


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--arm", required=True, choices=P["arms"])
    parser.add_argument("--scenario", required=True, choices=CONTEXTS)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--evaluate", action="store_true")
    parser.add_argument("--case", type=int, default=10, choices=range(10, 20))
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    args.out = args.out.resolve()
    args.out.mkdir(parents=True, exist_ok=True)
    p = dict(P, arm=args.arm, training_seed=args.seed, relative_out=str(args.out.relative_to(HERE)))
    if args.smoke:
        assert "smoke" in args.out.parts
        p = smoke_protocol(p)
    if args.evaluate:
        evaluate(args, p)
    elif not (args.out / "results/ready.json").exists():
        write(args.out / "results/runtime.json", dict(host=platform.node(), torch_version=torch.__version__,
            cuda_version=torch.version.cuda, gpu_model=None, device=P["training_device"]))
        train(args, p)
    print("DONE context mapping diagnosis", flush=True)
