"""Mechanism-structured extension of the frozen normalized-confidence trial."""

import argparse
from functools import partial
import json
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent/"mcwm_usage_v3"))
from usage_common import P as PREVIOUS, module, write

usage = module("factored_usage_runner", HERE.parent/"mcwm_usage_v3/run.py")
import adaptive_model
import control_model
from factored_model import FactoredBusModel, SLOT_PARENTS, GLOBAL_PARENTS, OUTPUTS

P = dict(PREVIOUS, **json.loads((HERE/"protocol.json").read_text()))


class MechanismModel(usage.UsageModel):
    def __init__(self, replay, protocol, out, device, costs):
        super().__init__(replay, protocol, out, device, costs)
        model = self.banks["direct"][0]
        write(Path(out)/"architecture.json", dict(structure=protocol["structure"],
            trainable_parameters_per_direct_member=sum(p.numel() for p in model.parameters()),
            slot_parents=SLOT_PARENTS, global_parents=GLOBAL_PARENTS,
            restricted_inputs=protocol["structure"] == "factored",
            output_sizes={k:len(v) for k,v in OUTPUTS.items()},
            cost_conditioning="Predicted normalized log duration and waiting/onboard increments"))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--structure", choices=P["structures"], required=True)
    parser.add_argument("--arm", choices=P["arms"], required=True)
    parser.add_argument("--scenario", choices=[s["name"] for s in P["scenarios"]], required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--evaluate", action="store_true")
    parser.add_argument("--case", type=int, default=0)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    args.out = args.out.resolve()
    args.out.mkdir(parents=True, exist_ok=True)
    p = dict(P, training_seed=args.seed, arm=args.arm, structure=args.structure)
    if args.smoke:
        assert "smoke" in args.out.parts
        p.update(warmup_events=128, source_event_budget=192, actor_taper_event=160,
                 model_refresh_events=[128,160], model_updates_per_member=2, model_refit_updates=2,
                 model_ensemble_members=1, model_max_fit_rows=256, model_max_validation_rows=64,
                 evaluation_horizon_seconds=1200)
    if args.evaluate:
        usage.evaluate(args, p)
        path = args.out/"results"/f"eval_{args.case:02d}.json"
        write(path, dict(json.loads(path.read_text()), structure=args.structure))
        return
    if (args.out/"results/ready.json").exists():
        return
    factory = partial(FactoredBusModel, masked=args.structure == "factored")
    adaptive_model.BusModel = control_model.BusModel = factory
    usage.base.HERE = HERE.parent/"mcwm_alignment_v2"
    usage.UsageTrainer.arm, usage.UsageTrainer.output = args.arm, args.out/"results"
    usage.base.ModelTrainer, usage.base.AdaptiveModel = usage.UsageTrainer, MechanismModel
    original_env = usage.base.AllLinesSimEnv

    def environment(*items, **kwargs):
        env = original_env(*items, **kwargs)
        if env.demand_scale > 1.25:
            env._observation_speed_profiles_by_demand[env.demand_scale] = env._observation_speed_profiles_by_demand[1.25]
        return env

    usage.base.AllLinesSimEnv = environment
    branch = argparse.Namespace(**vars(args))
    branch.arm = "mcwm"
    usage.base.train(branch, p)
    for name in ("protocol", "train_summary", "ready"):
        path = args.out/"results"/(name+".json")
        write(path, dict(json.loads(path.read_text()), arm=args.arm, structure=args.structure))


if __name__ == "__main__":
    main()
    print("DONE mechanism decomposition", flush=True)
