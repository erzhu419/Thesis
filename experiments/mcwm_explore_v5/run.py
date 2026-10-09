"""Separate matched trials of three model-use interventions."""

import argparse
import json
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent/"mcwm_usage_v3"))
from usage_common import P as PREVIOUS, module, write

usage = module("explore_usage_runner", HERE.parent/"mcwm_usage_v3/run.py")
P = dict(PREVIOUS, **json.loads((HERE/"protocol.json").read_text()))


def main():
    parser = argparse.ArgumentParser()
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
    p = dict(P, training_seed=args.seed, arm=args.arm)
    if args.smoke:
        assert "smoke" in args.out.parts
        p.update(warmup_events=128, source_event_budget=192, actor_taper_event=160,
            model_refresh_events=[128, 160], model_updates_per_member=2, model_refit_updates=2,
            model_ensemble_members=1, model_max_fit_rows=256, model_max_validation_rows=64,
            evaluation_horizon_seconds=1200)
    if args.evaluate:
        usage.evaluate(args, p)
        return
    if (args.out/"results/ready.json").exists():
        return
    trainer, model = usage.UsageTrainer, usage.UsageModel
    if args.arm == "decision_confidence":
        import adaptive_model
        import control_model
        from decision_model import decision_model_class, logged_targets_factory
        adaptive_model.targets = control_model.targets = logged_targets_factory(adaptive_model.targets)
        model = decision_model_class(usage)

        class DecisionTrainer(usage.UsageTrainer):
            def __init__(self, *items, **kwargs):
                super().__init__(*items, **kwargs)
                model.learner = self

        trainer = DecisionTrainer
    elif args.arm == "source_weighted":
        from source_weight import source_model_class, source_trainer_class
        trainer, model = source_trainer_class(usage), source_model_class(usage)
    elif args.arm.startswith("shared_"):
        import adaptive_model
        import control_model
        from shared_model import SharedBusModel, shared_model_class
        adaptive_model.BusModel = control_model.BusModel = SharedBusModel
        model = shared_model_class(usage)
    trainer.arm, trainer.output = args.arm, args.out/"results"
    usage.base.HERE = HERE.parent/"mcwm_alignment_v2"
    usage.base.ModelTrainer, usage.base.AdaptiveModel = trainer, model
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
        write(path, dict(json.loads(path.read_text()), arm=args.arm))


if __name__ == "__main__":
    main()
    print("DONE three-direction exploration", flush=True)
