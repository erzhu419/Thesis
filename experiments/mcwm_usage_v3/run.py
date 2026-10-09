"""Matched bus continuation and confidence/source-backup interventions."""

from usage_common import HERE, EXPERIMENTS, P, MODEL_ARMS, module, write
import argparse
from dataclasses import asdict, replace
import json
from pathlib import Path
import resource
import time

diagnostic = module("usage_diagnostics", EXPERIMENTS / "review_round3/mcwm.py")
base = diagnostic.old
import numpy as np
import torch
from model import metrics, losses
from mechanisms import TargetSupport, blend_targets, update_normalized_trust


class UsageModel(base.AdaptiveModel):
    def __init__(self, replay, protocol, out, device, costs):
        super().__init__(replay, protocol, out, device, costs)
        self.mode = protocol["arm"]
        self.support = TargetSupport(self.target["fit"], self.target["stop"],
                                     protocol["coverage_fit_rows"], protocol["coverage_quantile"])

    def update_trust(self, actor):
        if self.mode not in ("normalized", "source_repair"):
            return super().update_trust(actor)
        ix = torch.as_tensor(self.rng.integers(len(self.fit_ns), size=128), device=self.device)
        return update_normalized_trust(self.trust, self.fit["s"][ix], self.fit["a"][ix],
            self.fit_ns[ix], torch.zeros(len(ix), device=self.device), self.errors[ix],
            lambda states: actor(states, deterministic=True)[0],
            self.protocol["qdelta_updates_per_extra_step"])

    @torch.no_grad()
    def used_weight(self, s, a):
        if self.mode == "fixed_half":
            return torch.full((len(s),), self.protocol["fixed_model_weight"], device=self.device)
        return self.trust.weight(s, a)

    def generate(self, actor, size):
        batch, weights = super().generate(actor, size)
        if self.mode == "fixed_half":
            weights = self.used_weight(batch["observations"], batch["actions"])
        return batch, weights


class UsageTrainer(diagnostic.DiagnosticTrainer):
    def save_policy_artifact(self, path, *, extra):
        return super().save_policy_artifact(path, extra=dict(extra, arm=self.arm))

    @staticmethod
    def _concatenate_batches(first, second):
        result = base.TransitH2OTrainerV7._concatenate_batches(first, second)
        result["source_mask"] = torch.cat((
            torch.zeros(len(first["observations"]), dtype=torch.bool, device=first["observations"].device),
            torch.ones(len(second["observations"]), dtype=torch.bool, device=second["observations"].device)))
        return result

    @torch.no_grad()
    def _target(self, batch, *, backup):
        target = super()._target(batch, backup=backup)
        if self.arm != "source_repair" or "source_mask" not in batch:
            return target
        aug = self.augmentation
        source = batch["source_mask"]
        physical = source & (batch["delta_t_seconds"].flatten() > 0)
        eligible = (physical & (batch["terminals"].flatten() < .5)
                    & (batch["observations"][:, 0] < self.protocol["source_repair_max_horizon_fraction"]))
        ix = torch.nonzero(eligible).flatten()
        accepted = ix[aug.support.accepted(batch["observations"][ix], batch["actions"][ix])] if len(ix) else ix
        counts = torch.stack((source.sum(), physical.sum(), eligible.sum(),
                              torch.tensor(len(accepted), device=self.device))).float()
        self.repair_counts += counts
        if not len(accepted):
            return target
        s, a = batch["observations"][accepted], batch["actions"][accepted]
        generator_state = aug.generator.get_state()
        # Additional model targets must not consume the actor/replay RNG stream.
        with torch.random.fork_rng(devices=[torch.cuda.current_device()]), self.costs.phase("source_target_repair"):
            torch.manual_seed(self.model_seed+self.total_updates+900000)
            try:
                ns, reward, dt = aug.decode(s, aug.predict(s, a))
                model_batch = dict(observations=s, actions=a, next_observations=ns,
                    rewards=reward, delta_t_seconds=dt, discounts=torch.ones(len(s), device=self.device),
                    terminals=torch.zeros(len(s), device=self.device))
                model_target = base.TransitH2OTrainerV7._target(self, model_batch, backup=backup)
                weight = aug.used_weight(s, a)
                difference = model_target-target[accepted]
                target[accepted] = blend_targets(target[accepted], model_target, weight)
                self.repair_shift += (weight*difference).abs().sum()
            finally:
                aug.generator.set_state(generator_state)
        return target

    def update(self, *, online, online_only=False, collect_metrics=True):
        if not hasattr(self, "repair_counts"):
            self.repair_counts = torch.zeros(4, device=self.device)
            self.repair_shift = torch.zeros((), device=self.device)
        result = super().update(online=online, online_only=online_only, collect_metrics=collect_metrics)
        if self.collect_diagnostics:
            path = self.output / "diagnostics" / f"update{self.total_updates:06d}.json"
            row = json.loads(path.read_text())
            row["source_repair"] = dict(zip(("source_rows", "physical_rows", "eligible_rows", "corrected_rows"),
                                            self.repair_counts.tolist()))
            row["source_repair"]["mean_absolute_target_shift_corrected"] = float(
                self.repair_shift/self.repair_counts[-1].clamp_min(1))
            write(path, row)
            self.repair_counts.zero_()
            self.repair_shift.zero_()
        return result

    def model_diagnostics(self):
        result = super().model_diagnostics()
        aug = self.augmentation
        for split in ("fit", "stop", "calibrate"):
            data = aug.target[split]
            ix = torch.linspace(0, len(data["s"])-1, min(self.protocol["diagnostic_rows"], len(data["s"])),
                                device=self.device).long()
            batch = {k: v[ix] for k, v in data.items()}
            output = aug.predict(batch["s"], batch["a"])
            result[split]["components"] = metrics(output, batch, aug.norm)
            weights = aug.used_weight(batch["s"], batch["a"])
            result[split]["actual_weights"] = diagnostic.distribution(weights.cpu().numpy())
            result[split]["support_fraction"] = float(aug.support.accepted(batch["s"], batch["a"]).float().mean())
            long_hold = batch["a"].flatten() >= 0
            if long_hold.any():
                subset = {k: v[long_hold] for k, v in batch.items()}
                result[split]["holding_at_least_30s"] = metrics(output[long_hold], subset, aug.norm)
            else:
                result[split]["holding_at_least_30s"] = dict(rows=0)
        result["coverage_radius"] = float(aug.support.radius)
        result["weight_mode"] = self.arm
        return result


def offline_train(args, p):
    torch.set_num_threads(1)
    torch.cuda.reset_peak_memory_stats()
    start = time.monotonic()
    initializer = EXPERIMENTS / "mcwm_alignment_v2/inputs" / f"h2oplus_seed{args.seed}.pt"
    payload = torch.load(initializer, map_location="cpu", weights_only=False)
    assert payload["total_updates"] == 50000
    config = replace(base.TransitTrainerConfig(**payload["source_config"]), use_ratio_correction=False)
    del payload
    paths = base.load_protocol()
    dataset = Path(paths["dataset"])
    if not dataset.is_absolute():
        dataset = Path(paths["h2o_root"])/dataset
    replay = base.TransitReplayBufferV7(dataset, device="cuda", online_capacity=30000,
                                      max_offline_rows=10000 if args.smoke else 0)
    trainer = base.TransitH2OTrainerV7(replay, config=config, device="cuda", seed=args.seed)
    trainer.load_offline_initializer(initializer)
    updates = p["source_event_budget"]-p["warmup_events"]
    write(args.out/"results/protocol.json", dict(protocol=p, arm=args.arm, training_seed=args.seed,
          config=asdict(config), offline_rows=replay.offline_size, target_online_training_events=0))
    for step in range(updates):
        rate = config.actor_learning_rate*(p["actor_taper_multiplier"]
               if step+p["warmup_events"] >= p["actor_taper_event"] else 1.)
        for group in trainer.actor_optimizer.param_groups:
            group["lr"] = rate
        row = trainer.update(online=False, collect_metrics=(step+1) % 1000 == 0)
        if row:
            write(args.out/"results/progress.json", dict(continuation_updates=step+1, metrics=row))
    trainer.save_policy_artifact(args.out/"policy.pt", extra=dict(arm=args.arm, seed=args.seed))
    summary = dict(arm=args.arm, seed=args.seed, scenario=args.scenario, source_events=0,
        continuation_updates=updates, total_policy_updates=trainer.total_updates,
        total_critic_updates=trainer.total_updates, target_online_training_events=0,
        training_wall_seconds=time.monotonic()-start, model_history=[], phases={},
        peak_host_rss_mb=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024,
        peak_gpu_allocated_mb=torch.cuda.max_memory_allocated()/2**20)
    write(args.out/"results/train_summary.json", summary)
    write(args.out/"results/ready.json", summary)


def evaluate(args, p):
    import evaluate_control as control
    from SimpleSAC import evaluate_transit_v17_tail_service_reserve as reserve
    samples = []
    original = control.completion_safety_action

    def record(raw, action, mode, config):
        transformed = reserve.v16.reserved_hold_action(raw, action, reserve.PARENT_RESERVE_HORIZON_FRACTION, config)
        executed = original(raw, action, mode, config)
        hold = lambda x: float(control.policy_action_to_hold_seconds_v7(x[0], config))
        samples.append((hold(action), hold(transformed), hold(executed)))
        return executed

    control.completion_safety_action = record
    try:
        diagnostic.evaluate(args, p)
    finally:
        control.completion_safety_action = original
    path = args.out/"results"/f"eval_{args.case:02d}.json"
    row = json.loads(path.read_text())
    if samples:
        actions = np.asarray(samples)
        row["execution_actions"] = dict(
            raw_holding_seconds=diagnostic.distribution(actions[:, 0]),
            transformed_holding_seconds=diagnostic.distribution(actions[:, 1]),
            executed_holding_seconds=diagnostic.distribution(actions[:, 2]),
            reserve_changed_fraction=float(np.mean(np.abs(actions[:, 2]-actions[:, 1]) > 1e-6)),
            final_60s_fraction=float(np.mean(np.abs(actions[:, 2]-60) < 1e-6)))
        write(path, row)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--arm", required=True, choices=P["arms"]+["offline_only"])
    parser.add_argument("--scenario", required=True, choices=[s["name"] for s in P["scenarios"]])
    parser.add_argument("--seed", required=True, type=int)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--evaluate", action="store_true")
    parser.add_argument("--case", type=int, default=0)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    p = dict(P, training_seed=args.seed, arm=args.arm)
    if args.smoke:
        assert "smoke" in args.out.parts
        p.update(warmup_events=128, source_event_budget=192, actor_taper_event=160,
                 model_refresh_events=[128,160], model_updates_per_member=2, model_refit_updates=2,
                 model_ensemble_members=1, model_max_fit_rows=256, model_max_validation_rows=64)
    if args.evaluate:
        if args.smoke:
            p["evaluation_horizon_seconds"] = 1200
        evaluate(args, p)
    elif not (args.out/"results/ready.json").exists():
        if args.arm == "offline_only":
            offline_train(args, p)
        else:
            base.HERE = EXPERIMENTS/"mcwm_alignment_v2"
            UsageTrainer.arm, UsageTrainer.output = args.arm, args.out/"results"
            base.ModelTrainer, base.AdaptiveModel = UsageTrainer, UsageModel
            original_env = base.AllLinesSimEnv

            def environment(*items, **kwargs):
                env = original_env(*items, **kwargs)
                if env.demand_scale > 1.25:
                    env._observation_speed_profiles_by_demand[env.demand_scale] = env._observation_speed_profiles_by_demand[1.25]
                return env

            base.AllLinesSimEnv = environment
            # The inherited schedule enables models through the mcwm runner branch.
            branch = argparse.Namespace(**vars(args))
            branch.arm = "mcwm" if args.arm in MODEL_ARMS else args.arm
            base.train(branch, p)
            for name in ("protocol", "train_summary", "ready"):
                path = args.out/"results"/(name+".json")
                row = json.loads(path.read_text())
                row["arm"] = args.arm
                write(path, row)


if __name__ == "__main__":
    main()
    print("DONE model use trial", flush=True)
