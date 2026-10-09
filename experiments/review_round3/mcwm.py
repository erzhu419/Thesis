"""Matched model-loss coefficient control and non-invasive mechanism traces."""

from review_common import HERE, EXPERIMENTS, P, module, write, distribution
import argparse
import json
import time
import torch

old = module("round3_alignment", EXPERIMENTS / "mcwm_alignment_v2/run.py")
from model import losses


def combined_loss(real, model, fraction, arm):
    if arm == "h2o_matched":
        return real
    return (1 - fraction) * real + (fraction * model if model is not None else 0)


def gradient_relation(real, model, parameters, fraction):
    gr = torch.autograd.grad(real, parameters, retain_graph=True, allow_unused=True)
    gm = torch.autograd.grad(model, parameters, retain_graph=True, allow_unused=True)
    gr = torch.cat([(g if g is not None else torch.zeros_like(p)).flatten()
                    for g, p in zip(gr, parameters)])
    gm = torch.cat([(g if g is not None else torch.zeros_like(p)).flatten()
                    for g, p in zip(gm, parameters)])
    nr, nm = gr.norm(), gm.norm()
    return dict(real_gradient_norm=float(nr), model_gradient_norm=float(nm),
                scaled_real_gradient_norm=float((1-fraction)*nr),
                scaled_model_gradient_norm=float(fraction*nm),
                gradient_cosine=float(torch.dot(gr, gm)/(nr*nm)) if nr*nm > 0 else None)


def timed_actor(actor, samples):
    def forward(*args, **kwargs):
        started = time.perf_counter()
        result = actor(*args, **kwargs)
        samples.append(1000*(time.perf_counter()-started))
        return result
    return forward


def evaluate(args, protocol):
    import evaluate_control as control
    path = args.out / "results" / f"eval_{args.case:02d}.json"
    if path.exists() and "policy_inference_ms" in json.loads(path.read_text()):
        return
    original_loader = control.load_actor_checkpoint_v7
    samples = []

    def load(*items, **kwargs):
        actor, metadata = original_loader(*items, **kwargs)
        return timed_actor(actor, samples), metadata

    control.load_actor_checkpoint_v7 = load
    try:
        old.evaluate(args, protocol)
    finally:
        control.load_actor_checkpoint_v7 = original_loader
    row = json.loads(path.read_text())
    if samples:
        row["policy_inference_ms"] = distribution(samples)
        row["inference_scope"] = "CPU batch-one actor forward only; model and trust networks are not deployed; concurrent-server wall latency, not an isolated hardware benchmark"
        write(path, row)


class DiagnosticTrainer(old.ModelTrainer):
    arm = None
    output = None

    def _ensemble_td_loss(self, predictions, targets, weights=None):
        real = old.TransitH2OTrainerV7._ensemble_td_loss(self, predictions, targets, weights)
        if self.collect_diagnostics:
            self.real_losses.append(real)
            fraction = self.config.online_fraction
            expected = 2 if 0 < fraction < 1 else 1
            if len(self.real_losses) == expected:
                real_total = ((1-fraction)*self.real_losses[0] + fraction*self.real_losses[1]
                              if expected == 2 else real)
                self.loss_diagnostics = dict(real_td_loss=float(real_total.detach()))
            if len(self.real_losses) == expected and self.model_loss is not None:
                parameters = tuple(p for p in self.critic.parameters() if p.requires_grad)
                self.loss_diagnostics.update(gradient_relation(real_total, self.model_loss, parameters,
                                                               self.model_fraction))
        return combined_loss(real, self.model_loss, self.model_fraction, self.arm)

    def update(self, *, online, online_only=False, collect_metrics=True):
        collect_metrics = collect_metrics or (self.protocol["source_event_budget"] < 1000
                                               and (self.total_updates+1) % 16 == 0)
        self.collect_diagnostics = collect_metrics
        self.loss_diagnostics = {}
        self.real_losses = []
        result = super().update(online=online, online_only=online_only, collect_metrics=collect_metrics)
        if collect_metrics:
            row = dict(total_updates=self.total_updates, arm=self.arm,
                       real_td_coefficient=1. if self.arm == "h2o_matched" else 1-self.model_fraction,
                       metrics=result, **self.loss_diagnostics)
            if self.augmentation is not None:
                with self.costs.phase("mechanism_diagnostics"), torch.no_grad():
                    row["model"] = self.model_diagnostics()
            write(self.output / "diagnostics" / f"update{self.total_updates:06d}.json", row)
        self.real_losses = []
        return result

    def model_diagnostics(self):
        aug = self.augmentation
        result = dict(selected_family=aug.family, tau=float(aug.trust._tau))
        for split in ("fit", "stop", "calibrate"):
            batch = aug.target[split]
            # Fixed evenly spaced rows, no training RNG consumption or selection by error.
            index = torch.linspace(0, len(batch["s"])-1,
                                   min(P["mcwm"]["diagnostic_rows"], len(batch["s"])),
                                   device=aug.device).long()
            batch = {key: value[index] for key, value in batch.items()}
            errors = losses(aug.predict(batch["s"], batch["a"]), batch, aug.norm).mean(1)
            confidence = aug.trust._c_step(errors)
            qdelta = aug.trust.net(batch["s"], batch["a"]).flatten()
            weights = qdelta.exp()
            result[split] = dict(
                unweighted_prediction_loss=distribution(errors.cpu().numpy()),
                local_confidence=distribution(confidence.cpu().numpy()),
                qdelta=distribution(qdelta.cpu().numpy()),
                weights=distribution(weights.cpu().numpy()),
                effective_sample_size=float(weights.sum().square()/weights.square().sum()),
                weight_sum=float(weights.sum()))
        result["trust_training_error"] = distribution(aug.errors.cpu().numpy())
        return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--arm", required=True, choices=P["mcwm"]["arms"])
    parser.add_argument("--seed", required=True, type=int)
    parser.add_argument("--scenario", required=True, choices=P["mcwm"]["scenarios"])
    parser.add_argument("--out", required=True, type=old.Path)
    parser.add_argument("--evaluate", action="store_true")
    parser.add_argument("--case", type=int, default=0)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    args.out = args.out.resolve()
    args.out.mkdir(parents=True, exist_ok=True)
    p = dict(old.P, training_seed=args.seed, evaluation_blocks=P["mcwm"]["evaluation_blocks"])
    if args.smoke:
        if "smoke" not in args.out.parts:
            parser.error("Smoke needs a separate output directory")
        p.update(warmup_events=128, source_event_budget=192, actor_taper_event=160,
                 model_refresh_events=[128, 160], model_updates_per_member=2, model_refit_updates=2,
                 model_ensemble_members=1, model_max_fit_rows=256, model_max_validation_rows=64)
    old.HERE = HERE
    DiagnosticTrainer.arm, DiagnosticTrainer.output = args.arm, args.out / "results"
    old.ModelTrainer = DiagnosticTrainer
    if args.evaluate:
        if args.smoke:
            p["evaluation_horizon_seconds"] = 1200
        evaluate(args, p)
    elif not (args.out / "results/ready.json").exists():
        old.train(args, p)


if __name__ == "__main__":
    main()
    print("DONE review supplement", flush=True)
