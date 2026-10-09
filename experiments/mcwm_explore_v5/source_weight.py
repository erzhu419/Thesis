"""Downweight source real TD errors without replacing observed transitions."""

import torch


def reliability_weights(residual, scale, covered, eligible, floor):
    error = (residual/scale).square().mean(1)
    confidence = floor+(1-floor)*torch.exp(-.5*error)
    confidence = torch.where(covered, confidence, torch.full_like(confidence, floor))
    return torch.where(eligible, confidence, torch.ones_like(confidence))


def source_model_class(usage):
    from model import unpack

    class SourceModel(usage.UsageModel):
        def __init__(self, replay, protocol, out, device, costs):
            super().__init__(replay, protocol, out, device, costs)
            self._refresh_residual_scale()

        @torch.no_grad()
        def _refresh_residual_scale(self):
            stop = self.target["stop"]
            output = unpack(self.predict(stop["s"], stop["a"]))["g"]
            truth = (stop["g"]-self.norm.g_mean)/self.norm.g_std
            self.source_residual_scale = (output-truth).square().mean(0).sqrt().clamp_min(
                self.protocol["source_residual_scale_floor"])
            usage.write(self.out/"source_weight_scale.json", dict(
                stopping_global_residual_rms=self.source_residual_scale.tolist(),
                coverage_radius=float(self.support.radius)))

        def refit(self, replay, event):
            super().refit(replay, event)
            self._refresh_residual_scale()

        def update_trust(self, actor):
            from mechanisms import update_normalized_trust
            ix = torch.as_tensor(self.rng.integers(len(self.fit_ns), size=128), device=self.device)
            return update_normalized_trust(self.trust, self.fit["s"][ix], self.fit["a"][ix],
                self.fit_ns[ix], torch.zeros(len(ix), device=self.device), self.errors[ix],
                lambda states: actor(states, deterministic=True)[0],
                self.protocol["qdelta_updates_per_extra_step"])

        @torch.no_grad()
        def source_weights(self, batch):
            s, a = batch["observations"], batch["actions"]
            dt, reward = batch["delta_t_seconds"].flatten(), batch["rewards"].flatten()
            eligible = (dt > 0) & (batch["terminals"].flatten() < .5)
            indices = torch.nonzero(eligible).flatten()
            weights = torch.ones(len(s), device=self.device)
            covered_count = 0
            if len(indices):
                ss, aa = s[indices], a[indices]
                ns = batch["next_observations"][indices]
                truth = torch.cat((torch.log1p(dt[indices])[:, None],
                    torch.log1p(-reward[indices])[:, None], (ns[:, 3:5]-ss[:, 3:5])*20000), 1)
                truth = (truth-self.norm.g_mean)/self.norm.g_std
                prediction = unpack(self.predict(ss, aa))["g"]
                covered = self.support.accepted(ss, aa)
                weights[indices] = reliability_weights(prediction-truth, self.source_residual_scale,
                    covered, torch.ones(len(indices), dtype=torch.bool, device=self.device),
                    self.protocol["source_weight_floor"])
                covered_count = int(covered.sum())
            return weights, dict(rows=len(s), physical_rows=len(indices), covered_rows=covered_count,
                weight_sum=float(weights.sum()), weighted_physical_sum=float(weights[indices].sum()))

    return SourceModel


def source_trainer_class(usage):
    class SourceTrainer(usage.UsageTrainer):
        @torch.no_grad()
        def _target(self, batch, *, backup):
            result = super()._target(batch, backup=backup)
            if "source_mask" in batch and self.augmentation is not None:
                mask = batch["source_mask"]
                source = {k: v[mask] for k, v in batch.items() if k != "source_mask"}
                with self.costs.phase("source_reliability"):
                    self.current_source_weights, row = self.augmentation.source_weights(source)
                for key, value in row.items():
                    self.source_statistics[key] = self.source_statistics.get(key, 0.)+value
            return result

        def _ensemble_td_loss(self, predictions, targets, weights=None):
            # The mixed learner calls real offline TD first, then source TD.
            # Synthetic model TD calls the raw parent method, so it never enters here.
            if self.real_td_calls == 1 and self.current_source_weights is not None:
                assert len(targets) == len(self.current_source_weights)
                weights = self.current_source_weights if weights is None else weights*self.current_source_weights
            self.real_td_calls += 1
            return super()._ensemble_td_loss(predictions, targets, weights)

        def update(self, *, online, online_only=False, collect_metrics=True):
            assert online and not online_only and self.config.online_fraction == .5
            self.real_td_calls, self.current_source_weights = 0, None
            if not hasattr(self, "source_statistics"):
                self.source_statistics = {}
            result = super().update(online=online, online_only=online_only, collect_metrics=collect_metrics)
            assert self.real_td_calls == 2
            if self.collect_diagnostics:
                import json
                path = self.output/"diagnostics"/f"update{self.total_updates:06d}.json"
                row = json.loads(path.read_text())
                totals = self.source_statistics
                row["source_reliability"] = dict(totals,
                    mean_weight=totals.get("weight_sum", 0)/max(1, totals.get("rows", 0)),
                    mean_physical_weight=totals.get("weighted_physical_sum", 0)/max(1, totals.get("physical_rows", 0)))
                usage.write(path, row)
                self.source_statistics = {}
            return result

    return SourceTrainer
