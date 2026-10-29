"""Restore all original historical critic rows at unchanged group weight."""

from types import MethodType

import torch

from experiments.mcwm_critic_only_fresh_v109_20261029 import learning as original

TargetReplayAdapter = original.TargetReplayAdapter


class HistoricalCoverage:
    def __init__(self, learner, replay):
        if learner.config.bootstrap_probability != 1.0:
            raise ValueError("Historical coverage requires the original unmasked TD mean")
        self.auxiliary = replay._original_auxiliary_batch
        self.last_capture = self.updates = self.critic_rows = self.bootstrap_rows = self.bootstrap_calls = 0
        self.main_rng_preserved = self.common_targets_exact = True
        self.history_target = None
        target, td_loss = learner._target, learner._ensemble_td_loss

        def with_history_target(_learner, batch, *, backup):
            # The model target has128 rows and precedes this update's replay draw.
            if len(batch["observations"]) != 256:
                return target(batch, backup=backup)
            if self.auxiliary.captures != self.last_capture + 1:
                raise RuntimeError("Expected one mixed target after the current replay capture")
            if self.history_target is not None:
                raise RuntimeError("Previous historical target was not consumed")
            cpu_before = torch.get_rng_state()
            devices = [torch.cuda.current_device()] if learner.device.type == "cuda" else []
            cuda_before = torch.cuda.get_rng_state() if devices else None
            main = target(batch, backup=backup)
            cpu_after = torch.get_rng_state()
            cuda_after = torch.cuda.get_rng_state() if devices else None
            historical_batch = learner._concatenate_batches(self.auxiliary.offline, self.auxiliary.source)
            with torch.random.fork_rng(devices=devices):
                torch.set_rng_state(cpu_before)
                if devices:
                    torch.cuda.set_rng_state(cuda_before)
                historical = target(historical_batch, backup=backup)
                self.bootstrap_calls += 1
                self.bootstrap_rows += len(historical_batch["observations"])
            self.main_rng_preserved &= torch.equal(torch.get_rng_state(), cpu_after)
            if devices:
                self.main_rng_preserved &= torch.equal(torch.cuda.get_rng_state(), cuda_after)
            self.common_targets_exact &= (torch.equal(main[:64], historical[:64])
                                          and torch.equal(main[128:], historical[128:]))
            self.history_target = historical[64:128]
            self.last_capture = self.auxiliary.captures
            return main

        def with_history_loss(_learner, predictions, targets, weights=None):
            if self.history_target is None:
                return td_loss(predictions, targets, weights)
            if predictions.shape[1] != 128 or weights is not None:
                raise RuntimeError("Expected the original unweighted offline128 TD call first")
            batch = self.auxiliary.offline
            additional_q = learner.critic(batch["observations"][64:], batch["actions"][64:])
            self.critic_rows += len(batch["observations"][64:])
            history_q = torch.cat((predictions[:, :64], additional_q), dim=1)
            history_target = torch.cat((targets[:64], self.history_target))
            combined_q = torch.cat((history_q, predictions[:, 64:]), dim=1)
            combined_target = torch.cat((history_target, targets[64:]))
            # The production192-row mean is .5*mean(history128)+.5*mean(fresh64).
            balanced = torch.cat((targets.new_full((128,), .75), targets.new_full((64,), 1.5)))
            self.history_target = None
            self.updates += 1
            return td_loss(combined_q, combined_target, balanced)

        learner._target = MethodType(with_history_target, learner)
        learner._ensemble_td_loss = MethodType(with_history_loss, learner)


def mechanism_summary(learner, replay):
    result = original.mechanism_summary(learner, replay)
    if replay.arm == "no_fresh":
        return dict(result, history_coverage_updates=0, history_critic_rows_per_update=0,
                    extra_critic_rows=0, extra_bootstrap_rows=0, extra_bootstrap_calls=0,
                    main_rng_preserved=True, common_targets_exact=True)
    coverage = replay._historical_coverage
    return dict(result, mode="critic_only_fresh_historical_coverage",
                history_coverage_updates=coverage.updates, history_critic_rows_per_update=128,
                extra_critic_rows=coverage.critic_rows, extra_bootstrap_rows=coverage.bootstrap_rows,
                extra_bootstrap_calls=coverage.bootstrap_calls,
                main_rng_preserved=coverage.main_rng_preserved,
                common_targets_exact=coverage.common_targets_exact)


def install_coverage(learner, replay):
    if replay.arm == "fresh":
        replay._historical_coverage = HistoricalCoverage(learner, replay)
    summary = replay.summary

    def with_mechanism():
        return dict(summary(), mechanism=mechanism_summary(learner, replay))

    replay.summary = with_mechanism
    return replay


def install_mechanism(learner, replay):
    original.install_mechanism(learner, replay)
    return install_coverage(learner, replay)


def load_learner(root, out, arm, protocol):
    learner, replay, contract, stations, receipt = original.load_learner(root, out, arm, protocol)
    install_coverage(learner, replay)
    return learner, replay, contract, stations, receipt
