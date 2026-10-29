"""Keep v107 critic data; route auxiliary updates through the original batch."""

from types import MethodType

from experiments.mcwm_full_mcwm_dripping_pilot_v107_20261029 import learning as original

TargetReplayAdapter = original.TargetReplayAdapter


class OriginalAuxiliaryBatch:
    def __init__(self, learner, adapter):
        self.adapter = adapter
        self.offline = self.critic_offline = self.source = None
        self.captures = 0
        self.calls = dict(value=0, actor=0, temperature=0)
        self.original_used = dict(value=0, actor=0, temperature=0)
        sample = adapter._original_sample_mixed
        value_terms = learner._iql_value_terms
        actor_states = learner._actor_states
        temperature_states = learner._temperature_states

        def capture(batch_size, online_fraction, *, rng=None):
            offline, source = sample(batch_size, online_fraction, rng=rng)
            # v107 will mutate the returned offline suffix in place. All seven
            # original fields must remain independent for the original methods.
            self.offline = {key: value.clone() for key, value in offline.items()}
            self.critic_offline, self.source = offline, source
            self.captures += 1
            return offline, source

        def value(_learner, batch):
            preserved = self.for_channel("value", batch)
            return value_terms(preserved)

        def actor(_learner, offline, source):
            preserved = self.for_channel("actor", offline, source)
            return actor_states(preserved, source)

        def temperature(_learner, offline, source):
            preserved = self.for_channel("temperature", offline, source)
            return temperature_states(preserved, source)

        adapter._original_sample_mixed = capture
        learner._iql_value_terms = MethodType(value, learner)
        learner._actor_states = MethodType(actor, learner)
        learner._temperature_states = MethodType(temperature, learner)

    def for_channel(self, channel, batch, source=None):
        self.calls[channel] += 1
        if (self.offline is None or batch is not self.critic_offline
                or (channel != "value" and source is not self.source)):
            raise RuntimeError("Auxiliary update must use this update's captured original batch")
        self.original_used[channel] += 1
        return self.offline


def mechanism_summary(learner, replay):
    if replay.arm == "no_fresh":
        return dict(mode="original_no_fresh", auxiliary_captures=0,
            value_calls=0, actor_calls=0, temperature_calls=0,
            original_batch_used=dict(value=0, actor=0, temperature=0),
            critic_complete_replacements=replay.replaced, critic_replacement_unit="rows")
    mechanism = replay._original_auxiliary_batch
    return dict(mode="critic_only_fresh", auxiliary_captures=mechanism.captures,
        value_calls=mechanism.calls["value"], actor_calls=mechanism.calls["actor"],
        temperature_calls=mechanism.calls["temperature"],
        original_batch_used=dict(mechanism.original_used),
        critic_complete_replacements=replay.replaced, critic_replacement_unit="rows")


def install_mechanism(learner, replay):
    """Only fresh changes learning methods; no_fresh retains the v107 path."""
    if replay.arm == "fresh":
        replay._original_auxiliary_batch = OriginalAuxiliaryBatch(learner, replay)
    summary = replay.summary

    def with_mechanism():
        return dict(summary(), mechanism=mechanism_summary(learner, replay))

    replay.summary = with_mechanism
    return replay


def load_learner(root, out, arm, protocol):
    learner, replay, contract, stations, receipt = original.load_learner(root, out, arm, protocol)
    install_mechanism(learner, replay)
    return learner, replay, contract, stations, receipt
