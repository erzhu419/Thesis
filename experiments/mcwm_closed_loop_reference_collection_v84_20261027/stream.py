"""Stop only the frozen parent at its real sampling-window boundary."""
from experiments.mcwm_closed_loop_supervision_plan_v84_20261027.run import RemainingDayStream


class CollectionStream(RemainingDayStream):
    """Keep original prefix stepping, then bound the parent's sampling window.

    Enable the limit only after packet37. Each fork child calls enter_suffix
    before the half-issued root resumes; it then follows the original native
    path all the way to the real episode terminal.
    """
    def __init__(self, *args, prefix_limit_seconds=12600., **kwargs):
        super().__init__(*args, **kwargs)
        self.prefix_limit_seconds = float(prefix_limit_seconds)
        self.prefix_limit_enabled = False
        self.prefix_limit_reached = False
        self.is_suffix = False
        self._original_env_step = self.env.step_to_event

    def enable_prefix_limit(self):
        if self.is_suffix:
            raise RuntimeError('Only the parent has a sampling-window limit')
        self.prefix_limit_enabled = True

    def enter_suffix(self):
        self.is_suffix = True
        self.prefix_limit_enabled = False
        self.prefix_limit_reached = False

    def _advance_parent_window(self, commands):
        # A child can be forked inside this very step's selector callback.
        # Check its role here, after that callback, instead of capturing the
        # parent's role when step() installed this temporary advancement path.
        if self.is_suffix or not self.prefix_limit_enabled:
            return self._original_env_step(commands)
        env, bridge = self.env, self.env._bridge
        for event in env._current_events:
            bridge.apply_action(event, commands.get(event.bus_id, [0., 1.]))
        env.state, env.reward, env._current_events = {}, {}, []
        while not bridge.done:
            now = float(bridge.passenger_time_ledger.snapshot().time_seconds)
            if now >= self.prefix_limit_seconds:
                if now != self.prefix_limit_seconds:
                    raise RuntimeError('Native parent overshot the sampling window')
                self.prefix_limit_reached = True
                return
            if bridge.decision_queue:
                events, done, _departed = bridge.fetch_events()
                env.done = bool(done)
                env._current_events = events
                for event in events:
                    env.state[event.bus_id] = [[env._event_to_obs(event).tolist()]]
                    env.reward[event.bus_id] = env._compute_reward(event)
                return
            bridge._advance_one_step()
        env.done = True

    def step(self, policy):
        if self.prefix_limit_reached and not self.is_suffix:
            return self.release()
        if not self.prefix_limit_enabled or self.is_suffix:
            return super().step(policy)
        self.env.step_to_event = self._advance_parent_window
        try:
            return super().step(policy)
        finally:
            self.env.step_to_event = self._original_env_step
