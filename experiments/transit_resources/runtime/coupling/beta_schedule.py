"""
coupling/beta_schedule.py
=========================
Beta annealing schedule for Temporal Advantage Propagation (TAP).

Directly adapted from RoboDuet's GlobalSwitch.get_beta() (global_switch.py L62-70).

Stage 1 (ep 0..warmup): β=0, only lower policy trains
Stage 2 (ep warmup..warmup+ramp): β linearly increases 0→β_max
Stage 3 (ep warmup+ramp..): β=β_max, full coupling
"""


class BetaSchedule:
    """Linear beta annealing from RoboDuet."""

    def __init__(self, warmup_eps=50, ramp_eps=100, beta_max=0.5):
        """
        Args:
            warmup_eps: number of episodes with β=0 (only lower trains)
            ramp_eps: number of episodes to linearly ramp β to β_max
            beta_max: maximum β value (RoboDuet uses 0.5)
        """
        self.warmup = warmup_eps
        self.ramp = ramp_eps
        self.beta_max = beta_max

    def get_beta(self, episode):
        """
        Get current β for cross-advantage injection.

        Args:
            episode: current episode index (0-based)
        Returns:
            float: β ∈ [0, beta_max]
        """
        if episode <= self.warmup:
            return 0.0
        elif episode < self.warmup + self.ramp:
            progress = (episode - self.warmup) / self.ramp
            return self.beta_max * progress
        else:
            return self.beta_max

    @property
    def is_warmup(self):
        """Convenience: true during Stage 1."""
        return True  # caller should check get_beta(ep) == 0

    def stage_name(self, episode):
        if episode <= self.warmup:
            return "Stage1_Warmup"
        elif episode < self.warmup + self.ramp:
            return "Stage2_Ramp"
        else:
            return "Stage3_Full"
