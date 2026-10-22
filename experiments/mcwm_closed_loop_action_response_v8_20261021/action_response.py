"""World-only action memory and global response covariates for isolated v8.

The 229-dimensional actor/critic input is unchanged.  These covariates retain
commands across global events until that same vehicle/trip is observed again.
Remaining commanded holding is a clock, not a claim of actual SUMO departure.
Physical next-stop duration remains a different endpoint from global dt/cost.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch


LINE_NAMES = ("7X", "102X", "705X")
MEMORY_KINDS = (
    "pending_intervals", "remaining_command_hold", "onboard_remaining_command_hold",
    "elapsed_interval", "predicted_remaining_interval", "physical_prediction_count",
)
ACTION_KINDS = (
    "nonholding_interval", "nonholding_available",
    "hold", "hold_squared", "hold_onboard", "hold_queue", "hold_demand",
    "hold_forward_deviation", "hold_backward_deviation", "hold_nonholding_interval",
    "hold_pending_intervals", "hold_remaining_command_hold",
)
MEMORY_FEATURE_NAMES = tuple(f"{line}_{kind}" for line in LINE_NAMES for kind in MEMORY_KINDS)
FEATURE_NAMES = MEMORY_FEATURE_NAMES + tuple(
    f"{line}_{kind}" for line in (*LINE_NAMES, "other") for kind in ACTION_KINDS)


@dataclass(frozen=True)
class PendingCommand:
    line: str
    vehicle: str
    trip: str
    issued_at: float
    hold_seconds: float
    onboard_passengers: float
    nonholding_seconds: float | None


class ActionResponseMemory:
    """Causal command history; only observed event clocks close intervals.

    Call ``observe_event`` before scoring the next issue, save its features,
    then ``register`` after choosing the executed action.  The memory contains
    all observed commands; fitting still uses only the selected feedback packet.
    """

    def __init__(self, line_names=LINE_NAMES, max_hold_seconds=60.):
        self.line_names = tuple(line_names)
        self.max_hold_seconds = float(max_hold_seconds)
        self.pending: dict[tuple[str, str, str], PendingCommand] = {}

    @staticmethod
    def _key(event):
        return str(event["line_id"]), str(event["vehicle_id"]), str(event["trip_id"])

    def observe_event(self, event):
        """Close the earlier interval at its actual observed vehicle endpoint.

        A recycled fleet vehicle starting a new trip also closes stale memory;
        it never turns that new trip into a previous-trip physical label.
        """
        line, vehicle, _trip = self._key(event)
        for key in list(self.pending):
            if key[:2] == (line, vehicle):
                del self.pending[key]

    def register(self, issued, *, hold_seconds, onboard_passengers,
                 nonholding_seconds=None):
        line, vehicle, trip = self._key(issued)
        if line not in self.line_names:
            return
        hold, onboard = float(hold_seconds), float(onboard_passengers)
        if not 0 <= hold <= self.max_hold_seconds or onboard < 0:
            raise ValueError("Use the executed holding command and observed onboard count")
        nonholding = None if nonholding_seconds is None else max(0., float(nonholding_seconds))
        self.observe_event(issued)
        self.pending[(line, vehicle, trip)] = PendingCommand(
            line, vehicle, trip, float(issued["started_at"]), hold, onboard, nonholding)

    def features(self, now):
        """18 normalized known-clock covariates, independent of a new action.

        Count scale=16, holding scale=60 s, onboard scale=100 passengers,
        same-vehicle interval scale=300 s.  No future event queue is inspected.
        """
        result = np.zeros((len(self.line_names), len(MEMORY_KINDS)), np.float64)
        for command in self.pending.values():
            elapsed = float(now)-command.issued_at
            if elapsed < 0:
                raise ValueError("A command cannot enter memory before issuance")
            remaining = max(0., command.hold_seconds-elapsed)
            values = result[self.line_names.index(command.line)]
            values[:4] += [1/16, remaining/60, command.onboard_passengers*remaining/6000,
                           elapsed/300]
            if command.nonholding_seconds is not None:
                values[4:] += [max(0., command.hold_seconds+command.nonholding_seconds-elapsed)/300,
                               1/16]
        return result.reshape(-1).astype(np.float32)

    def summary(self, now):
        return dict(pending_commands=len(self.pending),
            pending_by_line={line: sum(value.line == line for value in self.pending.values())
                             for line in self.line_names},
            features=self.features(now).tolist(),
            memory_semantics="issued commands until observed same-vehicle event; holding clock is not actual departure")


def response_features(states, actions, memory_features, nonholding_seconds, numeric_lines):
    """Route-specific holding interactions and saved world-only memory.

    The action is already executed/capped; applying the action transform here
    would execute it twice.  The physical covariate is the available prediction
    saved at issuance, never the subsequently observed interval duration.
    """
    states = torch.as_tensor(states)
    if states.ndim == 1:
        states = states[None]
    actions = torch.as_tensor(actions, dtype=states.dtype, device=states.device).reshape(-1, 1)
    memories = torch.as_tensor(memory_features, dtype=states.dtype, device=states.device)
    if memories.ndim == 1:
        memories = memories[None]
    memories = memories.expand(len(states), -1)
    if torch.is_tensor(nonholding_seconds):
        physical = nonholding_seconds.to(dtype=states.dtype, device=states.device)
    else:
        # None is a genuinely unavailable source covariate, distinct from a
        # measured/predicted zero.  Keep that distinction as its own feature.
        physical = torch.as_tensor(np.asarray(nonholding_seconds, dtype=np.float32),
                                   dtype=states.dtype, device=states.device)
    physical = physical.reshape(-1, 1).expand(len(states), 1)
    available = torch.isfinite(physical).to(states.dtype)
    physical = torch.where(available.bool(), physical, torch.zeros_like(physical))
    slots = states[:, 5:].reshape(-1, 16, 14)
    current = (states[:, 2]*15).round().long()
    local = slots[torch.arange(len(states), device=states.device), current]
    line = local[:, 1].round().long()
    ids = torch.as_tensor(numeric_lines, dtype=torch.long, device=states.device)
    route = (line[:, None] == ids[None]).to(states.dtype)
    route = torch.cat((route, 1-route.sum(1, keepdim=True)), dim=1)
    holding = (actions+1.)/2.
    memory = memories.reshape(len(states), len(ids), len(MEMORY_KINDS))
    count, remaining = memory[:, :, 0].sum(1, keepdim=True), memory[:, :, 1].sum(1, keepdim=True)
    # Missing neighbor headways are normalized to one by the existing contract.
    interactions = torch.cat((physical/300, available, holding, holding.square(), holding*local[:, 8:9],
        holding*local[:, 7:8], holding*local[:, 9:10], holding*(local[:, 3:4]-1),
        holding*(local[:, 4:5]-1), holding*physical/300,
        holding*count, holding*remaining), dim=1)
    return torch.cat((memories, (route[:, :, None]*interactions[:, None]).flatten(1)), dim=1)


def fit_response_readout(features, residuals, *, ridge=1.):
    """Fit four observed global residuals, with a zero prior and no intercept."""
    x, y = features.double(), residuals.double()
    precision = x.T@x + float(ridge)*torch.eye(x.shape[1], dtype=x.dtype, device=x.device)
    return torch.linalg.solve(precision, x.T@y).to(features.dtype)


def decode_event_step(states, actions, output, learned_decode):
    """Exact same-batch transition; learned decoder only at batch boundaries.

    ``learned_decode(states, output)`` returns next_state, reward, duration.  The
    deterministic branch consumes no learned decoder RNG and retains all slots.
    """
    actions = torch.as_tensor(actions, dtype=states.dtype, device=states.device)
    current = (states[:, 2]*15).round().long()
    count = (states[:, 1]*16).round().long()
    within = current+1 < count
    next_states = states.clone()
    rewards, durations = states.new_zeros(len(states)), states.new_zeros(len(states))
    if (~within).any():
        next_states[~within], rewards[~within], durations[~within] = learned_decode(
            states[~within], output[~within])
    rows = torch.nonzero(within).flatten()
    next_states[rows, 2] = (current[rows]+1).to(states.dtype)/15
    next_states[rows, 5+current[rows]*14+13] = actions.reshape(-1)[rows]
    return next_states, rewards, durations
