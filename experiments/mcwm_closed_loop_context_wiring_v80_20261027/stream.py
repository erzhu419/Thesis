"""Bind causal critic context without changing the original event/action loop."""
from copy import deepcopy
import sys

from experiments.mcwm_closed_loop_v1.stream import EventStream, EI
from experiments.mcwm_closed_loop_capacity_context_v78_20261027.context import observe_context


class NativeContextReader:
    """Read the current native event only, before the event batch is applied."""

    def __init__(self, env):
        self.env = env

    def __call__(self, issued):
        # AllLinesSumoEnv creates its bridge lazily in reset, after reader setup.
        bridge = self.env._bridge
        api = sys.modules[type(bridge).__module__].traci
        events = [ev for ev in self.env._current_events
            if (str(ev.bus_id), str(ev.line_id), str(ev.stop_id)) ==
               (str(issued['trip_id']), str(issued['line_id']), str(issued['station_id']))]
        if len(events) != 1:
            raise RuntimeError('Issue does not identify one current native event')
        ev = events[0]
        return observe_context(api, trip_id=ev.bus_id, line_id=ev.line_id,
            station_index=ev.stop_idx,
            route_stops=bridge.line_obj_dic[ev.line_id].stop_id_l,
            issue_clock=float(issued['started_at']), event_clock=float(ev.sim_time))


def issue_binding(issued):
    return dict(episode_id=issued['episode_id'], sequence=issued['sequence'],
        trip_id=issued['trip_id'], vehicle_id=issued['vehicle_id'],
        line_id=issued['line_id'], station_id=issued['station_id'],
        event_clock=float(issued['event'][EI['sim_time_seconds']]),
        issue_clock=float(issued['started_at']))


class ContextEventStream(EventStream):
    """Attach ctx_s at issue and ctx_ns at the next chronological issue.

    The next endpoint may belong to another bus and may have zero elapsed time.
    All original transition fields, physical intervals, F, thinning and feedback
    delay are produced by the unchanged parent step/_close implementation.
    """

    def __init__(self, *args, context_reader, on_issue=None, **kwargs):
        self.context_reader, self.user_on_issue = context_reader, on_issue
        self._context_waiting = None
        super().__init__(*args, on_issue=self._bind_issue, **kwargs)

    def reset(self):
        self._context_waiting = None
        super().reset()

    def _bind_issue(self, issued):
        context = deepcopy(self.context_reader(issued))
        clocks = context['clocks']
        now = float(issued['started_at'])
        # A failed native getter is a missing observation, never an observed zero.
        # A non-atomic read is rejected rather than attached to an earlier state.
        if (clocks['issue'] != now or clocks['obs_start'] != now
                or clocks['obs_end'] != now):
            raise RuntimeError('Context was not sampled atomically at its issue')
        if clocks['event'] != float(issued['event'][EI['sim_time_seconds']]):
            raise RuntimeError('Context event time differs from its original event')
        context['binding'] = issue_binding(issued)
        issued['ctx_s'] = context
        row = self._context_waiting
        if row is not None:
            if (row.completed_at != now or row.next_vehicle_id != issued['vehicle_id']
                    or row.episode_id != issued['episode_id']):
                raise RuntimeError('Chronological next context differs from transition endpoint')
            row.ctx_ns = context
            self._context_waiting = None
        if self.user_on_issue is not None:
            self.user_on_issue(issued)

    def _close(self, raw_ns, snapshot, next_vehicle_id='', terminal=False):
        pending, count = self._pending, len(self._queue)
        super()._close(raw_ns, snapshot, next_vehicle_id, terminal)
        if len(self._queue) == count:
            return
        row = self._queue[-1]
        row.ctx_s = pending['ctx_s']
        if terminal:
            row.ctx_ns = dict(observed=False, z=None, reasons=['absorbing_terminal'],
                binding=dict(kind='absorbing_terminal', episode_id=row.episode_id,
                    issue_clock=row.completed_at))
        else:
            self._context_waiting = row

    def release(self, now=None):
        if self._context_waiting is not None:
            raise RuntimeError('Cannot release an incomplete context endpoint')
        return super().release(now)
