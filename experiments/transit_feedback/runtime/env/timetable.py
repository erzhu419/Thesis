class Timetable(object):

    def __init__(self, launch_time, launch_turn, direction, target_headway=360.0):
        self.baseline_launch_time = float(launch_time)
        self.launch_time = float(launch_time)
        self.planned_launch_time = float(launch_time)
        self.actual_launch_time = None
        self.direction = direction
        self.launch_turn = launch_turn
        self.launched = False
        # Written by upper policy before dispatch; default 360s (6 min)
        self.target_headway = float(target_headway)
        self.planned_headway = float(target_headway)
        self.dispatch_lateness = 0.0
        self.planned_shift = 0.0
        self._upper_queried = False
