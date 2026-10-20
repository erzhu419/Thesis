"""Passive issue-time head diagnostics for the unchanged global world models."""

from bootstrap import formal
from learning import losses, row_table, targets
import torch

HEADS = ("global", "local", "count", "line", "station", "neighbor")


class DiagnosticWorld(formal.ScoredWorld):
    @torch.no_grad()
    def observe(self, rows, now):
        details = {}
        for row in rows:
            pending = self.pending.get(row.sequence)
            if pending is None:
                continue
            y = targets(row_table([row]), self.stations, self.device)
            family = pending["family_at_issue"]
            physical = {name: torch.expm1((output[0, :4]*self.norm.g_std+
                        self.norm.g_mean)[:2].clamp(0, 20)).tolist()
                        for name, output in pending["outputs"].items()}
            details[row.sequence] = dict(selected_head_losses=dict(zip(HEADS,
                losses(pending["outputs"][family], y, self.norm)[0].tolist())),
                family_head_losses={name: dict(zip(HEADS,
                    losses(output, y, self.norm)[0].tolist()))
                    for name, output in pending["outputs"].items()},
                family_physical_errors={name: dict(duration_absolute_error=abs(value[0]-row.dt),
                    cost_absolute_error=abs(value[1]+row.r)) for name, value in physical.items()})
        logs = super().observe(rows, now)
        for log in logs:
            log.update(details[log["sequence"]])
        return logs
