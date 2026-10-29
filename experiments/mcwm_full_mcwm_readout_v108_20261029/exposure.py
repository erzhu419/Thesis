"""Reconstruct v107 target-row exposure using only its private NumPy RNG."""
import numpy as np


def reconstruct(p):
    """Return deterministic sampling counts, not gradients or control effects."""
    frozen = dict(private_seed=606201, feedback_records_per_packet=5,
        policy_updates_per_packet=8, fresh_rows=64, fresh_pool_capacity=285,
        feedback_first_deadline=1200, feedback_last_deadline=18000,
        feedback_interval_seconds=300)
    if any(p[key] != value for key, value in frozen.items()):
        raise ValueError("This readout is fixed to the declared v107 sampler")
    deadlines = list(range(p["feedback_first_deadline"],
        p["feedback_last_deadline"] + 1, p["feedback_interval_seconds"]))
    packets, rows_per_packet = len(deadlines), p["feedback_records_per_packet"]
    calls_per_packet, rows_per_call = p["policy_updates_per_packet"], p["fresh_rows"]
    draws_per_packet = calls_per_packet * rows_per_call
    early_packets = sum(deadline <= 9000 for deadline in deadlines)
    early_rows = early_packets * rows_per_packet
    counts = np.zeros(packets * rows_per_packet, dtype=np.int64)
    phase_cohorts = np.zeros((2, 2), dtype=np.int64)
    rng = np.random.default_rng(p["private_seed"])
    first_packet_counts = None
    for packet_index in range(packets):
        admitted = (packet_index + 1) * rows_per_packet
        pool_size = min(admitted, p["fresh_pool_capacity"])
        first_pool_row = admitted - pool_size
        for _ in range(calls_per_packet):
            selected = first_pool_row + rng.integers(0, pool_size, size=rows_per_call)
            counts += np.bincount(selected, minlength=len(counts))
            phase = int(packet_index >= early_packets)
            phase_cohorts[phase, 0] += np.count_nonzero(selected < early_rows)
            phase_cohorts[phase, 1] += np.count_nonzero(selected >= early_rows)
        if packet_index == 0:
            first_packet_counts = counts[:rows_per_packet].tolist()
    zero_rows = np.flatnonzero(counts == 0).tolist()
    later_draws = (packets - early_packets) * draws_per_packet
    groups = []
    for first in range(0, packets, 6):
        last = min(first + 6, packets)
        group = counts[first * rows_per_packet:last * rows_per_packet]
        groups.append(dict(admission_packets=[first + 1, last], rows=len(group),
            total_draws=int(group.sum()), draws_per_row_mean=float(group.mean())))
    return dict(
        interpretation="Uniform sampling with replacement from the currently admitted pool. A row can be drawn repeatedly or never drawn; admission is not a guaranteed training use. These are exposure counts, not Bellman targets or causal effect estimates.",
        row_indexing="Zero-based row indices in packet admission order; packet numbers are one-based.",
        private_seed=p["private_seed"], packets=packets,
        calls=packets * calls_per_packet, rows=len(counts),
        total_draws=int(counts.sum()),
        row_frequency=dict(min=int(counts.min()), max=int(counts.max()), mean=float(counts.mean())),
        zero_used=[dict(row_index=index, packet=index // rows_per_packet + 1,
            deadline=deadlines[index // rows_per_packet]) for index in zero_rows],
        first_packet=dict(row_indices=list(range(rows_per_packet)),
            actual_counts_during_first_packet=first_packet_counts,
            actual_counts_over_full_run=counts[:rows_per_packet].tolist(),
            expected_per_row_during_first_packet=draws_per_packet / rows_per_packet,
            expected_per_row_over_full_run=sum(draws_per_packet / ((i + 1) * rows_per_packet)
                                               for i in range(packets))),
        last_packet=dict(row_indices=list(range(len(counts) - rows_per_packet, len(counts))),
            actual_counts=counts[-rows_per_packet:].tolist(),
            expected_per_row=draws_per_packet / len(counts)),
        first27_in_last30=dict(first_packets=early_packets, first_rows=early_rows,
            first_cohort_last_deadline=9000, later_packets=packets - early_packets,
            later_total_draws=later_draws, actual_draws=int(phase_cohorts[1, 0]),
            fraction=float(phase_cohorts[1, 0] / later_draws)),
        phase_by_admission_cohort=dict(
            first27_packets=dict(first27_rows=int(phase_cohorts[0, 0]),
                                 last30_rows=int(phase_cohorts[0, 1])),
            last30_packets=dict(first27_rows=int(phase_cohorts[1, 0]),
                                last30_rows=int(phase_cohorts[1, 1]))),
        admission_packet_groups=groups,
        row_counts=counts.tolist())
