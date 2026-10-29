"""Select latest-five covariates and capacity-covered outcomes independently."""
from experiments.mcwm_closed_loop_capacity_feedback_v72_20261026.selection import (
    CapacityPacketBudget, capacity_risk, selection_checks as capacity_checks,
    summarize_selection as capacity_summary)


class DecoupledPacketBudget(CapacityPacketBudget):
    """Same eligible window, two selections; no additional labels or RNG draws."""

    def take(self, now):
        available = {row.sequence: row for row in self.rows}
        outcomes, _ = super().take(now)
        receipt = self.receipts[-1]
        covariates = [available[seq] for seq in
            receipt['capacity_selection']['original_latest5_sequence_ids']]
        cov_ids = [row.sequence for row in covariates]
        outcome_ids = [row.sequence for row in outcomes]
        cov_set, outcome_set = set(cov_ids), set(outcome_ids)
        receipt.update(covariate_sequence_ids=cov_ids, outcome_sequence_ids=outcome_ids,
            covariate_records=len(cov_ids), outcome_records=len(outcome_ids),
            covariate_max_available_at=max(row.available_at for row in covariates),
            overlap_records=len(cov_set & outcome_set),
            union_records=len(cov_set | outcome_set))
        return covariates, outcomes


def selection_checks(receipts, *, expected_packets=57, expected_records=285):
    checks = capacity_checks(receipts, expected_packets=expected_packets,
        expected_records=expected_records)
    cov_ids, outcome_ids = [], []
    checks.update(covariate_latest5=True, both_pools_five_per_packet=True,
        independent_pool_accounting=True, covariate_availability=True)
    for row in receipts:
        cov, outcomes = row['covariate_sequence_ids'], row['outcome_sequence_ids']
        checks['covariate_latest5'] &= cov == row['capacity_selection']['original_latest5_sequence_ids']
        checks['both_pools_five_per_packet'] &= bool(
            len(cov) == len(outcomes) == row['covariate_records'] == row['outcome_records'] == 5)
        checks['independent_pool_accounting'] &= bool(outcomes == row['sequence_ids']
            and row['overlap_records'] == len(set(cov) & set(outcomes))
            and row['union_records'] == len(set(cov) | set(outcomes))
            and 5 <= row['union_records'] <= 6)
        checks['covariate_availability'] &= bool(
            row['deadline']-300 < row['covariate_max_available_at'] <= row['deadline'])
        cov_ids.extend(cov)
        outcome_ids.extend(outcomes)
    checks.update(unique_covariates=len(set(cov_ids)) == len(cov_ids) == expected_records,
        unique_outcomes=len(set(outcome_ids)) == len(outcome_ids) == expected_records,
        explicit_information_union=len(set(cov_ids) | set(outcome_ids))
            == sum(row['union_records'] for row in receipts),
        bounded_information_union=len(set(cov_ids) | set(outcome_ids)) <= expected_packets*6)
    return checks


def summarize_selection(receipts):
    cov = {seq for row in receipts for seq in row['covariate_sequence_ids']}
    outcomes = {seq for row in receipts for seq in row['outcome_sequence_ids']}
    return dict(capacity_summary(receipts), covariate_unique_records=len(cov),
        outcome_unique_records=len(outcomes), target_record_union=len(cov | outcomes),
        target_record_overlap=len(cov & outcomes),
        target_record_union_bound=6*len(receipts),
        information_contract='285 state/action rows plus285 complete outcome labels; '
            'distinct selected union is reported, at most342 rows per full fresh day.',
        invariant_checks=selection_checks(receipts, expected_packets=len(receipts),
            expected_records=5*len(receipts)))
