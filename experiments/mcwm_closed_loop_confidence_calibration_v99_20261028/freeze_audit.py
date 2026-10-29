"""Bind independent C queries to the fixed T fit and the frozen references."""
import argparse
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
T = HERE.parent/'mcwm_closed_loop_collection_domain_v97_20261028'


def audit():
    read = lambda path: json.loads(path.read_text())
    p = read(HERE/'protocol.json')
    source = read(HERE/'source_protocol.json')
    original = read(HERE/'source_model.json')
    prior = read(T/'freeze_audit.json')
    fitted = read(HERE/'source_fit.json')
    fitted_summary = read(HERE/'source_fit_summary.json')
    bias = fitted['bias_model']
    checks = dict(
        prior_reference_audit=prior['all_pass'] and all(prior['checks'].values()),
        original_model_exact=original == read(T/'source_model.json'),
        original_protocol_exact=source == read(T/'source_protocol.json'),
        original_references=p['frozen_reference'] == source['frozen_reference'] == original['reference_contract'],
        original_initializer=p['source_initializer'] == 6207,
        original_C_seeds=p['external_seeds'] == source['independent_cohorts']['C']
            == [178102402,1119764361,1386094634,536907412],
        original_scenes=p['scenarios'] == ['stable','local_speed_drop'],
        fixed_T_fit=fitted['integrity']['all_pass'] and all(fitted['integrity']['checks'].values())
            and fitted['budget']['real_bias_fits'] == 1 and bias['n'] == 12
            and bias['rank'] == 2 and bias['ridge'] == 1.,
        T_ready=fitted_summary['status'] == 'complete'
            and fitted_summary['decision']['ready_for_calibration'] is True,
        same_primary=fitted['primary_model'] == 'service_history6',
        same_scaling=bias['q_range'] == p['q_range_for_scaling_only'],
        same_candidate_domain=fitted['candidate_deployment_domain'] == p['candidate_deployment_domain'],
        query_domain_frozen=p['collection_domain']['q_empirical_range_is_gate'] is False,
        C_budget=p['budget']['total_cap'] == 676800 and p['budget']['expected_collectors'] == 8
            and p['budget']['expected_pairs'] == 24 and p['budget']['expected_physical_labels'] == 48,
        fixed_calibration=p['calibration']['alpha'] == .2 and p['calibration']['blocks'] == 4,
        no_bias_refit=p['budget']['fit_count'] == 0,
        no_neural_learning=all(p['budget'][key] == 0 for key in ('initializer_updates','actor_updates',
            'critic_updates','V_updates','alpha_updates','world_updates','trust_updates')),
        V_sealed=p['calibration']['V'] == source['independent_cohorts']['V'] == [5696843])
    return dict(prior,all_pass=all(checks.values()),checks=checks,
        external_seeds=p['external_seeds'],source_model_path=str(HERE/'source_model.json'),
        source_protocol_path=str(HERE/'source_protocol.json'),source_fit_path=str(HERE/'source_fit.json'),
        reused_actor_audit=str(T/'freeze_audit.json'),new_reference_roots=0,
        new_fits=0,new_neural_updates=0,new_suffixes=0,new_labels=0,v87_truth_used=0,native_seconds=0)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out',type=Path,required=True)
    args=parser.parse_args()
    result=audit()
    raw=json.dumps(result,ensure_ascii=False,separators=(',',':'),allow_nan=False)+'\n'
    args.out.parent.mkdir(parents=True,exist_ok=True)
    with args.out.open('x') as handle:handle.write(raw)
    print(raw,end='',flush=True)
    if not result['all_pass']:raise RuntimeError('Frozen T/C contract failed; C not started')


if __name__=='__main__':main()
