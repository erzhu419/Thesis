"""Bind the revised query contract to the already-audited frozen inputs.

The actors and twelve old actor/F forwards were validated in v96. Reuse that
evidence and compare small source documents, without loading checkpoints.
"""
import argparse
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
OLD = HERE.parent/'mcwm_closed_loop_confidence_collection_v96_20261028'


def audit():
    read = lambda path: json.loads(path.read_text())
    p = read(HERE/'protocol.json')
    source = read(HERE/'source_protocol.json')
    model = read(HERE/'source_model.json')
    prior = read(HERE/'prior_freeze_audit.json')
    checks = dict(
        prior_actor_audit=prior['all_pass'] and all(prior['checks'].values()),
        prior_audit_exact=prior == read(OLD/'freeze_audit.json'),
        source_protocol_exact=source == read(OLD/'source_protocol.json'),
        source_model_exact=model == read(OLD/'source_model.json'),
        original_model_identity=p['frozen_common_support'] == model['common_support'],
        frozen_reference=p['frozen_reference'] == source['frozen_reference'] == model['reference_contract'],
        frozen_T_seeds=p['external_seeds'] == source['independent_cohorts']['T'] == [309177151,1476154684],
        frozen_scenes=p['scenarios'] == source['frozen_reference']['scenarios'],
        frozen_scaling=p['q_range_for_scaling_only'] == source['q_range_for_scaling_only'],
        empirical_range_not_query_gate=p['collection_domain']['q_empirical_range_is_gate'] is False,
        candidate_domain_predeclared=p['candidate_deployment_domain']['frozen_before_new_labels'] is True,
        unchanged_T_budget=p['budget']['total_cap'] == 338400 and p['budget']['expected_pairs'] == 12
            and p['budget']['expected_physical_labels'] == 24,
        failed_v96_spend_retained=p['budget']['prior_v96_failed_collection_seconds'] == 12600,
        endpoint=p['native_contract']['true_endpoint'] == 24000
            and p['native_contract']['label_available_at'] == 24030,
        no_learning=p['budget']['fit_count'] == p['budget']['calibration_validation_queries'] == 0
            and all(p['budget'][key] == 0 for key in ('initializer_updates','actor_updates',
                'critic_updates','V_updates','alpha_updates','world_updates','trust_updates')),
        unchanged_execution=p['native_contract']['F'] == 'unchanged/applied once'
            and p['native_contract']['trust'] == 'unchanged')
    return dict(prior, all_pass=all(checks.values()), checks=checks,
        source_model_path=str(HERE/'source_model.json'), source_protocol_path=str(HERE/'source_protocol.json'),
        reused_actor_audit=str(OLD/'freeze_audit.json'), reused_reference_roots=prior['reference_roots'],
        new_reference_roots=0, reference_roots=prior['reference_roots'],
        new_fits=0, new_neural_updates=0, new_suffixes=0, new_labels=0,
        v87_truth_used=0, native_seconds=0)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    result = audit()
    raw = json.dumps(result,ensure_ascii=False,separators=(',',':'),allow_nan=False)+'\n'
    args.out.parent.mkdir(parents=True,exist_ok=True)
    with args.out.open('x') as handle:
        handle.write(raw)
    print(raw,end='',flush=True)
    if not result['all_pass']:
        raise RuntimeError('Frozen source identity or new collection contract failed')


if __name__ == '__main__':
    main()
