"""Read frozen resident critics and factual labels without rebuilding or training."""
from __future__ import annotations

import argparse
from copy import deepcopy
import json
from pathlib import Path
import time

from bootstrap import HERE, write
import numpy as np
import torch

from experiments.mcwm_closed_loop_critic_decomposition_v35_20261024.decomposition import decompose
from experiments.mcwm_closed_loop_critic_decomposition_v35_20261024.coverage import coverage
from experiments.mcwm_closed_loop_critic_shadow_v33_20261024.facts import load_facts
from experiments.mcwm_closed_loop_memory_truth_v34_20261024.frozen import optimizer_steps
from experiments.mcwm_closed_loop_update_attribution_v27_20261023.shadow import _rng, _equal


def parameters(learners):
    names=('actor','critic','target_critic','value','ratio')
    return {arm:dict(modules={n:deepcopy(getattr(l,n).state_dict()) for n in names},
                     alpha=l.log_alpha.detach().clone()) for arm,l in learners.items()}


def counters(learners):
    return {arm:dict(updates=l.total_updates,feedback=l.replay_buffer.target_received,
                     optimizer_steps=optimizer_steps(l)) for arm,l in learners.items()}


def run(args):
    torch.set_num_threads(1)
    protocol=json.loads((HERE/'protocol.json').read_text())
    out=args.out.resolve();out.mkdir(parents=True,exist_ok=False)
    started=time.time()
    try:
        v34=HERE.parent/protocol['v34_directory']
        original=v34/'runs'/'seed6202'/args.scenario
        checkpoint=original/'frozen_models'/'final_models.pt'
        if not checkpoint.is_file():
            raise FileNotFoundError('v35 requires the resident frozen cache; rebuilding is not allowed')
        parent=json.loads((original/'summary.json').read_text())
        if not parent['integrity']['all_pass'] or len(parent['states'])!=6:
            raise RuntimeError('Require all six complete predeclared v34 states')
        payload=torch.load(checkpoint,map_location='cpu',weights_only=False)
        learners=payload['learners'];contract=payload['contract']
        if payload['seed']!=6202 or payload['scenario']!=args.scenario or not payload['proof']['all_pass']:
            raise RuntimeError('Wrong resident frozen models')
        if any(p.requires_grad for l in learners.values() for name in
               ('actor','critic','target_critic','value','ratio') for p in getattr(l,name).parameters()):
            raise RuntimeError('The saved v34 parameters are not frozen')
        before_parameters=parameters(learners);before_counts=counters(learners)
        before_F={a:deepcopy(vars(l.action_transform)) for a,l in learners.items()}
        saved_rng=_rng()
        private_rng={a:{n:deepcopy(getattr(l,n).bit_generator.state) for n in ('rng','diagnostic_rng')}
                     for a,l in learners.items()}
        p33=json.loads((HERE.parent/protocol['v33_directory']/'protocol.json').read_text())
        facts=load_facts(HERE.parents[1]/p33['original_runs'][args.scenario],contract,p33)
        rows=[r for packet in facts['packets'] for r in packet['rows']]
        # Verify coverage describes exactly the admitted rows in the cache.
        for arm,l in learners.items():
            recent=list(l.replay_buffer.recent)
            if len(recent)!=len(rows) or len(rows)!=285:
                raise RuntimeError('Wrong factual row count')
            for row,stored in zip(rows,recent):
                if not all(np.array_equal(getattr(row,k),stored[k]) for k in ('s','a')):
                    raise RuntimeError('Factual coverage rows differ from the frozen replay')
                if arm=='memory' and not np.array_equal(np.asarray(row.memory_s['vector'],np.float32),stored['memory_s']):
                    raise RuntimeError('Factual coverage memory differs from the frozen replay')
        selected=[];state_checks=[]
        with (out/'heads.jsonl').open('x') as handle:
            for ref in parent['states']:
                source=Path(ref['path'])
                record=json.loads(source.read_text())
                # Truth fields are deliberately absent from these diagnostic inputs.
                view={k:record[k] for k in ('slot','selected','actions','holds','prediction')}
                result=decompose(learners,view['selected'],view['actions'],reference=view['prediction'])
                handle.write(json.dumps(dict(slot=view['slot'],source=str(source),
                    sequence=view['selected']['sequence'],issue=view['selected']['started_at'],
                    decomposition=result),separators=(',',':'),allow_nan=False)+'\n')
                state_checks.append(result['checks'])
                selected.append(view)
                print('SCHEDULEURM_PROGRESS '+json.dumps(dict(phase='frozen_heads',
                    current=len(selected),total=6,unit='states',slot=view['slot'])),flush=True)
        action_coverage=coverage(facts,selected,contract)
        write(out/'coverage.json',action_coverage)
        after_counts=counters(learners)
        checks=dict(original_v34_integrity=parent['integrity']['all_pass'],
            frozen_model_rebuild_proof=payload['proof']['all_pass'],
            exact_original_facts=facts['proof']['all_pass'],
            original285_and456=all(v['feedback']==285 and v['updates']==50456 for v in after_counts.values()),
            six_predeclared_states=len(selected)==6 and len({v['slot'] for v in selected})==6,
            all_state_checks=all(all(c.values()) for c in state_checks),
            all_weights_unchanged=_equal(before_parameters,parameters(learners)),
            counters_and_optimizer_steps_unchanged=before_counts==after_counts,
            global_rng_unchanged=_equal(saved_rng,_rng()),
            private_rng_unchanged=all(_equal(private_rng[a][n],getattr(l,n).bit_generator.state)
                for a,l in learners.items() for n in ('rng','diagnostic_rng')),
            F_unchanged=before_F=={a:vars(l.action_transform) for a,l in learners.items()},
            no_world_or_trust=all(l.world is None for l in learners.values()))
        summary=dict(status='complete' if all(checks.values()) else 'integrity_failed',
            integrity=dict(all_pass=all(checks.values()),checks=checks),scenario=args.scenario,
            input_checkpoint=str(checkpoint),reference_v34=str(original),
            counters=dict(before=before_counts,after=after_counts),states=len(selected),
            facts=facts['metadata'],budget=protocol['compute_budget'],
            wall=dict(started=started,finished=time.time()))
        write(out/'summary.json',summary)
        if not all(checks.values()):raise RuntimeError('Frozen critic diagnostic integrity failed')
    except Exception as error:
        write(out/'failure.json',dict(type=type(error).__name__,message=str(error)))
        raise


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--scenario',choices=('stable','local_speed_drop'),required=True)
    parser.add_argument('--out',type=Path,required=True)
    run(parser.parse_args())
