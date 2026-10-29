"""Independent precision audit of saved arrays, without rescoring a model."""
import argparse
import json
from pathlib import Path
import sys

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parents[1]))
import numpy as np
import torch
from experiments.mcwm_closed_loop_capacity_decomposition_v82_20261027.decompose import audit_saved_gradients


def audit(root=HERE):
    rows=[]
    for seed in (6207,6208):
        directory=root/'runs'/f'seed{seed}'
        original=json.loads((directory/'readout.json').read_text())
        other_failures=[key for key,value in original['checks'].items() if not value and
            not any(key.endswith('/'+mode+'_'+kind) for mode in ('det','M8')
                for kind in ('gradient_reconstruction','SAC_reconstruction'))]
        for cell in original['cells']:
            stem=f"{cell['scenario']}_{cell['model_arm']}"
            modes={}
            for mode in ('det','m8'):
                with np.load(directory/(stem+'_gradient_'+mode+'.npz')) as archive:
                    q={key:torch.as_tensor(archive['q__'+key]) for key in
                        ('base_min','full_min','direct','head_switch')}
                    gradients={key:torch.as_tensor(archive['gradients__'+key]) for key in
                        ('base_Q','direct_capacity','head_switch','full_Q','entropy','full_SAC')}
                    result=audit_saved_gradients(q,torch.as_tensor(archive['gate_probability']),gradients)
                modes[mode]=dict(all_pass=result['all_pass'],checks=result['checks'],
                    maximum_Q_closure_error=[float(x) for x in result['Q_closure_error'].abs().amax(0)],
                    maximum_SAC_closure_error=[float(x) for x in result['SAC_closure_error'].abs().amax(0)],
                    maximum_independent_gate_error={key:float(value.abs().max())
                        for key,value in result['gate_reference_errors'].items()},
                    maximum_gate_bound=float(result['Q_gate_closure_bound'].max()),
                    maximum_mean_logstd_bound=float(result['mean_logstd_bound'].max()))
            with np.load(directory/(stem+'_q.npz')) as archive:
                base=archive['base_heads'].astype(np.float64)
                exact_full=(base+archive['residual_heads'].astype(np.float64)).min(0)
                unique=archive['unique'].astype(bool)
                base_idx=np.where(unique,base.min(0),-np.inf).argmax(1)
                full_idx=np.where(unique,exact_full,-np.inf).argmax(1)
                actions=archive['executed_actions']
                axis=np.arange(len(actions))
                changed=actions[axis,base_idx]!=actions[axis,full_idx]
                native=archive['preference_changed'].astype(bool)
            rows.append(dict(seed=seed,scenario=cell['scenario'],arm=cell['model_arm'],
                non_precision_failures=other_failures,modes=modes,
                exact_addition_rank_changed=int(changed.sum()),native_rank_changed=int(native.sum()),
                rank_change_classification_diff_due_to_addition_roundoff=int((changed!=native).sum())))
    passed=all(not row['non_precision_failures'] and all(mode['all_pass'] for mode in row['modes'].values())
        for row in rows)
    return dict(all_pass=passed,cells=rows,models_rescored=0,new_optimizer_steps=0,new_simulation_seconds=0,
        initial_results_preserved=True,
        correction='Original gradient tolerance used the small post-cancellation gradient scale. '
            'Independent FP64 gate formulas now use unreduced reverse-term magnitudes; '
            'mean/logstd retain the original strict tolerance.',
        scope='Precision audit only, not a policy or control intervention')


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out',type=Path,required=True)
    args=parser.parse_args()
    value=audit()
    raw=json.dumps(value,separators=(',',':'),allow_nan=False)+'\n'
    if len(raw.encode())>48000:
        raise ValueError('Precision summary exceeds48KB')
    args.out.parent.mkdir(parents=True,exist_ok=True)
    with args.out.open('x') as output:
        output.write(raw)
    print(json.dumps(dict(all_pass=value['all_pass'],models_rescored=0,new_optimizer_steps=0)),flush=True)
    if not value['all_pass']:
        raise RuntimeError('Independent saved-array precision audit failed')
