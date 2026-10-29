"""Resident frozen v81 action/gradient decomposition; no native or update work."""
import argparse
from copy import deepcopy
from dataclasses import asdict
import json
from pathlib import Path
import sys
import time

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parents[1]))

import numpy as np
import torch

from experiments.mcwm_closed_loop_context_wiring_v80_20261027.learning import make_learner
from experiments.mcwm_closed_loop_update_attribution_v27_20261023.shadow import _equal,_rng,_state
from experiments.mcwm_closed_loop_capacity_decomposition_v82_20261027.data import (
    load_run,exact_outcome_exposures)
from experiments.mcwm_closed_loop_capacity_decomposition_v82_20261027.decompose import (
    executed_grid,q_decomposition,actor_gradient_decomposition,audit_saved_gradients)
from experiments.mcwm_closed_loop_capacity_decomposition_v82_20261027.pressure import pressure


def common_bank(runs,tail_window,batch_seconds):
    """Union by native provenance; never merge equal-valued states across arms."""
    selected=[]
    for arm in ('fresh','no_fresh'):
        run=runs[arm]
        selected.extend(dict(row,bank='covariate',origin_arm=arm) for row in run['covariates'])
        seen=set()
        for row in run['issues']:
            clock=row['started_at']
            if not tail_window[0] <= clock < tail_window[1]:
                continue
            key=(row['line_id'],int((clock-tail_window[0])//batch_seconds))
            if key not in seen:
                selected.append(dict(row,bank='tail',origin_arm=arm))
                seen.add(key)
    identities=[(row['origin_arm'],row['sequence']) for row in selected]
    if len(set(identities)) != len(identities):
        raise ValueError('Selected state-bank provenance is duplicated')
    metadata=[{key:row[key] for key in ('origin_arm','sequence','line_id','started_at','bank')}
              for row in selected]
    return selected,metadata


def load_frozen(directory,seed,arm):
    learner,replay,contract,_,initializer=make_learner(seed,arm=arm)
    payload=torch.load(Path(directory)/'final_learner.pt',map_location='cpu',weights_only=False)
    state=payload['state']
    if state['updates'] != 50456 or state['online_updates'] != 456:
        raise ValueError('Frozen model does not contain the original456 response updates')
    if asdict(learner.config) != payload['config']:
        raise ValueError('Frozen policy configuration differs from its source initializer')
    for name,values in state['modules'].items():
        module=getattr(learner,name)
        module.load_state_dict(values,strict=True)
        module.eval()
        module.requires_grad_(False)
        if not _equal(values,module.state_dict()):
            raise RuntimeError('Frozen module did not load exactly: '+name)
    with torch.no_grad():
        learner.log_alpha.copy_(state['log_alpha'])
    learner.log_alpha.requires_grad_(False)
    learner.total_updates=state['updates']
    learner.online_critic_updates=state['online_updates']
    if replay.target_received or replay.recent or replay.fresh_outcomes:
        raise ValueError('Frozen diagnostic unexpectedly admitted feedback')
    return learner,contract,payload,initializer


def scalar_deployment(learner,bank):
    proposals=[]
    executed=[]
    for row in bank:
        proposal=learner.act(row['s'][None],deterministic=True)[0]
        proposals.append(proposal)
        executed.append(learner.action_transform.numpy(row['s'],proposal))
    return np.asarray(proposals,np.float32),np.asarray(executed,np.float32)


def finite(value):
    if isinstance(value,torch.Tensor):
        return not value.is_floating_point() or bool(torch.isfinite(value).all())
    if isinstance(value,dict):
        return all(finite(item) for item in value.values())
    if isinstance(value,(list,tuple)):
        return all(finite(item) for item in value)
    return True


def arrays(value,prefix=''):
    """Numeric payload only; original state/log provenance stays resident."""
    if isinstance(value,dict):
        result={}
        for key,item in value.items():
            result.update(arrays(item,prefix+str(key)+'__'))
        return result
    if isinstance(value,torch.Tensor):
        return {prefix[:-2]:value.detach().cpu().numpy()}
    if isinstance(value,np.ndarray):
        return {prefix[:-2]:value}
    return {}


def save_arrays(path,payload,metadata):
    values=arrays(payload)
    values.update(meta_sequence=np.asarray([r['sequence'] for r in metadata],np.int64),
        meta_origin_arm=np.asarray([r.get('origin_arm','fresh') for r in metadata]),
        meta_line_id=np.asarray([r['line_id'] for r in metadata]),
        meta_started_at=np.asarray([r['started_at'] for r in metadata],np.float64))
    np.savez_compressed(path,**values)


def _checks(q,gdet,gm8):
    checks=dict(scores_finite=finite(q),deterministic_gradients_finite=finite(gdet),
        M8_gradients_finite=finite(gm8))
    for key in ('head_reconstruction_error','min_reconstruction_error','advantage_reconstruction_error'):
        checks[key]=bool((q[key] <= q['roundoff_bound']).all())
    for name,g in (('det',gdet),('M8',gm8)):
        numerical=audit_saved_gradients(g['q'],g['gate_probability'],g['gradients'])
        checks.update({name+'_'+key:value for key,value in numerical['checks'].items()})
    return checks


def run_seed(seed,out,protocol):
    from experiments.mcwm_closed_loop_capacity_decomposition_v82_20261027.aggregate import (
        summarize_q,summarize_grad,summarize_pressure)
    out=Path(out)
    out.mkdir(parents=True,exist_ok=True)
    if any(out.iterdir()):
        raise FileExistsError('Frozen diagnostic output already exists: '+str(out))
    started=time.monotonic()
    source=HERE.parent/protocol['source_directory']
    noise=np.asarray(protocol['noise8'],np.float32)
    theta_reference={f"{row['seed']}/{row['scenario']}/{row['arm']}":row['theta']
                     for row in protocol['final_theta_reference']}
    cells=[]
    all_checks={}
    for scene in protocol['scenarios']:
        base=source/'runs/executed_cql/development'/f'seed{seed}'/scene
        learners,contracts,payloads,initializers={},{},{},{}
        for arm in ('fresh','no_fresh'):
            learners[arm],contracts[arm],payloads[arm],initializers[arm]=load_frozen(base/arm,seed,arm)
        # Constructors restore initializer RNG. Only the completed frozen pair
        # marks the start of the no-random-draw diagnostic interval.
        rng_before=_rng()
        before={arm:deepcopy(_state(model)) for arm,model in learners.items()}
        runs={arm:load_run(base/arm,contracts[arm]) for arm in ('fresh','no_fresh')}
        bank,metadata=common_bank(runs,protocol['tail_window_seconds'],protocol['batch_seconds'])
        states=torch.as_tensor(np.stack([r['s'] for r in bank]),dtype=torch.float32)
        context=torch.as_tensor(np.stack([r['context'] for r in bank]),dtype=torch.float32)
        proposal_actions={arm:scalar_deployment(model,bank) for arm,model in learners.items()}
        grid=executed_grid(learners['fresh'].action_transform,states)
        other_grid=executed_grid(learners['no_fresh'].action_transform,states)
        common=torch.cat((grid['executed_actions'],
            torch.as_tensor(np.stack([r['a'] for r in bank]),dtype=torch.float32),
            torch.as_tensor(proposal_actions['fresh'][1],dtype=torch.float32),
            torch.as_tensor(proposal_actions['no_fresh'][1],dtype=torch.float32)),1)
        pair_checks=dict(same_F_grid=torch.equal(grid['executed_actions'],other_grid['executed_actions']),
            eight_common_candidates=common.shape==(len(bank),8),
            exact_two_covariate_pools=sum(r['bank']=='covariate' for r in bank)==570,
            same_frozen_initializer=str(initializers['fresh'])==str(initializers['no_fresh']))
        exposure=exact_outcome_exposures(runs['fresh']['outcomes'],seed)
        pair_checks['private_exposure_rng_exact']=_equal(exposure['final_rng_state'],payloads['fresh']['outcome_rng'])
        pair_checks['private_exposure_draw_count']=exposure['total_draws']==29184 and exposure['calls']==456
        for arm,model in learners.items():
            own_tail=[i for i,row in enumerate(bank) if row['bank']=='tail' and row['origin_arm']==arm]
            checks=dict(pair_checks,own_tail_exists=bool(own_tail))
            proposals,executed=proposal_actions[arm]
            checks['own_frozen_tail_proposals_exact']=all(np.array_equal(proposals[i],bank[i]['proposed_a']) for i in own_tail)
            checks['own_frozen_tail_executions_exact']=all(np.array_equal(executed[i],bank[i]['a']) for i in own_tail)
            tail_reproduction=dict(n=len(own_tail),
                maximum_proposal_error=max((float(np.max(np.abs(proposals[i]-bank[i]['proposed_a']))) for i in own_tail),default=None),
                maximum_executed_action_error=max((float(np.max(np.abs(executed[i]-bank[i]['a']))) for i in own_tail),default=None))
            mismatches=[i for i in own_tail if not (np.array_equal(proposals[i],bank[i]['proposed_a'])
                and np.array_equal(executed[i],bank[i]['a']))]
            if mismatches:
                i=mismatches[0]
                tail_reproduction['first_difference']=dict(metadata[i],
                    frozen_proposal=float(proposals[i,0]),logged_proposal=float(bank[i]['proposed_a'][0]),
                    frozen_action=float(executed[i,0]),logged_action=float(bank[i]['a'][0]))
            reference=theta_reference[f'{seed}/{scene}/{arm}']
            checks['frozen_theta_matches_previous_readout']=np.array_equal(
                model.critic.action_weight.detach().numpy(),np.asarray(reference,np.float32))
            with torch.no_grad():
                q=q_decomposition(model.critic,states,context,common)
            q['executed_actions']=common
            det=actor_gradient_decomposition(model,states,context)
            m8=actor_gradient_decomposition(model,states,context,noise=noise)
            checks.update(_checks(q,det,m8))
            cell=dict(seed=seed,scenario=scene,model_arm=arm,
                state_bank=dict(total=len(bank),covariates=570,tail=sum(r['bank']=='tail' for r in bank),
                    own_tail_reproduced=len(own_tail),provenance='origin_arm,sequence; cross-arm identities never merged'),
                q=summarize_q(q,metadata),gradient_det=summarize_grad(det,metadata),
                gradient_m8=summarize_grad(m8,metadata),td_pressure=None,
                own_tail_reproduction=tail_reproduction,
                final_theta=[float(v) for v in model.critic.action_weight.detach()],
                checks=checks,new_optimizer_steps=0,new_simulation_seconds=0)
            stem=out/f'{scene}_{arm}'
            save_arrays(stem.with_name(stem.name+'_q.npz'),q,metadata)
            save_arrays(stem.with_name(stem.name+'_gradient_det.npz'),det,metadata)
            save_arrays(stem.with_name(stem.name+'_gradient_m8.npz'),m8,metadata)
            if arm=='fresh':
                rows=runs[arm]['outcomes']
                p=pressure(model,rows,noise,weights=exposure['weights'])
                pmetadata=[dict(line_id=r['line_id'],started_at=r['started_at'],sequence=r['sequence'],
                    bank='outcome',origin_arm=arm) for r in rows]
                cell['td_pressure']=summarize_pressure(p,pmetadata,exposure['weights'])
                pscale=max(1.,float(p['q'].abs().max()),float(p['target'].abs().max()))
                checks['pressure_finite']=finite(p)
                checks['pressure_error_reconstruction']=float(p['error_reconstruction_error']) <= 16*torch.finfo(p['q'].dtype).eps*pscale
                save_arrays(stem.with_name(stem.name+'_pressure.npz'),p,pmetadata)
                np.savez_compressed(stem.with_name(stem.name+'_exposure.npz'),
                    weights=exposure['weights'],sequence_ids=exposure['sequence_ids'])
            cells.append(cell)
        pair_checks['scoring_all_rng_unchanged']=_equal(rng_before,_rng())
        for arm,model in learners.items():
            pair_checks[arm+'_frozen_model_Adam_counts_unchanged']=_equal(before[arm],_state(model))
            pair_checks[arm+'_no_parameter_grad_accumulation']=all(p.grad is None
                for name in ('actor','critic','target_critic','value') for p in getattr(model,name).parameters())
            pair_checks[arm+'_no_target_feedback_admitted']=model.replay_buffer.target_received==0
        all_checks.update({scene+'/'+key:value for key,value in pair_checks.items()})
        for cell in cells[-2:]:
            cell['checks'].update(pair_checks)
            all_checks.update({scene+'/'+cell['model_arm']+'/'+key:value for key,value in cell['checks'].items()})
        print(json.dumps(dict(seed=seed,scenario=scene,all_pass=all(pair_checks.values()),
            states=len(bank),new_optimizer_steps=0,new_simulation_seconds=0)),flush=True)
    result=dict(status='complete' if all(all_checks.values()) else 'integrity_failed',
        all_pass=all(all_checks.values()),seed=seed,checks=all_checks,cells=cells,
        wall_seconds=time.monotonic()-started,new_optimizer_steps=0,new_simulation_seconds=0,
        scope='Frozen final-parameter diagnostic; not historical gradients or control intervention',
        arrays_remain_resident=True)
    raw=json.dumps(result,ensure_ascii=False,allow_nan=False,separators=(',',':'))+'\n'
    if len(raw.encode()) > 48000:
        raise ValueError('Planned per-seed summary exceeds48KB: '+str(len(raw.encode())))
    with (out/'readout.json').open('x') as target:
        target.write(raw)
    if not result['all_pass']:
        raise RuntimeError('Frozen decomposition integrity failed; see resident readout')
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--seed',type=int,required=True)
    parser.add_argument('--out',type=Path,required=True)
    args=parser.parse_args()
    protocol=json.loads((HERE/'protocol.json').read_text())
    if args.seed not in protocol['seeds']:
        raise ValueError('Seed outside frozen diagnostic cohort')
    run_seed(args.seed,args.out,protocol)


if __name__=='__main__':
    main()
