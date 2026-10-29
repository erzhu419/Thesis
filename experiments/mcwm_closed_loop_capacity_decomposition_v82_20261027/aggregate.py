"""Small summaries of frozen score/gradient effects; full arrays stay resident."""
import numpy as np
import torch


def array(value):
    return value.detach().cpu().numpy() if isinstance(value,torch.Tensor) else np.asarray(value)


def number(value):
    value=float(value)
    return float(f'{value:.7g}') if np.isfinite(value) else None


def window(clock):
    for label,start,end in (('pre9000',0,9000),('9000_12000',9000,12000),
            ('12000_16200',12000,16200),('16200_18300',16200,18300),
            ('18300_24000',18300,24000)):
        if start <= clock < end:
            return label
    return 'outside'


def groups(metadata):
    result={}
    for name,key in (('line','line_id'),('bank','bank'),('origin','origin_arm')):
        values=[str(row.get(key,'unknown')) for row in metadata]
        result[name]={value:np.flatnonzero(np.asarray(values)==value) for value in sorted(set(values))}
    values=[window(row['started_at']) for row in metadata]
    result['window']={value:np.flatnonzero(np.asarray(values)==value) for value in sorted(set(values))}
    return result


def summarize_q(payload,metadata):
    q={k:array(v) for k,v in payload.items() if isinstance(v,(torch.Tensor,np.ndarray))}
    n=len(metadata)
    choice=q['unique_support_count']>1
    unique=q['unique'].astype(bool)
    unique[:,0]=False
    base,full=q['base_advantage'],q['full_advantage']
    reversals=(base*full<0)&unique
    ties=((base==0)!=(full==0))&unique
    head=(q['head_changed']&q['unique']).any(1)
    delta=full-base
    def stats(ids):
        valid=unique[ids]
        mask=choice[ids]
        values=np.abs(delta[ids][valid])
        b=np.abs(base[ids][valid])
        return dict(n=len(ids),choice=int(mask.sum()),rank_changed=int(q['preference_changed'][ids].sum()),
            strict_reversal_states=int(reversals[ids].any(1).sum()),
            tie_change_states=int(ties[ids].any(1).sum()),head_changed_states=int(head[ids].sum()),
            mean_abs_base_advantage=number(b.mean()) if len(b) else None,
            mean_abs_capacity_advantage=number(np.abs(q['direct_advantage'][ids][valid]).mean()) if len(b) else None,
            mean_abs_head_switch_advantage=number(np.abs(q['head_switch_advantage'][ids][valid]).mean()) if len(b) else None,
            mean_abs_advantage_change=number(values.mean()) if len(values) else None,
            max_abs_advantage_change=number(values.max()) if len(values) else None,
            change_to_base_abs_ratio=number(values.sum()/b.sum()) if len(b) and b.sum()>0 else None,
            max_abs_value_reconstruction_error=number(q['min_reconstruction_error'][ids].max()),
            max_abs_advantage_reconstruction_error=number(q['advantage_reconstruction_error'][ids].max()))
    columns=('n','choice','rank_changed','strict_reversal_states','tie_change_states',
        'head_changed_states','mean_abs_base_advantage','mean_abs_advantage_change',
        'max_abs_advantage_change','change_to_base_abs_ratio')
    compact={kind:{key:[stats(ids)[col] for col in columns] for key,ids in rows.items()}
        for kind,rows in groups(metadata).items()}
    return dict(global_stats=stats(np.arange(n)),group_columns=columns,groups=compact,
        aliases_excluded_from_direction_counts=True,strict_reversal='A_base*A_full<0; tie changes separate',
        preferred_action_is_Q_argmax_not_a_deployed_policy=True)


def summarize_grad(payload,metadata):
    gradients={k:array(v) for k,v in payload['gradients'].items()}
    base=gradients['base_Q']+gradients['entropy']
    full=gradients['full_SAC']
    change=full-base
    def vector(value):
        return [number(x) for x in np.asarray(value).reshape(-1)]
    def stats(ids):
        b,f,d=base[ids],full[ids],change[ids]
        den=np.abs(b).sum(0)
        ratio=np.divide(np.abs(d).sum(0),den,out=np.full(3,np.nan),where=den>0)
        # This is a raw strict sign count; zero/tiny baseline gradients are not
        # assigned an arbitrary replacement denominator.
        return dict(n=len(ids),mean_abs_base_SAC_gradient=vector(np.abs(b).mean(0)),
            mean_abs_full_SAC_gradient=vector(np.abs(f).mean(0)),
            mean_abs_direct_gradient=vector(np.abs(gradients['direct_capacity'][ids]).mean(0)),
            mean_abs_head_switch_gradient=vector(np.abs(gradients['head_switch'][ids]).mean(0)),
            mean_abs_entropy_gradient=vector(np.abs(gradients['entropy'][ids]).mean(0)),
            absolute_change_to_base_ratio=vector(ratio),
            strict_gradient_sign_reversals=(b*f<0).sum(0).astype(int).tolist(),
            max_abs_change=vector(np.abs(d).max(0)))
    columns=('n','absolute_change_to_base_ratio','strict_gradient_sign_reversals')
    grouped=({kind:{key:[stats(ids)[col] for col in columns] for key,ids in rows.items()}
        for kind,rows in groups(metadata).items() if kind in ('line','window','bank')}
        if payload['samples'] > 1 else {})
    return dict(mode=payload['mode'],samples=payload['samples'],coordinates=payload['actor_gradient_coordinates'],
        global_stats=stats(np.arange(len(metadata))),group_columns=columns,groups=grouped,
        max_reconstruction_error=number(array(payload['gradient_reconstruction_error']).max()),
        max_SAC_reconstruction_error=number(array(payload['SAC_gradient_reconstruction_error']).max()),
        scalar_network_parameter_gradient=False,
        critic_ablation_changes_frozen_actor_actions=False)


def summarize_pressure(payload,metadata,weights):
    gradient=array(payload['gradient'])
    weights=np.asarray(weights,dtype=np.float64)
    forced=array(payload['forced_60']).astype(bool)
    single=array(payload['single_support']).astype(bool)
    clipped=array(payload['huber_clipped']).astype(bool)
    n=len(metadata)
    columns=('n','weight_share','signed_theta_partial','absolute_theta_partial','share_of_absolute_pressure')
    def weighted(w, detailed):
        w=w/w.sum()
        denominator=(np.abs(gradient)*w[None]).sum(1)
        def subset(ids):
            contributions=(gradient[:,ids]*w[None,ids]).sum(1)
            absolute=(np.abs(gradient[:,ids])*w[None,ids]).sum(1)
            return dict(n=len(ids),weight_share=number(w[ids].sum()),
                signed_theta_partial=[number(x) for x in contributions],
                absolute_theta_partial=[number(x) for x in absolute],
                share_of_absolute_pressure=[number(x) for x in np.divide(absolute,denominator,
                    out=np.full(len(absolute),np.nan),where=denominator>0)])
        result=dict(all=subset(np.arange(n)),forced60=subset(np.flatnonzero(forced)),
            single_other=subset(np.flatnonzero(single&~forced)),choice=subset(np.flatnonzero(~single)))
        if detailed:
            result['group_columns']=columns
            result['by_line']={key:[subset(ids)[col] for col in columns]
                for key,ids in groups(metadata)['line'].items()}
            result['by_window']={key:[subset(ids)[col] for col in columns]
                for key,ids in groups(metadata)['window'].items()}
        return result
    errors={key:[number(x) for x in array(value).mean(1)]
        for key,value in payload.get('error_components',{}).items()}
    return dict(n=n,uniform=weighted(np.ones(n),False),exposure_weighted=weighted(weights,True),
        exposure_draws=int(weights.sum()),huber_clipped_fraction=[number(x) for x in clipped.mean(1)],
        raw_error_component_means=errors,
        historical_gradient_reconstruction=False,
        scope='Final frozen-model TD rescore with exact old draw-count weights, not historical parameter-update attribution')
