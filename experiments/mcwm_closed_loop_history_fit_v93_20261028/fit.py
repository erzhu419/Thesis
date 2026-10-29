"""Three fixed ridge solves on the complete v92 training cohort, followed by readout."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parents[1]))
from experiments.mcwm_closed_loop_history_fit_v93_20261028.source import load_training
import numpy as np

MODELS=('line3','service5','service_history6')
WIDTHS=(3,5,6)
FEATURES=dict(line3=['line_7X','line_102X','line_705X'],
    service5=['line_7X','line_102X','line_705X','load_centered','margin_centered'],
    service_history6=['line_7X','line_102X','line_705X','load_centered','margin_centered','corridor_q_centered'])


def write(path,value):
    path=Path(path)
    if path.exists():raise FileExistsError(path)
    temporary=path.with_name(path.name+'.writing')
    with temporary.open('x') as handle:
        json.dump(value,handle,ensure_ascii=False,separators=(',',':'),allow_nan=False)
        handle.write('\n')
    temporary.rename(path)


def run():
    protocol=json.loads((HERE/'protocol.json').read_text())
    if not (protocol['models']==list(MODELS) and protocol['ridge']==1.
            and protocol['denominator']==15079. and protocol['hold_scale']==7.5
            and protocol['budget']['model_solves']==3):
        raise RuntimeError('The fixed three-readout contract must agree')
    if (HERE/'fit_attempt.json').exists() or (HERE/'model.json').exists():
        raise RuntimeError('Inspect the existing fit; never repeat its solves')
    source=load_training(protocol['source_summary'],protocol['frozen_protocol'])
    if not source['all_pass']:raise RuntimeError('Training input gate failed before any solve')
    audit=json.loads(Path(protocol['source_reference_audit']).read_text())
    if not (audit['all_pass'] and audit['reused_pairs']==11):
        raise RuntimeError('Frozen references and reuse manifest have not passed')
    write(HERE/'fit_attempt.json',dict(models=list(MODELS),ridge=1.,training_pairs=12,
        source_summary=source['source_summary'],requested_solves=3,new_native_seconds=0))
    started=time.monotonic()
    solves=0
    try:
        y=np.asarray([row['target'] for row in source['rows']],dtype=np.float64)
        models={}
        for name,width in zip(MODELS,WIDTHS):
            matrix=np.asarray(source['matrices'][name],dtype=np.float64)
            if matrix.shape!=(12,width):raise RuntimeError('The declared matrix shape changed')
            coef=np.linalg.solve(matrix.T@matrix+np.eye(width,dtype=np.float64),matrix.T@y)
            solves+=1
            prediction=matrix@coef
            residual=matrix.T@(y-prediction)-coef
            models[name]=dict(coef=coef.tolist(),feature_names=FEATURES[name],
                prediction=prediction.tolist(),design_rank=int(np.linalg.matrix_rank(matrix)),
                solve_index=solves,normal_equation_residual_max=float(np.max(np.abs(residual))))
        line=np.asarray(models['line3']['coef'])
        line_differences={name:float(np.max(np.abs(np.asarray(row['coef'][:3])-line)))
                          for name,row in models.items()}
        checks=dict(frozen_training_source=source['all_pass'],three_fixed_solves=solves==3,
            full_declared_ranks=all(row['design_rank']==width for row,width in zip(models.values(),WIDTHS)),
            finite_coefficients_and_predictions=all(np.isfinite(row['coef']).all()
                and np.isfinite(row['prediction']).all() for row in models.values()),
            line_coefficients_preserved=max(line_differences.values())<=protocol['line_coefficient_tolerance'],
            normal_equations=all(row['normal_equation_residual_max']<=protocol['normal_equation_tolerance']
                for row in models.values()))
        model=dict(name=protocol['name'],models=models,training_rows=source['rows'],
            design=dict(matrices={name:np.asarray(matrix).tolist()
                for name,matrix in source['matrices'].items()},line_centers=source['line_centers']),
            source_audit={key:source[key] for key in ('all_pass','checks','source_summary','source_protocol')},
            ridge=1.,denominator=15079.,hold_scale=7.5,precision='float64',
            line_coefficient_max_differences=line_differences,
            integrity=dict(all_pass=all(checks.values()),checks=checks),
            budget=dict(protocol['budget'],model_solves=solves),
            condition_contract=protocol['condition'],reference_contract=protocol['reference'],
            reference_source_paths=audit['reference_source_paths'],
            common_support=dict(action_difference=protocol['fitting']['a'],
                reference_min=3.,reference_max_exclusive=3.75,candidate_seconds=7.5,
                maximum_history_age_seconds=900,decision_threshold=0.,maximum_interventions=1,
                q_training_ranges={line:[min(row['features'][2] for row in source['rows'] if row['line_id']==line),
                    max(row['features'][2] for row in source['rows'] if row['line_id']==line)]
                    for line in ('7X','102X','705X')}),
            label_available_at=24030,fitted_offline_after_label_maturity=True,
            scope='Training-only response fits; not a deployed actor or evaluated control policy',
            future_validation=protocol['future_validation'],validation_started=False,
            elapsed_seconds=time.monotonic()-started)
        write(HERE/'model.json',model)
        from experiments.mcwm_closed_loop_history_fit_v93_20261028.analysis import analyze
        result=analyze(model)
        encoded=json.dumps(result,ensure_ascii=False,separators=(',',':'),allow_nan=False)
        if len(encoded.encode())>64000:raise ValueError('Keep the training readout below64KB')
        write(HERE/'summary.json',result)
        print(json.dumps(dict(status=result['status'],models_saved=str(HERE/'model.json'),
            solves=solves,new_native_seconds=0,elapsed_seconds=time.monotonic()-started)),flush=True)
        if not model['integrity']['all_pass']:raise RuntimeError('Numerical fit integrity failed; do not refit')
        return result
    except Exception as error:
        write(HERE/'failure.json',dict(type=type(error).__name__,message=str(error),
            model_solves=solves,model_saved=(HERE/'model.json').exists(),new_native_seconds=0,
            new_labels=0,v87_truth_used=0))
        raise


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.parse_args();run()
