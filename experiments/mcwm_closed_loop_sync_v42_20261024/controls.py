"""Fixed-parameter controls only; no actor snapshot is reconstructed here."""
import math
from experiments.mcwm_closed_loop_trace_design_v40_20261024.behavior import (
    execution_measure,ratio_from_measures,
)
from experiments.mcwm_closed_loop_overlap_v41_20261024.overlap import conditional_overlap


def oracle_gate_behavior(behavior,target):
    result=dict(behavior)
    for key in ('positive_probability','zero_probability'):
        result[key]=target[key]
    return result


def score(transform,state,action,behavior,target,point_geometry,all_geometry):
    numerator=execution_measure(transform,state,action,target,geometry=point_geometry)
    denominator=execution_measure(transform,state,action,behavior,geometry=point_geometry)
    ratio=ratio_from_measures(denominator,numerator)
    overlap=conditional_overlap(transform,state,behavior,target,geometry=all_geometry)
    if ratio['log_c'] is None or overlap['expected_log_c_under_behavior'] is None:
        raise RuntimeError('Unresolved control measure is not zero credit')
    return dict(log_c=ratio['log_c'],log_ratio=ratio['log_ratio'],c=ratio['c'],
        expected_log_c=overlap['expected_log_c_under_behavior'],log_overlap=overlap['log_overlap'],
        overlap=overlap['total_overlap'],atom_overlap=overlap['atom_overlap'],
        continuous_overlap=overlap['continuous_overlap'],normalization_error=overlap['normalization_error'],
        jensen_gap=overlap['jensen_gap'],training_authorized=False)


def summarize(rows,count):
    selected=rows[:count]
    old=math.fsum(row['log_c'] for row in selected)
    geom=math.fsum(row['expected_log_c'] for row in selected)
    arith=math.fsum(row['log_overlap'] for row in selected)
    return dict(links=count,L_old=old,L_geom=geom,L_arith=arith,
        old_minus_geom=old-geom,jensen_gap=arith-geom,
        overlap_mean=math.fsum(row['overlap'] for row in selected)/count,
        atom_overlap_mean=math.fsum(row['atom_overlap'] for row in selected)/count,
        continuous_overlap_mean=math.fsum(row['continuous_overlap'] for row in selected)/count,
        mixed_path_log_weight=count*math.log(.9),
        full_old_log_weight=old+count*math.log(.9),
        full_fixed_state_geom_log_weight=geom+count*math.log(.9))
