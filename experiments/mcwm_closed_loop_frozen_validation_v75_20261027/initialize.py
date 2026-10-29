"""Recreate only the original50000-step all-data IQL prefix; no simulator calls."""
from __future__ import annotations

import argparse
from dataclasses import asdict
import json
import logging
import os
from pathlib import Path
import random
import sys
import time

HERE = Path(__file__).resolve().parent
SEEDS = (6206, 6207, 6208)
OFFLINE_UPDATES = 50000
OFFLINE_ROWS = 3284718
CHECKPOINT_EVERY = 10000
# Exact original v8 producer metadata; v12 and v14 copied that IQL initializer.
SOURCE_CONFIG = {'activation': 'relu', 'actor_layer_norm': False, 'actor_learning_rate': 0.0003, 'actor_q_reduction': 'min-all', 'alpha_learning_rate': 0.0003, 'backup_entropy': True, 'batch_size': 256, 'bootstrap_probability': 1.0, 'cql_max_target_backup': False, 'cql_n_actions': 10, 'cql_penalty_subset_replacement': False, 'cql_penalty_subset_size': None, 'critic_layer_norm': False, 'critic_learning_rate': 0.0003, 'critic_loss': 'huber', 'ensemble_size': 2, 'gradient_clip_norm': None, 'hidden_dim': 256, 'huber_delta': 1.0, 'initial_temperature': 1.0, 'iql_advantage_temperature': 3.0, 'iql_expectile': 0.7, 'iql_max_weight': 100.0, 'mixed_backup_lambda': 0.1, 'offline_cql_alpha': 0.0, 'offline_objective': 'iql', 'offline_temperature_action_samples': 1, 'offline_update_order': 'critic-before-actor', 'online_actor_batch_mode': 'latest', 'online_actor_update_interval': 1, 'online_backup': 'h2oplus', 'online_cql_alpha': 0.0, 'online_fraction': 0.5, 'online_update_order': 'actor-before-critic', 'ratio_learning_rate': 0.0001, 'ratio_max_weight': 1.0, 'ratio_min_weight': 1e-05, 'ratio_update_interval': 4, 'reward_scale': 0.0001, 'soft_target_rate': 0.005, 'target_entropy': 0.0, 'target_subset_replacement': False, 'target_subset_size': 2, 'temperature_state_source': 'actor', 'use_ratio_correction': True, 'value_layer_norm': False, 'value_learning_rate': 0.0003}


def validate_template(payload):
    if (payload.get('artifact_schema') != 'transit-offline-initializer-v7-v1'
            or payload.get('total_updates') != OFFLINE_UPDATES
            or payload.get('online_critic_updates') != 0
            or payload.get('source_config') != SOURCE_CONFIG
            or payload.get('extra', {}).get('source_training_seed') != 6201):
        raise ValueError('The source archive differs from the original50000-step IQL recipe')
    return dict(payload['source_config'])


def checkpoint_progress(total):
    if not isinstance(total, int) or not 0 <= total <= OFFLINE_UPDATES:
        raise ValueError('Initializer checkpoint exceeds the fixed offline budget')
    return min(CHECKPOINT_EVERY-total % CHECKPOINT_EVERY, OFFLINE_UPDATES-total)


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix+'.partial')
    temporary.write_text(json.dumps(value, ensure_ascii=False, allow_nan=False, indent=2)+'\n')
    temporary.replace(path)


def run(args):
    import numpy as np
    import torch
    h2o = Path(os.environ['THESIS_H2O_ROOT']).resolve()
    sys.path.insert(0, str(h2o))
    from SimpleSAC.transit_v7.replay_buffer import TransitReplayBufferV7
    from SimpleSAC.transit_v7.trainer import TransitH2OTrainerV7, inspect_offline_initializer_v7
    from SimpleSAC.transit_v3.trainer import TransitTrainerConfig
    from SimpleSAC.train_transit_v3 import run_updates

    if args.seed not in SEEDS:
        raise ValueError('Only the three preregistered new training seeds are supported')
    if not args.device.startswith('cuda') or not torch.cuda.is_available():
        raise ValueError('The original full-float32-cache recipe requires its GPU runtime')
    inputs = args.inputs_dir.resolve()
    seed_dir = inputs/f'seed{args.seed}'
    seed_dir.mkdir(parents=True, exist_ok=True)
    destination = inputs/f'h2oplus_seed{args.seed}.pt'
    gate_path = seed_dir/'initializer_gate.json'
    checkpoint = seed_dir/'checkpoint.pt'
    if gate_path.exists():
        gate = json.loads(gate_path.read_text())
        if gate.get('all_pass') is True and destination.exists():
            print(json.dumps(dict(status='complete', reused_completed_seed=args.seed)), flush=True)
            return gate
        raise ValueError('An existing initializer gate is incomplete; review this seed before restart')
    started = time.monotonic()
    payload = torch.load(args.template, map_location='cpu', weights_only=False)
    config = TransitTrainerConfig(**validate_template(payload))
    expected_contract = payload['extra']['contract_config']
    del payload
    torch.set_num_threads(6)
    torch.set_float32_matmul_precision('high')
    np.random.seed(args.seed)
    random.seed(args.seed)
    logger = logging.getLogger('v75_initializer')
    logger.setLevel(logging.INFO)
    logger.addHandler(logging.StreamHandler())
    logger.addHandler(logging.FileHandler(seed_dir/'training.log', mode='a'))

    def progress(phase, current):
        print('SCHEDULEURM_PROGRESS '+json.dumps(dict(phase=phase, current=current,
            total=OFFLINE_UPDATES, unit='offline_updates', seed=args.seed,
            elapsed_seconds=time.monotonic()-started)), flush=True)

    progress('load_original_all_data_recipe', 0)
    replay = TransitReplayBufferV7(args.dataset, split='train', device=args.device,
        online_capacity=120000, max_offline_rows=0, subsample_seed=args.seed,
        device_cache_observation_dtype='float32')
    if replay.offline_size != OFFLINE_ROWS or replay.contract_config != expected_contract:
        raise ValueError('Full original train split or physical contract differs')
    trainer = TransitH2OTrainerV7(replay, config=config, device=args.device, seed=args.seed)
    compiled = trainer.compile_networks(mode='default')
    resumed_at = 0
    if checkpoint.exists():
        extra = trainer.load(checkpoint)
        if extra.get('source_training_seed') != args.seed or extra.get('phase') != 'offline_pretraining':
            raise ValueError('Initializer resume belongs to a different seed or phase')
        resumed_at = trainer.total_updates
        checkpoint_progress(resumed_at)
    cache = replay.materialize_device_cache(include_online=False)
    recipe = dict(source_template=str(args.template.resolve()),
        original_producer='H2O+ v8 fewshot, original V7 trainer offline prefix',
        dataset=str(args.dataset.resolve()), split='train', offline_rows=replay.offline_size,
        max_offline_rows=0, online_capacity=120000, config=asdict(config),
        contract=expected_contract, cpu_threads=6, matmul_precision='high',
        device_cache_observation_dtype='float32', compiled_modules=compiled,
        compile_mode='default', checkpoint_every=CHECKPOINT_EVERY,
        log_every=250, offline_mode='offline', ratio_pretraining_updates=0,
        simulator_events=0, device_cache=cache)
    write(seed_dir/'recipe.json', recipe)
    extra = dict(phase='offline_pretraining', source_method='h2oplus',
        source_run_name=f'h2oplus_seed{args.seed}', source_training_seed=args.seed,
        training_seed=args.seed, contract_config=expected_contract, mdp_version='v7',
        initialization_study='v75 independent validation',
        source_recipe='v8 original IQL50000 prefix; v73 response recipe unchanged')
    if not checkpoint.exists():
        trainer.save(checkpoint, extra=dict(extra, offline_updates_completed=0), include_online_replay=False)
    latest = {}
    while trainer.total_updates < OFFLINE_UPDATES:
        count = checkpoint_progress(trainer.total_updates)
        latest = run_updates(trainer, count, mode='offline', log_every=250, logger=logger)
        if trainer.online_critic_updates or replay.online_size:
            raise ValueError('Offline initializer unexpectedly acquired online updates or replay')
        trainer.save(checkpoint, extra=dict(extra, offline_updates_completed=trainer.total_updates),
            include_online_replay=False)
        progress('original_offline_iql', trainer.total_updates)
    if trainer.total_updates != OFFLINE_UPDATES:
        raise ValueError('The initializer did not stop at exactly50000 offline updates')
    if not destination.exists():
        trainer.save_offline_initializer(destination, extra=dict(extra,
            offline_updates_completed=OFFLINE_UPDATES, trainer_total_updates=trainer.total_updates,
            offline_metrics=latest, initializer_recipe=recipe))
    inspected = inspect_offline_initializer_v7(destination)
    checks = dict(total50000=inspected['total_updates'] == OFFLINE_UPDATES,
        source_recipe_exact=inspected['source_config'] == SOURCE_CONFIG,
        correct_seed=inspected['extra']['source_training_seed'] == args.seed,
        pure_offline=trainer.online_critic_updates == 0 and replay.online_size == 0,
        complete_original_training_split=replay.offline_size == OFFLINE_ROWS,
        original_contract=replay.contract_config == expected_contract,
        output_within_new_inputs=destination.parent == inputs)
    result = dict(status='complete' if all(checks.values()) else 'integrity_failed',
        all_pass=all(checks.values()), checks=checks, seed=args.seed,
        offline_updates=trainer.total_updates, online_updates=trainer.online_critic_updates,
        recipe_exact=checks['source_recipe_exact'], initializer_path=str(destination),
        resumed_from_offline_update=resumed_at, new_offline_updates_in_this_process=OFFLINE_UPDATES-resumed_at,
        offline_rows=replay.offline_size, native_events=0, ratio_pretraining_updates=0,
        initializer_bytes=destination.stat().st_size, recipe_path=str(seed_dir/'recipe.json'),
        checkpoint_path=str(checkpoint), elapsed_seconds=time.monotonic()-started)
    write(gate_path, result)
    print(json.dumps(result, ensure_ascii=False), flush=True)
    if not result['all_pass']:
        raise ValueError('Generated initializer gate did not pass')
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--seed', type=int, required=True, choices=SEEDS)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--inputs-dir', type=Path, default=HERE/'inputs')
    parser.add_argument('--dataset', type=Path, default=Path(os.environ.get('THESIS_DATASET', '/missing-original-dataset')))
    parser.add_argument('--template', type=Path, default=HERE.parent/'mcwm_closed_loop_formal_20261020/inputs/h2oplus_seed6201.pt')
    run(parser.parse_args())


if __name__ == '__main__':
    main()
