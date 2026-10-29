"""Verify the fixed source endpoint and a restored next update, without rollout."""
import json
from pathlib import Path
import time

import torch

from state import compare_states, export_state, load_final, save_final

HERE = Path(__file__).resolve().parent


def finish_recovery(learner, args, training):
    p = json.loads((HERE/'protocol.json').read_text())
    out = args.out
    checkpoint = out/'full_state.pt'
    reference = args.thesis_root/p['reference_actor']
    result = dict(status='verification_failed', training=training,
        checkpoint=str(checkpoint), reference_actor=str(reference),
        budget=dict(reused_offline_updates=50000, new_offline_updates=0,
            source_continuation_updates=training['continuation_updates'],
            source_events=training['source_events'], diagnostic_updates=0, diagnostic_attempts=0,
            target_native_seconds=0, target_feedback_labels=0),
        identity={}, restoration={}, next_update={},
        next_stage_authorized=False)
    started = time.monotonic()
    model_class = type(learner.augmentation)
    saved = False
    try:
        save_final(learner, checkpoint, extra=dict(seed=args.seed,
            source_events=training['source_events'],
            initializer=str(args.initializer), historical_reference_actor=str(reference)))
        saved = True
        old = torch.load(reference, map_location='cpu', weights_only=False)
        result['actor_identity'] = compare_states(old['actor'],
            {key:value.detach().cpu() for key,value in learner.actor.state_dict().items()})
        result['identity'] = dict(actor_exact=result['actor_identity']['all_pass'],
            historical_seed=old['extra']['training_seed'] == args.seed == p['training_seed'],
            historical_endpoint=old['extra']['updates'] == learner.total_updates == 70000,
            original_source_events=training['source_events'] == p['expected_source_events'],
            original_update_budget=training['continuation_updates'] == 20000
                and learner.online_critic_updates == 20000,
            original_source_episodes=training['demand_scale_episode_counts'] == p['expected_source_episodes'],
            original_model_mixture=training['model_td_coefficient'] == .25
                and training['real_td_coefficient'] == .75,
            target_labels_unused=training['target_online_training_events'] == 0)
        if not all(result['identity'].values()):
            result['decision'] = 'HOLD_IDENTITY_MISMATCH_NO_EXPANSION'
            return result

        before = export_state(learner)
        # This first diagnostic branch continues the live, un-restored endpoint.
        result['budget']['diagnostic_attempts'] += 1
        learner.update(online=True, collect_metrics=False)
        result['budget']['diagnostic_updates'] += 1
        expected_next = export_state(learner)
        load_final(learner, checkpoint, model_class=model_class, out=out)
        result['restoration'] = compare_states(before, export_state(learner))
        del before
        if not result['restoration']['all_pass']:
            result['decision'] = 'HOLD_RESTORATION_MISMATCH_NO_EXPANSION'
            return result
        result['budget']['diagnostic_attempts'] += 1
        learner.update(online=True, collect_metrics=False)
        result['budget']['diagnostic_updates'] += 1
        result['next_update'] = compare_states(expected_next, export_state(learner))
        del expected_next
        result['status'] = 'complete' if result['next_update']['all_pass'] else 'verification_failed'
        result['decision'] = ('ONE_HISTORICAL_ENDPOINT_RECOVERED' if result['status'] == 'complete'
            else 'HOLD_NEXT_UPDATE_MISMATCH_NO_EXPANSION')
        return result
    except Exception as error:
        result['error'] = dict(type=type(error).__name__, message=str(error))
        result['decision'] = 'HOLD_TECHNICAL_FAILURE_NO_RETRY'
        raise
    finally:
        # Throwaway diagnostic steps never replace or advance the exported endpoint.
        if saved and result['budget']['diagnostic_attempts']:
            try:
                load_final(learner, checkpoint, model_class=model_class, out=out)
                result['restored_terminal_updates'] = learner.total_updates
            except Exception as error:
                result['status'] = 'verification_failed'
                result['decision'] = 'HOLD_TERMINAL_RESTORE_FAILURE_NO_EXPANSION'
                result['terminal_restore_error'] = dict(type=type(error).__name__, message=str(error))
        result['terminal_updates_at_exit'] = learner.total_updates
        result['checkpoint_bytes'] = checkpoint.stat().st_size if saved else None
        result['verification_wall_seconds'] = time.monotonic()-started
        with (out/'recovery_summary.json').open('x') as handle:
            json.dump(result, handle, ensure_ascii=False, indent=2, allow_nan=False)
            handle.write('\n')
        print(json.dumps(dict(stage='recovery_verification', status=result['status'],
            decision=result.get('decision'), budget=result['budget'])), flush=True)
