"""One frozen causal factual comparison; callers own fit accounting and logs."""
from experiments.mcwm_cold_start_contract_v128_20261029.model import (
    features, hold_only_features, probability,
)
from experiments.mcwm_cold_start_contract_v128_20261029.timing import causal_events


def samples(history, model):
    return [(features(*inputs) if model == 'full' else hold_only_features(inputs[1]), y)
            for inputs, y in history]


def initialize(prefix, truth, fit):
    history = [(tuple(row['inputs']), truth(row)) for row in prefix]
    full = fit(samples(history, 'full'), 3, dict(phase='initialize', model='full'))
    hold = fit(samples(history, 'hold_only'), 2, dict(phase='initialize', model='hold_only'))
    return dict(history=history, full=tuple(full['beta']), hold_only=tuple(hold['beta']))


def forecast(rows, initial, truth, fit, emit):
    """Emit predictions before truth access; only release events expand history."""
    by_sequence = {row['meta']['sequence']: row for row in rows}
    roots = [dict(meta=row['meta'], selection=row['selection']) for row in rows]
    events = causal_events(roots)
    history = list(initial['history'])
    frozen = tuple(initial['full'])
    current = dict(full=tuple(initial['full']), hold_only=tuple(initial['hold_only']))
    observed, predictions = {}, []
    releases = 0
    latest_deadline = 9000
    for event in events:
        if event['kind'] == 'initialize':
            if len(event['sequences']) != len(history):
                raise ValueError('Initialization history does not match this scene')
            emit(dict(kind='initial_state', rows=len(history), full=frozen,
                      hold_only=current['hold_only']))
        elif event['kind'] == 'release':
            for sequence in event['sequences']:
                history.append((tuple(by_sequence[sequence]['inputs']), observed[sequence]))
            latest_deadline = event['packet_deadline']
            releases += 1
            for model, dimension in (('full', 3), ('hold_only', 2)):
                fitted = fit(samples(history, model), dimension,
                    dict(phase='refit', model=model, packet_deadline=latest_deadline,
                         added_sequences=event['sequences']))
                current[model] = tuple(fitted['beta'])
            emit(dict(kind='release', packet_deadline=latest_deadline,
                sequences=event['sequences'], history_rows=len(history),
                positives=sum(y for _, y in history), coefficients=dict(current)))
        else:
            row = by_sequence[event['sequence']]
            n, h = row['inputs']
            prediction = dict(sequence=event['sequence'], issue_time=event['issue_time'],
                inputs=[n, h], history_rows=len(history), latest_feedback_deadline=latest_deadline,
                coefficients=dict(current), probabilities=dict(
                    fresh=probability(current['full'], features(n, h)),
                    frozen=probability(frozen, features(n, h)),
                    hold_only=probability(current['hold_only'], hold_only_features(h)),
                    past_prevalence=(sum(y for _, y in history)+1)/(len(history)+2),
                    clock_zero_risk=0.0))
            emit(dict(kind='prediction', **prediction))
            # The caller flushes the prediction before this first target access.
            y = truth(row)
            observed[event['sequence']] = y
            emit(dict(kind='outcome', sequence=event['sequence'], target=y))
            predictions.append(dict(prediction, target=y))
    return predictions, dict(predictions=len(predictions), released_packets=releases,
        final_history_rows=len(history), latest_used_deadline=latest_deadline,
        frozen_unchanged=frozen == initial['full'],
        last_coefficients={model:list(beta) for model,beta in current.items()})
