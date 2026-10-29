"""Small paired service-clock readouts trained only on admitted v21 pairs."""
import numpy as np

LINES = ('7X', '102X', '705X')
KINDS = ('arrival', 'departure')
SCALE_SECONDS = 60.
HOLD_DIFFERENCE = 7.5


def shift_targets(low, high):
    if low['stations'] != high['stations']:
        raise ValueError('Paired service targets must refer to the same future stops')
    return dict(stations=low['stations'], **{kind:[
        None if a is None or b is None else float(b)-float(a)-HOLD_DIFFERENCE
        for a,b in zip(low[kind],high[kind])] for kind in KINDS})


def design(features, history, use_history):
    values = list(features[:4])
    if use_history:
        if history.get('status') != 'supported' or history.get('mean_residual_scaled') is None:
            return None
        values.append(float(history['mean_residual_scaled']))
    return np.asarray(values, dtype=float)


class ServiceShiftReadout:
    def __init__(self, *, use_history=False, ridge=1.):
        self.use_history, self.ridge = bool(use_history), float(ridge)
        self.records, self.weights, self.support, self.refits = [], {}, {}, 0

    def accept(self, records, now):
        known = {r['id'] for r in self.records}
        for record in records:
            if record['role'] != 'train' or record.get('selected_for_training') is not True:
                raise ValueError('Service fitting accepts only budget-admitted training pairs')
            if record['available_at'] > now or record['admitted_at'] > now:
                raise ValueError('Service label cannot be fitted before chronological release')
            if record['id'] in known:
                raise ValueError('A paired service row cannot be fitted twice')
            known.add(record['id'])
            self.records.append(record)

    def refit(self):
        for kind in KINDS:
            for i in range(3):
                rows = [(r,design(r['features'],r['service_history'],self.use_history)) for r in self.records
                    if i < len(r['service_targets'][kind]) and r['service_targets'][kind][i] is not None]
                rows = [(r,x) for r,x in rows if x is not None]
                key = kind,i
                self.support[key] = {line:sum(r['line_id']==line for r,_ in rows) for line in LINES}
                if not rows:
                    self.weights[key] = None
                    continue
                x = np.stack([x for _,x in rows])
                y = np.asarray([r['service_targets'][kind][i] for r,_ in rows])/SCALE_SECONDS
                self.weights[key] = np.linalg.solve(x.T@x+self.ridge*np.eye(x.shape[1]),x.T@y)
        self.refits += 1

    def predict(self, features, history):
        x = design(features,history,self.use_history)
        line = LINES[int(np.argmax(features[:3]))]
        predictions, support = {}, {}
        for kind in KINDS:
            predictions[kind],support[kind] = [],[]
            for i in range(3):
                weight = self.weights.get((kind,i))
                n = self.support.get((kind,i),{}).get(line,0)
                support[kind].append(n)
                predictions[kind].append(float(x@weight*SCALE_SECONDS)+HOLD_DIFFERENCE
                    if x is not None and weight is not None and n else None)
        n = sum(v is not None for values in predictions.values() for v in values)
        return dict(status='supported' if n==6 else 'partially_supported' if n else 'unsupported',
            **predictions,support_pairs=support,
            semantics='Predicted high7.5s-minus-low0s arrival/departure differences; fits residuals from7.5s with target scale60 and fixed ridge1.')

    def summary(self):
        return dict(model='service_shift_history5' if self.use_history else 'service_shift_shared4',
            pairs=len(self.records),pair_ids=[r['id'] for r in self.records],
            columns=5 if self.use_history else 4,ridge=self.ridge,target_scale=SCALE_SECONDS,
            refits=self.refits,by_line={line:sum(r['line_id']==line for r in self.records) for line in LINES},
            endpoint_support={kind:[self.support.get((kind,i),{}) for i in range(3)] for kind in KINDS})
