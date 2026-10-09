"""Read the resident chronological data in blocks and emit only aggregate JSON."""

import json
from pathlib import Path

import h5py
import numpy as np

from headway_common import HERE,state_reward


def selected(states):
    index = np.rint(states[:,2]*15).astype(int)
    return states[:,5:229].reshape(-1,16,14)[np.arange(len(states)),index]


def run():
    counts = dict(rows=0,nonterminal_rows=0,zero_time_rows=0,zero_time_nonzero_headway_rewards=0,
                  different_selected_line_rows=0,next_current_slot_not_zero=0,
                  headway_ratio_at_clip_limit_rows=0)
    sums = dict(headway_reward=0.,physical_time_seconds=0.,historical_reward=0.)
    with h5py.File(HERE.parent / "bus_dual_aligned_v4/data/dataset.h5") as data:
        for start in range(0,len(data["actions"]),65536):
            stop = start+65536
            s,ns = data["observations"][start:stop],data["next_observations"][start:stop]
            dt,r,terminal = data["delta_t_seconds"][start:stop],data["headway_rewards"][start:stop],data["terminals"][start:stop]>.5
            current,next_current = selected(s),selected(ns)
            physical = ~terminal
            zero = physical&(dt==0)
            counts["rows"] += len(s)
            counts["nonterminal_rows"] += int(physical.sum())
            counts["zero_time_rows"] += int(zero.sum())
            counts["zero_time_nonzero_headway_rewards"] += int((zero&(abs(r)>1e-6)).sum())
            counts["different_selected_line_rows"] += int((physical&(current[:,1]!=next_current[:,1])).sum())
            counts["next_current_slot_not_zero"] += int((physical&(ns[:,2]>0)).sum())
            counts["headway_ratio_at_clip_limit_rows"] += int((physical&((next_current[:,3]>=5)|(next_current[:,4]>=5))).sum())
            sums["headway_reward"] += float(r[physical].astype(np.float64).sum())
            sums["physical_time_seconds"] += float(dt[physical].astype(np.float64).sum())
            sums["historical_reward"] += float(state_reward(ns[physical]).astype(np.float64).sum())
    n = counts["nonterminal_rows"]
    report = dict(experiment="existing dual v4 dataset diagnostic",analysis_location="server",counts=counts,
        fractions={k:counts[k]/n for k in counts if k not in ("rows","nonterminal_rows")},
        nonterminal_means={k:v/n for k,v in sums.items()},
        identity_scope="Stored global states have no vehicle ID; different line proves a different vehicle, equal line does not prove identity.")
    (HERE / "existing_contract_diagnostic.json").write_text(json.dumps(report,indent=2)+"\n")
    print(json.dumps(report))


if __name__=="__main__":
    run()
