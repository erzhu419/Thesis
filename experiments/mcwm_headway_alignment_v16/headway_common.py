"""Isolated headway contract; do not shadow the simulator's common package."""

import json
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
P = json.loads((HERE / "protocol.json").read_text())
sys.path.insert(0, str(HERE.parent / "bus_dual_aligned_v4"))
import aligned_common as old
from aligned_common import CONFIG, state, physical_action, pack_event_batch_v7, write
import numpy as np


def reward(forward, backward, tf, tb, fp=True, bp=True):
    f,b,tf,tb = np.broadcast_arrays(forward,backward,tf,tb)
    fp,bp = np.broadcast_arrays(fp,bp)
    df,db = abs(f-tf),abs(b-tb)
    ratio = tf / np.maximum(tb,1e-6)
    weight = df / (df+db+1e-6)
    both = -weight*df-(1-weight)*db-.5*abs(f-ratio*b)/((1+ratio)/2)
    base = np.where(fp&bp,both,np.where(fp,-df,np.where(bp,-db,-50.)))
    penalty = np.where(fp&(tf>0),20*np.tanh((df-.5*tf)/30),0.)+np.where(bp&(tb>0),20*np.tanh((db-.5*tb)/30),0.)
    return base-np.maximum(0.,penalty)


def raw_reward(raw):
    raw = np.asarray(raw)
    return float(reward(raw[3],raw[4],raw[5],raw[6],raw[11]>.5,raw[12]>.5))


def state_reward(states):
    values = np.asarray(states)
    index = np.rint(values[:,2]*15).astype(int)
    slots = values[:,5:229].reshape(-1,16,14)[np.arange(len(values)),index]
    tf,tb = slots[:,5]*720,slots[:,6]*720
    return reward(slots[:,3]*tf,slots[:,4]*tb,tf,tb,slots[:,11]>.5,slots[:,12]>.5)


def folder(scene,arm,seed,smoke=False):
    return HERE / ("smoke" if smoke else "runs") / scene / f"{arm}_seed{seed}"


def data_folder(smoke=False):
    return HERE / ("smoke_data" if smoke else "data")


def init_folder(seed,smoke=False):
    return HERE / ("smoke_inputs" if smoke else "inputs") / f"seed{seed}"
