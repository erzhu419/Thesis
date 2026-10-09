"""Reuse the proven semantic global model and normalized Bellman trust."""

import copy
import importlib.util
import sys

import numpy as np
import torch
from torch import nn

from aligned_common import P, HERE, EXPERIMENTS, STATE_DIM, ridge_of_state, write
from dual_networks import Encoder
sys.path.insert(0, str(EXPERIMENTS / "mcwm_bus"))
from model import WIDTH, Normalizer, targets as old_targets, losses, unpack
from mc_wm.policy.qdelta_bellman import QDeltaBellman
sys.path.insert(0, str(EXPERIMENTS / "mcwm_usage_v3"))
from mechanisms import update_normalized_trust


def targets(data, stations):
    old = dict(s=data["observations"][:,:229], ns=data["next_observations"][:,:229],
               a=data["actions"][:,:1], dt=data["delta_t_seconds"], r=data["rewards"])
    result = old_targets(old, stations, "cpu")
    result["s"], result["a"] = data["observations"], data["actions"]
    return result


class Model(nn.Module):
    def __init__(self, base=None):
        super().__init__()
        self.encoder = Encoder(layer_norm=True, activation="silu")
        self.head = nn.Sequential(nn.Linear(258,256),nn.SiLU(),nn.Linear(256,WIDTH))
        self.base = copy.deepcopy(base).requires_grad_(False) if base is not None else None
        if base is not None:
            nn.init.zeros_(self.head[-1].weight)
            nn.init.zeros_(self.head[-1].bias)

    def forward(self,s,a):
        output = self.head(torch.cat((self.encoder(s),a),-1))
        return output if self.base is None else output+self.base(s,a)


class Models:
    def __init__(self,replay,seed,folder,smoke=False):
        self.replay,self.seed,self.folder,self.smoke = replay,seed,folder,smoke
        self.rng = np.random.default_rng(seed+50000)
        role = replay.offline["role"].astype(int)
        physical = (replay.offline["delta_t_seconds"] > 0)&(replay.offline["terminals"] < .5)
        partitions = {"fit":np.flatnonzero(physical&(role<8)), "stop":np.flatnonzero(physical&(role==8)),
                      "calibrate":np.flatnonzero(physical&(role==9))}
        if smoke:
            rows = np.flatnonzero(physical)
            partitions = dict(zip(("fit","stop","calibrate"), np.array_split(rows,[int(.7*len(rows)),int(.85*len(rows))])))
        if any(not len(v) for v in partitions.values()):
            raise ValueError("Nonempty trajectory-separated world-model partitions are required")
        self.target = {k:self.take(replay.offline,v,8192 if k=="fit" else 2048) for k,v in partitions.items()}
        ids = np.flatnonzero((replay.online["delta_t_seconds"][:replay.online_size]>0)&(replay.online["terminals"][:replay.online_size]<.5))
        boundary = int(.8*len(ids))
        self.source = {"fit":self.take(replay.online,ids[:boundary],8192),"stop":self.take(replay.online,ids[boundary:],2048)}
        self.stations = torch.as_tensor(replay.stations)
        self.norm = Normalizer(targets(self.source["fit"],self.stations))
        self.banks = {k:[] for k in ("source","direct","residual")}
        self.history = []
        for member in range(P["model_members"]):
            torch.manual_seed(seed+50000+member)
            source = Model()
            self.fit(source,self.source,P["model_updates"])
            source.eval().requires_grad_(False)
            self.banks["source"].append(source)
            for family in ("direct","residual"):
                torch.manual_seed(seed+51000+member)
                candidate = Model(source if family=="residual" else None)
                self.fit(candidate,self.target,P["model_updates"])
                self.banks[family].append(candidate.eval().requires_grad_(False))
        self.trust = QDeltaBellman(STATE_DIM,2,gamma=P["trust_gamma"],device="cpu")
        self.generator = torch.Generator().manual_seed(seed+40000)
        self.select(P["warmup_events"])

    def take(self,data,ids,limit):
        ids = np.sort(self.rng.choice(ids,min(len(ids),limit),replace=False))
        return {k:torch.as_tensor(data[k][ids],dtype=torch.float32) for k in
                ("observations","actions","next_observations","rewards","delta_t_seconds")}

    def fit(self,model,data,steps,weights=None):
        fit, stop = targets(data["fit"],self.stations), targets(data["stop"],self.stations)
        params = [p for p in model.parameters() if p.requires_grad]
        optimizer = torch.optim.Adam(params,lr=3e-4)
        best,saved = float("inf"),None
        steps = 8 if self.smoke else steps
        for step in range(steps):
            ids = torch.as_tensor(self.rng.integers(len(fit["s"]),size=128))
            y = {k:v[ids] for k,v in fit.items()}
            error = losses(model(y["s"],y["a"]),y,self.norm).mean(1)
            loss = error.mean() if weights is None else (error*weights[ids]).sum()/weights[ids].sum()
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params,10.)
            optimizer.step()
            if (step+1)%100==0 or step+1==steps:
                with torch.no_grad():
                    score = float(losses(model(stop["s"],stop["a"]),stop,self.norm).mean())
                if score < best:
                    best,saved = score,copy.deepcopy(model.state_dict())
        model.load_state_dict(saved)

    @torch.no_grad()
    def predict(self,s,a,family=None):
        return torch.stack([m(s,a) for m in self.banks[family or self.family]]).mean(0)

    def select(self,event):
        y = targets(self.target["calibrate"],self.stations)
        with torch.no_grad():
            scores = {k:float(losses(self.predict(y["s"],y["a"],k),y,self.norm).mean()) for k in ("direct","residual")}
        self.family = min(scores,key=scores.get)
        self.history.append(dict(source_events=event,selected_family=self.family,calibration_loss=scores))
        write(self.folder / "history.json",self.history)

    def refit(self,event):
        from aligned_common import MCROOT
        spec = importlib.util.spec_from_file_location("aligned_policy_density",MCROOT / "mc_wm/residual/policy_density.py")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        PolicyDensity = module.PolicyDensity
        data = self.replay.online
        start,stop = max(0,self.replay.online_size-2048),self.replay.online_size
        reference = np.concatenate((data["observations"][start:stop],data["actions"][start:stop]),-1)
        density = PolicyDensity(strategy="buffer",w_min=.1,random_state=self.seed)
        density.fit_reference(reference)
        fit = self.target["fit"]
        query = torch.cat((fit["observations"],fit["actions"]),-1).numpy()
        # cdist computes the same kNN weights without a rows x refs x features temporary.
        query = torch.as_tensor((query-density._ref_mean)/density._ref_std)
        reference = torch.as_tensor(density._ref_sa)
        k = min(density.knn_k,len(reference))
        distances = torch.cat([torch.cdist(query[i:i+256],reference,compute_mode="donot_use_mm_for_euclid_dist")
            .topk(k,largest=False).values.square().mean(1).add(1e-8).sqrt() for i in range(0,len(query),256)])
        bandwidth = float(np.median(distances.numpy()))+1e-6
        raw = torch.exp(-distances/bandwidth)
        weights = density.w_min+(1-density.w_min)*raw/raw.max()
        for family in ("direct","residual"):
            for model in self.banks[family]:
                model.encoder.requires_grad_(True)
                model.head.requires_grad_(True)
                self.fit(model,self.target,P["model_refit_updates"],weights)
                model.eval().requires_grad_(False)
        self.select(event)

    def update_trust(self,actor):
        y = targets(self.target["calibrate"],self.stations)
        ids = torch.as_tensor(self.rng.integers(len(y["s"]),size=128))
        batch = {k:v[ids] for k,v in y.items()}
        with torch.no_grad():
            error = losses(self.predict(batch["s"],batch["a"]),batch,self.norm).mean(1)
        ns = self.target["calibrate"]["next_observations"][ids]
        return update_normalized_trust(self.trust,batch["s"],batch["a"],ns,torch.zeros(len(ids)),error,
            lambda s:actor(s,deterministic=True)[0],1)

    @torch.no_grad()
    def generate(self,actor,size):
        fit = self.target["fit"]
        eligible = torch.nonzero(fit["observations"][:,0]<.75).flatten()
        ids = eligible[torch.randint(len(eligible),(size,),generator=self.generator)]
        s = fit["observations"][ids]
        a = actor(s,deterministic=True)[0]
        ns,reward,dt = self.decode(s,self.predict(s,a))
        if self.replay.objective == "headway":
            reward = torch.as_tensor(ridge_of_state(ns.numpy()),dtype=torch.float32)
        batch = dict(observations=s,actions=a,next_observations=ns,rewards=reward,
                     delta_t_seconds=dt,discounts=torch.ones(size),terminals=torch.zeros(size))
        return batch,self.trust.weight(s,a)

    @torch.no_grad()
    def decode(self,s,output):
        pred = unpack(output)
        global_values = pred["g"]*self.norm.g_std+self.norm.g_mean
        dt = torch.expm1(global_values[:,0].clamp(0,10)).clamp_min(1.)
        dt = torch.minimum(dt,(1-s[:,0])*24000-1)
        reward = -torch.expm1(global_values[:,1].clamp(0,20))
        draw = lambda logits:torch.multinomial(logits.softmax(-1).flatten(0,-2),1,generator=self.generator).reshape(logits.shape[:-1])
        count,line = draw(pred["count"])+1,draw(pred["line"])
        logits = pred["station"].clone()
        legal = torch.arange(47)[None,None]<=self.stations[line,None]
        logits.masked_fill_(~legal,-torch.inf)
        station = draw(logits)
        mask = torch.arange(16)[None]<count[:,None]
        ns = torch.zeros_like(s)
        ns[:,0],ns[:,1] = s[:,0]+dt/24000,count/16
        ns[:,3:5] = (s[:,3:5]+global_values[:,2:4]/20000).clamp(0,1.25)
        slots = ns[:,5:229].reshape(-1,16,14)
        slots[:,:,0],slots[:,:,1],slots[:,:,2] = mask,line,station/self.stations[line]
        values = pred["local"]*self.norm.local_std+self.norm.local_mean
        lower = torch.tensor([0,0,.001,.001,0,0,0,0])
        upper = torch.tensor([5,5,10,10,2,1,2,2])
        slots[:,:,3:11] = values.clamp(lower,upper)
        flags = torch.bernoulli(pred["flags"].sigmoid(),generator=self.generator)
        slots[:,:,11:13] = flags
        slots[:,:,3:5] = torch.where(flags>.5,slots[:,:,3:5],1.)
        slots[:,:,:13] *= mask[:,:,None]
        slots[:,:,13] = -1.
        return ns,reward,dt

    def save(self):
        self.folder.mkdir(parents=True,exist_ok=True)
        torch.save(dict(banks={k:[m.state_dict() for m in v] for k,v in self.banks.items()},
                        normalizer=self.norm.state_dict(),family=self.family,trust=self.trust.net.state_dict()),self.folder / "models.pt")
