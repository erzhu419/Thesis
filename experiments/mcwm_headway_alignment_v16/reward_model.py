"""Model next decision index; compare raw reward and reconstructed feedback."""

import copy
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

from headway_common import P, state_reward, write
import dual_model as old
from dual_networks import Encoder
from model import WIDTH

old_targets,old_losses,old_unpack = old.targets,old.losses,old.unpack


def targets(data,stations):
    result = old_targets(data,stations)
    result["current"] = (data["next_observations"][:,2]*15).round().long()
    return result


def losses(output,y,norm):
    return torch.cat((old_losses(output[:,:WIDTH],y,norm),
        F.cross_entropy(output[:,WIDTH:],y["current"],reduction="none")[:,None]),1)


class Model(old.Model):
    def __init__(self,base=None):
        super().__init__(base)
        self.head[-1] = nn.Linear(256,WIDTH+16)
        if base is not None:
            nn.init.zeros_(self.head[-1].weight)
            nn.init.zeros_(self.head[-1].bias)


class RawReward(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = Encoder(layer_norm=True,activation="silu")
        self.head = nn.Sequential(nn.Linear(258,128),nn.SiLU(),nn.Linear(128,1))

    def forward(self,s,a):
        return self.head(torch.cat((self.encoder(s),a),-1)).flatten()


class Models(old.Models):
    raw = False

    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        if self.raw:
            fit = self.target["fit"]["rewards"]
            self.reward_mean,self.reward_std = fit.mean(),fit.std().clamp_min(1.)
            with torch.random.fork_rng(devices=[]):
                torch.manual_seed(self.seed+87000)
                self.reward_model = RawReward()
            self.reward_rng = np.random.default_rng(self.seed+88000)
            self.fit_reward(P["auxiliary_reward_updates"],P["warmup_events"])

    def fit_reward(self,steps,event):
        fit,stop = self.target["fit"],self.target["stop"]
        self.reward_model.requires_grad_(True)
        optimizer = torch.optim.Adam(self.reward_model.parameters(),lr=3e-4)
        best,saved = float("inf"),None
        for step in range(8 if self.smoke else steps):
            ids = torch.as_tensor(self.reward_rng.integers(len(fit["rewards"]),size=128))
            predicted = self.reward_model(fit["observations"][ids],fit["actions"][ids])
            loss = (predicted-(fit["rewards"][ids]-self.reward_mean)/self.reward_std).square().mean()
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.reward_model.parameters(),10.)
            optimizer.step()
            if (step+1)%100==0 or step+1==(8 if self.smoke else steps):
                with torch.no_grad():
                    score = float((self.reward_model(stop["observations"],stop["actions"])-(stop["rewards"]-self.reward_mean)/self.reward_std).square().mean())
                if score<best:
                    best,saved = score,copy.deepcopy(self.reward_model.state_dict())
        self.reward_model.load_state_dict(saved)
        self.reward_model.eval().requires_grad_(False)
        with torch.no_grad():
            calibration = self.target["calibrate"]
            predicted = self.predict_reward(calibration["observations"],calibration["actions"])
            actual = calibration["rewards"]
            write(self.folder / f"reward_probe_{event:05d}.json",dict(event=event,stop_normalized_mse=best,
                actual_reward_mean=float(actual.mean()),predicted_reward_mean=float(predicted.mean()),
                raw_reward_mae=float((predicted-actual).abs().mean()),rows=len(actual)))

    def predict_reward(self,s,a):
        return (self.reward_model(s,a)*self.reward_std+self.reward_mean).clamp_max(0.)

    def refit(self,event):
        super().refit(event)
        if self.raw:
            self.fit_reward(P["model_refit_updates"],event)

    @torch.no_grad()
    def decode(self,s,output):
        ns,reward,dt = super().decode(s,output)
        count = (ns[:,1]*16).round().long()
        logits = output[:,WIDTH:].clone()
        logits.masked_fill_(torch.arange(16)[None]>=count[:,None],-torch.inf)
        current = torch.multinomial(logits.softmax(-1),1,generator=self.generator).flatten()
        ns[:,2] = current.float()/15
        rows = torch.arange(len(s))
        previous_index = (s[:,2]*15).round().long()
        source_slots = s[:,5:229].reshape(-1,16,14)
        slots = ns[:,5:229].reshape(-1,16,14)
        # A next-station transition cannot change the controlled vehicle's line.
        line = source_slots[rows,previous_index,1].round().long()
        source_station = (source_slots[rows,previous_index,2]*self.stations[line]).round()
        station = (slots[rows,current,2]*self.stations[line]).round()
        station = torch.maximum(station,source_station+1).minimum(self.stations[line])
        slots[rows,current,1],slots[rows,current,2] = line.float(),station/self.stations[line]
        return ns,reward,dt

    @torch.no_grad()
    def generate(self,actor,size):
        batch,weights = super().generate(actor,size)
        batch["discounts"].fill_(P["discount"])
        if self.raw:
            batch["rewards"] = self.predict_reward(batch["observations"],batch["actions"])
        return batch,weights

    def save(self):
        super().save()
        if self.raw:
            torch.save(dict(model=self.reward_model.state_dict(),mean=self.reward_mean,std=self.reward_std),self.folder / "raw_reward.pt")


old.targets,old.losses,old.Model = targets,losses,Model
old.unpack = lambda output: old_unpack(output[:,:WIDTH])
old.ridge_of_state = state_reward
