"""Fixed-parent bus mechanisms with an equal-capacity unrestricted control."""

import torch
from torch import nn

from model import WIDTH, TransitStateEncoderV7


SLOT_PARENTS = {
    "event": tuple(range(14)),
    "duration": (0, 1, 2, 3, 4, 5, 6, 10, 11, 12, 13),
    "service": (0, 1, 2, 5, 6, 7, 8, 9, 10, 13),
    "headway": (0, 1, 2, 3, 4, 5, 6, 10, 11, 12, 13),
    "cost": (0, 1, 2, 5, 6, 7, 8, 9, 10, 13),
}
GLOBAL_PARENTS = {name: ((0, 1, 2) if name in ("duration", "headway") else tuple(range(5)))
                  for name in SLOT_PARENTS}
OUTPUTS = {
    "event": list(range(4, 20))+[20+69*j+k for j in range(16) for k in range(59)],
    "duration": [0],
    "service": [2, 3]+[20+69*j+k for j in range(16) for k in range(65, 69)],
    "headway": [20+69*j+k for j in range(16) for k in range(59, 65)],
    "cost": [1],
}


class FactoredBusModel(nn.Module):
    def __init__(self, base=None, *, masked=True):
        super().__init__()
        self.encoder = nn.ModuleDict({name: TransitStateEncoderV7() for name in OUTPUTS})
        self.head = nn.ModuleDict({name: nn.Sequential(
            nn.Linear(260 if name == "cost" else 257, 128), nn.SiLU(),
            nn.Linear(128, len(indices))) for name, indices in OUTPUTS.items()})
        for name in OUTPUTS:
            mask = torch.ones(229)
            if masked:
                mask.zero_()
                mask[list(GLOBAL_PARENTS[name])] = 1
                for slot in range(16):
                    mask[[5+14*slot+k for k in SLOT_PARENTS[name]]] = 1
            self.register_buffer("parents_"+name, mask)
            self.register_buffer("outputs_"+name, torch.tensor(OUTPUTS[name]))
        self.base = base
        if base is not None:
            base.requires_grad_(False)
            for head in self.head.values():
                nn.init.zeros_(head[-1].weight)
                nn.init.zeros_(head[-1].bias)

    def forward(self, states, actions):
        baseline = None
        if self.base is not None:
            with torch.no_grad():
                baseline = self.base(states, actions)
        output = states.new_zeros((len(states), WIDTH))
        for name in ("event", "duration", "service", "headway"):
            features = self.encoder[name](states*getattr(self, "parents_"+name))
            output[:, getattr(self, "outputs_"+name)] = self.head[name](torch.cat((features, actions), -1))
        if baseline is not None:
            output = output+baseline
        # A cost prediction shares the predicted event duration and population changes.
        features = self.encoder["cost"](states*self.parents_cost)
        cost = self.head["cost"](torch.cat((features, actions, output[:, [0, 2, 3]]), -1))
        output = output.clone()
        output[:, 1:2] = cost if baseline is None else cost+baseline[:, 1:2]
        return output
