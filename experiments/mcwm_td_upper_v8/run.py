"""Reuse the verified v7 learner without modifying its source or old runs."""

import argparse
from pathlib import Path
import platform
import torch

from td_common import HERE, P, FIXED, CONTEXTS, module, write, smoke_protocol

previous = module("td_upper_training", HERE.parent / "mcwm_context_v7/run.py")
previous.HERE, previous.FIXED = HERE, FIXED


class FixedTrainer(previous.ContextTrainer):
    def configure_context(self, p, scenario, bundle):
        super().configure_context(p, scenario, bundle)
        self.coefficient = FIXED[self.arm]
        self.output = HERE / p["relative_out"] / "results"


previous.ContextTrainer = FixedTrainer


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--arm", required=True, choices=P["arms"])
    parser.add_argument("--scenario", required=True, choices=CONTEXTS)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--evaluate", action="store_true")
    parser.add_argument("--case", type=int, default=10, choices=range(20))
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    args.out = args.out.resolve()
    args.out.mkdir(parents=True, exist_ok=True)
    p = dict(P, arm=args.arm, training_seed=args.seed, relative_out=str(args.out.relative_to(HERE)))
    if args.smoke:
        assert "smoke" in args.out.parts
        p = smoke_protocol(p)
    if args.evaluate:
        previous.evaluate(args, p)
    elif not (args.out / "results/ready.json").exists():
        gpu = None if args.smoke else torch.cuda.get_device_name(0)
        if gpu is not None:
            assert P["expected_gpu_model"] in gpu, gpu
        write(args.out / "results/runtime.json", dict(host=platform.node(), torch_version=torch.__version__,
            cuda_version=torch.version.cuda, gpu_model=gpu, device="cpu" if args.smoke else "cuda"))
        previous.train(args, p)
    print("DONE fixed TD upper-range study", flush=True)
