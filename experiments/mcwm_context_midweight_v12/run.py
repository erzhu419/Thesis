"""Change fixed coefficients only; retain the v11 gated trainer verbatim."""

import argparse
from pathlib import Path
import platform
import torch

from midweight_common import HERE, P, FIXED, CONTEXTS, bundle_path, module, write, smoke_protocol

previous = module("midweight_training", HERE.parent / "mcwm_context_mapping_v11/run.py")
for implementation in (previous, previous.previous):
    implementation.HERE, implementation.FIXED, implementation.bundle_path = HERE, FIXED, bundle_path
MappingTrainer, train, evaluate = previous.MappingTrainer, previous.train, previous.evaluate


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--arm", required=True, choices=P["arms"])
    parser.add_argument("--scenario", required=True, choices=CONTEXTS)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--evaluate", action="store_true")
    parser.add_argument("--case", type=int, default=10, choices=range(10, 20))
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    args.out = args.out.resolve()
    args.out.mkdir(parents=True, exist_ok=True)
    p = dict(P, arm=args.arm, training_seed=args.seed, relative_out=str(args.out.relative_to(HERE)))
    if args.smoke:
        assert "smoke" in args.out.parts
        p = smoke_protocol(p)
    if args.evaluate:
        evaluate(args, p)
    elif not (args.out / "results/ready.json").exists():
        write(args.out / "results/runtime.json", dict(host=platform.node(), torch_version=torch.__version__,
            cuda_version=torch.version.cuda, gpu_model=None, device=P["training_device"]))
        train(args, p)
    print("DONE fixed intermediate-weight comparison", flush=True)
