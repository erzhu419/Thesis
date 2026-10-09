"""One independent training or evaluation shard."""

import argparse
import os

import torch

from common import P

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--kind", choices=P["arms"], required=True)
    parser.add_argument("--phase", choices=("train", "evaluate"), required=True)
    parser.add_argument("--mode", choices=P["modes"], default="dual")
    parser.add_argument("--objective", choices=P["objectives"], required=True)
    parser.add_argument("--arm", required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--case", type=int, default=0)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    threads = P["threads"][args.kind] if args.phase == "train" else 1
    os.environ["OMP_NUM_THREADS"] = str(threads)
    torch.set_num_threads(threads)
    if args.kind == "resac":
        from resac_runner import run
    else:
        from cs_runner import run
    run(args)
    print("DONE legacy bus dual", vars(args), flush=True)
