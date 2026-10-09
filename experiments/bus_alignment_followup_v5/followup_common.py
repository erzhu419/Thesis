"""One independent follow-up; completed policy endpoints remain resident."""

from itertools import product
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
EXPERIMENTS = HERE.parent
P = json.loads((HERE / "protocol.json").read_text())


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")
    temporary.replace(path)


def label(item):
    prefix = "smoke" if item.get("smoke") else "formal"
    if item["family"] == "mc":
        return f"{prefix}/mc/{item['variant']}/{item['scene']}/{item['arm']}/{item['seed']}"
    return f"{prefix}/{item['family']}/{item['mode']}/{item['objective']}/{item['arm']}/{item['seed']}"


def folder(item):
    return HERE / label(item)


def policies(smoke=False):
    seeds = P["cs_seeds"][:1] if smoke else P["cs_seeds"]
    for mode, objective, seed in product(("holding", "dual"), P["cs_objectives"], seeds):
        for arm in (["pristine_mlp"] if smoke else P["cs_recipe_arms"]):
            yield dict(family="cs_recipe", mode=mode, objective=objective, arm=arm, seed=seed, smoke=smoke)
    for objective, seed in product(P["cs_objectives"], seeds):
        for arm in (P["cs_context_arms"][:2] if smoke else P["cs_context_arms"]):
            yield dict(family="cs_context", mode="dual", objective=objective, arm=arm, seed=seed, smoke=smoke)
    for variant, scene, arm, seed in product(P["mc_variants"], P["mc_scenes"][:1] if smoke else P["mc_scenes"],
            P["mc_arms"], P["mc_seeds"][:1] if smoke else P["mc_seeds"]):
        yield dict(family="mc", variant=variant, scene=scene, objective="headway", arm=arm, seed=seed, smoke=smoke)


def needs_training(item):
    return item["family"] == "mc" or item["arm"] in ("pristine_mlp", "context_symbolic", "context_control")


def parent(item):
    if needs_training(item):
        return folder(item)
    arm = dict(stabilized_mlp="legacy_mlp", stabilized_nau="legacy_nau",
               constant_symbolic="smooth_soft", constant_control="soft_no_symbolic")[item["arm"]]
    if item["mode"] == "dual":
        kind = "cs_native" if item["family"] == "cs_recipe" else "cs_symbolic"
        return EXPERIMENTS / "bus_legacy_dual_v1/runs" / kind / "dual" / item["objective"] / f"{arm}_seed{item['seed']}"
    arm = "csbapr_no_nau" if item["arm"] == "stabilized_mlp" else "csbapr"
    return EXPERIMENTS / "reward_cross_v1/runs/cs" / item["objective"] / "normal" / f"{arm}_seed{item['seed']}"


def evaluation_cases(item):
    if item.get("smoke"):
        return [("guarded", 0)] if item["family"] == "mc" else [("native", 0)]
    return list(product(("direct", "guarded"), range(10))) if item["family"] == "mc" else [("native", c) for c in range(18)]


def speed_controls():
    for arm in ("zero_neutral", "zero_fast"):
        yield dict(family="cs_recipe", mode="dual", objective="headway", arm=arm, seed=0, smoke=False)
