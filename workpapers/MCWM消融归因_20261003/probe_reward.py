"""Compare two existing reward estimates using resident headway model artifacts."""

import inspect
import json
from pathlib import Path
import shlex
import sys

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]


def probe(root):
    import json
    import os
    from pathlib import Path
    import sys
    import h5py
    import numpy as np
    import torch

    root = Path(root)
    experiments = root / "experiments"
    runtime = json.loads((experiments / "bus_control_v2/node_runtime.json").read_text())["train"]
    os.environ.update(runtime["env"])
    sys.path[:0] = [runtime["env"]["THESIS_H2O_ROOT"], runtime["env"]["THESIS_MCWM_ROOT"],
                    str(experiments / "bus_dual_aligned_v4")]
    from dual_model import Models, Model, Normalizer, targets
    from aligned_common import ridge_of_state

    torch.set_num_threads(4)
    with h5py.File(experiments / "bus_dual_aligned_v4/data/dataset.h5") as data:
        n = len(data["actions"])
        selected = np.linspace(0, n-1, min(n, 200000), dtype=int)
        roles = data["role"][:]
        dt, terminal = data["delta_t_seconds"][:], data["terminals"][:]
        eligible = selected[(roles[selected] == 9) & (dt[selected] > 0) & (terminal[selected] < .5)]
        ids = eligible[np.linspace(0, len(eligible)-1, min(512, len(eligible)), dtype=int)]
        stations = torch.as_tensor(json.loads(data.attrs["stations"]), dtype=torch.float32)
        batch = {k:torch.as_tensor(data[k][ids], dtype=torch.float32) for k in
                 ("observations", "actions", "next_observations", "delta_t_seconds")}
        batch["rewards"] = torch.as_tensor(data["headway_rewards"][ids])
    assert len(ids) > 0
    assert np.max(np.abs(ridge_of_state(batch["next_observations"].numpy())-batch["rewards"].numpy())) < .01
    truth = batch["rewards"]
    def error(prediction):
        diff = prediction-truth
        return dict(mae=float(diff.abs().mean()), rmse=float(diff.square().mean().sqrt()),
            mean_prediction=float(prediction.mean()), mean_truth=float(truth.mean()), mean_bias=float(diff.mean()))
    results = []
    with torch.no_grad():
        for scene in ("calibrated", "joint75"):
            for seed in (66001, 66002):
                model_path = experiments / "bus_alignment_followup_v5/formal/mc/clip10" / scene / "mcwm" / str(seed) / "models/models.pt"
                payload = torch.load(model_path, map_location="cpu", weights_only=False)
                aug = Models.__new__(Models)
                aug.stations, aug.family, aug.banks = stations, payload["family"], {}
                aug.norm = Normalizer(targets(batch, stations))
                aug.norm.load_state_dict(payload["normalizer"])
                for family in ("direct", "residual"):
                    aug.banks[family] = []
                    for state in payload["banks"][family]:
                        model = Model(Model() if family == "residual" else None)
                        model.load_state_dict(state)
                        aug.banks[family].append(model.eval().requires_grad_(False))
                aug.generator = torch.Generator().manual_seed(90201)
                predictions = aug.predict(batch["observations"], batch["actions"])
                decoded_rewards, reconstructed_rewards = [], []
                for _ in range(4):
                    ns, reward, _ = aug.decode(batch["observations"], predictions)
                    decoded_rewards.append(reward)
                    reconstructed_rewards.append(torch.as_tensor(ridge_of_state(ns.numpy())))
                results.append(dict(scene=scene, seed=seed, selected_family=aug.family, rows=len(ids),
                    reward_head=error(torch.stack(decoded_rewards).mean(0)),
                    state_reconstruction=error(torch.stack(reconstructed_rewards).mean(0)),
                    reconstruction_draw_mean_mae=float(torch.stack([(r-truth).abs().mean() for r in reconstructed_rewards]).mean()),
                    model_path=str(model_path)))
    return dict(cohort="bus_alignment_followup_v5/clip10/headway", role=9, results=results,
        scope="Same calibration rows and logged actions, four diagnostic draws; not fresh generalization or control performance.",
        source=str(root), weights_or_datasets_downloaded=False)


if __name__ == "__main__":
    sys.path.insert(0, str(ROOT.parent / "scheduleurm/skill"))
    import scheduler
    remote = "/home/zhengliang01/scheduleurm_work/Thesis"
    program = inspect.getsource(probe)+f"\nimport json\nprint(json.dumps(probe({remote!r})))"
    python = "/home/zhengliang01/scheduleurm_work/conda_envs/csbapr-gpu-py310/bin/python"
    _, output, _ = scheduler.run_on("node007", shlex.quote(python)+" -c "+shlex.quote(program), timeout=180, check=True)
    result = json.loads(output)
    (HERE / "headway_reward_probe.json").write_text(json.dumps(result, ensure_ascii=False, indent=2)+"\n")
    print(json.dumps(result, ensure_ascii=False, indent=2))
