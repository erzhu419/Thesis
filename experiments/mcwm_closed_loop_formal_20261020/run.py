#!/usr/bin/env python3
"""Matched target-clock adaptation and common-stream prediction experiments."""

import argparse
import copy
from dataclasses import asdict, replace
import json
import os
from pathlib import Path
import platform
import sys
import time

for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(key, "1")
os.environ.setdefault("LIBSUMO_AS_TRACI", "quiet")
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "mcwm_closed_loop_v1"))
import numpy as np
import torch
from learning import (ClosedLoopTrainer, DomainReplay, OnlineWorld, TransitTrainerConfig,
                      learned_rows, load_table, row_table, targets)
from stream import EventStream, TransitMDPConfigV7, create_target_env
from scenarios import install_scenario


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2)+"\n")


def set_cpu_optimizer_runtime(learner):
    # These resident initializers were trained with CUDA-capturable fused Adam.
    # Preserve weights, moments, step counts and hyperparameters on CPU.
    for optimizer in (learner.actor_optimizer,learner.critic_optimizer,learner.value_optimizer,
                      learner.alpha_optimizer,learner.ratio.optimizer):
        if optimizer is not None:
            for group in optimizer.param_groups:
                group.update(fused=False,capturable=False)


class PacketBudget:
    """Release the same number of distinct outcomes in each prespecified clock window."""
    def __init__(self, p):
        self.p, self.rows, self.index, self.receipts = p, [], 0, []
        self.deadlines = list(range(p["feedback_first_deadline"], p["feedback_last_deadline"]+1,
                                    p["feedback_interval_seconds"]))

    def ingest(self, rows):
        self.rows.extend(rows)

    def due(self, now):
        return self.index < len(self.deadlines) and now >= self.deadlines[self.index]

    def take(self, now):
        deadline = self.deadlines[self.index]
        if now < deadline:
            raise ValueError("Feedback deadline has not arrived")
        lower = deadline-self.p["feedback_interval_seconds"]
        ready = [row for row in self.rows if row.available_at <= deadline]
        candidates = [row for row in ready if row.available_at > lower and row.dt > 0 and not row.terminal]
        n = self.p["feedback_records_per_packet"]
        if len(candidates) < n:
            raise RuntimeError(f"Insufficient feedback at {deadline}: {len(candidates)} eligible, {n} required")
        selected = sorted(candidates, key=lambda row: (row.available_at, row.sequence))[-n:]
        selected.sort(key=lambda row: (row.available_at, row.sequence))
        selected_ids = {row.sequence for row in selected}
        discarded = [row.sequence for row in ready if row.sequence not in selected_ids]
        self.rows = [row for row in self.rows if row.available_at > deadline]
        self.receipts.append(dict(deadline=deadline, processed_at=now, records=n,
            sequence_ids=[row.sequence for row in selected],
            max_available_at=max(row.available_at for row in selected)))
        self.index += 1
        return selected, discarded


class FormalReplay(DomainReplay):
    def __init__(self, table, *args, **kwargs):
        table = dict(table)
        self.history_weights = table.pop("sample_weight")
        self.history_weights = self.history_weights/self.history_weights.sum()
        super().__init__(table, *args, **kwargs)

    def sample(self, size, *, scope="offline", rng=None, **unused):
        if scope == "online":
            return super().sample(size, scope=scope, rng=rng)
        recent_size = size//2 if self.recent else 0
        ids = rng.choice(len(self.history["s"]), size=size-recent_size, p=self.history_weights)
        mapping = dict(observations="s",actions="a",next_observations="ns",rewards="r",
                       delta_t_seconds="dt",terminals="terminal")
        batch = {key: torch.as_tensor(self.history[value][ids],dtype=torch.float32,device=self.device)
                 for key,value in mapping.items()}
        batch["actions"] = batch["actions"].reshape(-1,1)
        for key in ("rewards","delta_t_seconds","terminals"):
            batch[key] = batch[key].reshape(-1)
        batch["discounts"] = torch.ones(len(ids),device=self.device)
        if recent_size:
            from learning import tensor_batch
            recent = tensor_batch(row_table(self.recent),recent_size,rng,self.device)
            batch = {key: torch.cat((batch[key],recent[key])) for key in batch}
        return batch


class ScoredWorld(OnlineWorld):
    name = "online"
    fixed_reference = False

    def register(self, issued):
        super().register(issued)
        if issued["sequence"] in self.pending:
            self.pending[issued["sequence"]]["family_at_issue"] = self.family

    def observe(self, rows, now):
        prior = {}
        for row in rows:
            prediction = self.pending.get(row.sequence)
            if prediction is not None:
                family = prediction["family_at_issue"]
                g = prediction["outputs"][family][0,:4]*self.norm.g_std+self.norm.g_mean
                physical = torch.expm1(g[:2].clamp(0,20)).tolist()
                prior[row.sequence] = (family,abs(physical[0]-row.dt),abs(physical[1]+row.r))
        initial_family = self.family
        enabled = self.context_enabled
        # Packet calibration is recomputed against the refitted model below.
        self.context_enabled = False
        try:
            logs = super().observe(rows,now)
        finally:
            self.context_enabled = enabled
        if self.fixed_reference:
            self.family = initial_family
        for log in logs:
            family,dt_mae,cost_mae = prior[log["sequence"]]
            log.update(model_arm=self.name,selected_family_at_issue=family,
                selected_score=log["scores"][family],duration_absolute_error=dt_mae,
                cost_absolute_error=cost_mae)
        return logs

    @torch.no_grad()
    def recenter_context(self, rows):
        if not self.context_enabled:
            return
        for row in rows:
            y = targets(row_table([row]), self.stations, self.device)
            standardized = (y["g"]-self.norm.g_mean)/self.norm.g_std
            for family,model in self.models.items():
                residual = (standardized-model(y["s"],y["a"])[:,:4])[0]
                self.context[family].lerp_(residual,.1)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode",choices=("control","prediction"),required=True)
    parser.add_argument("--arm",choices=("replay","online","context","common"),required=True)
    parser.add_argument("--seed",type=int,required=True)
    parser.add_argument("--scenario",choices=("stable","local_speed_drop"),required=True)
    parser.add_argument("--out",type=Path,required=True)
    parser.add_argument("--protocol",type=Path,default=HERE/"protocol.json")
    parser.add_argument("--preflight",action="store_true")
    parser.add_argument("--device",choices=("cpu","cuda"),default="cpu")
    return parser.parse_args()


def run(args):
    p = json.loads(args.protocol.read_text())
    if args.preflight:
        p.update(model_init_updates=2,model_refit_updates=1,policy_updates_per_packet=1,
            feedback_first_deadline=1200,feedback_last_deadline=1800,
            target_record_budget=15,policy_update_budget=3,shock_at_seconds=1350)
    if args.seed not in p["training_seeds"] or ((args.mode=="prediction") != (args.arm=="common")):
        raise ValueError("Invalid seed or mode/arm pairing")
    out = args.out.resolve()
    if out.exists() and any(out.iterdir()):
        raise FileExistsError(f"Refuse to overwrite run: {out}")
    out.mkdir(parents=True,exist_ok=True)
    started, last_progress = time.monotonic(), time.monotonic()
    torch.set_num_threads(1)
    def progress(stage,completed=0,total=24000,**extra):
        print("SCHEDULEURM_PROGRESS "+json.dumps(dict(current=completed,total=total,unit="sim_seconds",
            phase=stage,phase_index=1 if stage=="load_initializer" else 2 if stage=="initial_world_fit" else 3,
            phase_count=3,elapsed_seconds=time.monotonic()-started,**extra)),flush=True)
    progress("load_initializer")
    metadata = json.loads((HERE/"inputs/history_metadata.json").read_text())
    with np.load(HERE/"inputs/history.npz") as archive:
        history = {key:archive[key] for key in ("s","a","ns","r","dt","terminal","sample_weight")}
    source = load_table(HERE/"inputs/source_9101.npz",p["source_rows"])
    contract = TransitMDPConfigV7(**metadata["contract"])
    stations = np.asarray(metadata["station_max_by_line"],np.float32)
    def make_replay():
        return FormalReplay(history,source,contract,stations,args.device,p["recent_window"])
    initializer = HERE/"inputs"/f"h2oplus_seed{args.seed}.pt"
    payload = torch.load(initializer,map_location="cpu",weights_only=False)
    if payload["total_updates"] != p["initializer_updates"] or payload["online_critic_updates"] != 0:
        raise ValueError("Wrong formal initialization budget")
    if payload["extra"]["source_training_seed"] != args.seed:
        raise ValueError("Initializer seed mismatch")
    config = replace(TransitTrainerConfig(**payload["source_config"]),use_ratio_correction=False)
    del payload
    replay = make_replay()
    learner = ClosedLoopTrainer(replay,config=config,device=args.device,seed=args.seed)
    learner.load_offline_initializer(initializer)
    if args.device=="cpu":
        set_cpu_optimizer_runtime(learner)
    initial_updates = learner.total_updates
    world_by_name = {}
    if args.mode=="prediction" or args.arm!="replay":
        progress("initial_world_fit")
        devices = [torch.cuda.current_device()] if args.device=="cuda" else []
        with torch.random.fork_rng(devices=devices):
            torch.manual_seed(args.seed+20000)
            base_world = ScoredWorld(replay,seed=args.seed,updates=p["model_init_updates"])
        if args.mode=="prediction":
            world_by_name = {"frozen_model_reference":copy.deepcopy(base_world),
                             "online":base_world,"context":copy.deepcopy(base_world)}
            for name,world in world_by_name.items():
                world.name = name
                world.replay = make_replay()
                world.context_enabled = name=="context"
                world.fixed_reference = name=="frozen_model_reference"
        else:
            base_world.name,base_world.context_enabled = args.arm,args.arm=="context"
            world_by_name[args.arm] = base_world
            learner.world = base_world
    budget = PacketBudget(p)
    prediction_keys = list(world_by_name)
    write(out/"protocol.json",dict(arguments={**vars(args),"out":str(out),"protocol":str(args.protocol)},
        protocol=p,initializer=str(initializer),initial_updates=initial_updates,
        history={key:metadata[key] for key in ("selected_rows","terminal_rows","strata","seed","blocks")},
        source_rows=len(source["s"]),policy_config=asdict(config),prediction_models=prediction_keys,
        runtime=dict(python=platform.python_version(),torch=torch.__version__),
        context="After each packet fit, recenter four global residual outputs on that arrived packet; prevents clearing all batched context before it can affect future predictions"))
    def issue(event):
        for world in world_by_name.values():
            world.register(event)
    def policy(state):
        return learner.act(state[None],deterministic=True)[0]
    env = hook = stream = None
    total_packet_rows = rounds = decisions_after_update = 0
    try:
        env = create_target_env(os.environ["THESIS_SUMO_ROOT"],
            Path(os.environ["THESIS_H2O_ROOT"])/"bus_h2o/network_data/a_sorted_busline_edge.xml",
            episode_seconds=p["horizon_seconds"],seed=args.seed+p["target_seed_offset"])
        stream = EventStream(env,stations,contract,domain="target",delay_seconds=p["feedback_delay_seconds"],
                             record_every=1,on_issue=issue)
        stream.reset()
        hook = install_scenario(env,args.scenario,shock_at=p["shock_at_seconds"],
            factor=p["speed_factor"],metric_every=p["metric_interval_seconds"])
        native_version = hook.api.getVersion()[1]
        progress("target_loop",stream.now)
        with (out/"prequential.jsonl").open("w") as error_file:
            while not stream.done:
                previous_decisions = stream.issued_count
                budget.ingest(stream.step(policy))
                if rounds and args.mode=="control":
                    decisions_after_update += stream.issued_count-previous_decisions
                while budget.due(stream.now):
                    packet,discarded = budget.take(stream.now)
                    total_packet_rows += len(packet)
                    for world in world_by_name.values():
                        for sequence in discarded:
                            world.pending.pop(sequence,None)
                        logs = world.observe(packet,stream.now)
                        if len(logs)!=len(packet):
                            raise RuntimeError("A selected label lacks its saved pre-action prediction")
                        for log in logs:
                            error_file.write(json.dumps(log)+"\n")
                    error_file.flush()
                    if args.mode=="prediction":
                        for world in world_by_name.values():
                            if not world.fixed_reference:
                                world.replay.add_target(packet)
                                world.refit(p["model_refit_updates"])
                                world.recenter_context(packet)
                    else:
                        replay.add_target(packet)
                        for world in world_by_name.values():
                            world.refit(p["model_refit_updates"])
                            world.recenter_context(packet)
                        for _ in range(p["policy_updates_per_packet"]):
                            learner.continue_step()
                    rounds += 1
                    progress("target_loop",stream.now,packets=rounds,target_records=total_packet_rows,
                             policy_updates=learner.total_updates-initial_updates)
                    last_progress=time.monotonic()
                    write(out/"progress.json",dict(target_time_seconds=stream.now,packets=rounds,
                        target_records=total_packet_rows,policy_updates=learner.total_updates-initial_updates,
                        worlds={name:world.summary() for name,world in world_by_name.items()}))
                # Process records used only for diagnostic physical interval counts.
                stream.release_intervals()
                if time.monotonic()-last_progress>=45:
                    progress("target_loop",stream.now,packets=rounds,target_records=total_packet_rows)
                    last_progress=time.monotonic()
                if args.preflight and stream.now>=2100:
                    break
        hook.capture_final()
        snapshot=env._bridge.passenger_time_ledger.snapshot()
        if total_packet_rows != p["target_record_budget"]:
            raise RuntimeError(f"Target record budget differs: {total_packet_rows}")
        expected_updates=p["policy_update_budget"] if args.mode=="control" else 0
        if learner.total_updates-initial_updates != expected_updates:
            raise RuntimeError("Policy update budget differs")
        if not args.preflight and (not stream.done or snapshot.time_seconds != p["horizon_seconds"]):
            raise RuntimeError("Target did not complete its full horizon")
        from SimpleSAC.transit_v5.evaluation import summarize_passengers_v5
        from SimpleSAC.transit_v3.evaluation import summarize_service
        passengers=summarize_passengers_v5(env._bridge,snapshot)
        if passengers["active_state_count_balance_error"] != 0:
            raise RuntimeError("Passenger ledger balance failed")
        cost=2*snapshot.accrued_wait_seconds+snapshot.accrued_in_vehicle_seconds
        summary=dict(status="complete",mode=args.mode,arm=args.arm,scenario=args.scenario,seed=args.seed,
            preflight=args.preflight,full_horizon_completed=stream.done,target_time_seconds=stream.now,
            target_online_training_events=total_packet_rows,target_control_events=stream.issued_count,
            policy_updates=learner.total_updates-initial_updates,packets=rounds,
            decisions_after_policy_update=decisions_after_update,sumo_version=native_version,
            generalized_passenger_seconds=cost,
            generalized_seconds_per_departed_passenger=cost/passengers["departed_passengers"],
            passengers=passengers,service=summarize_service(env._bridge),
            worlds={name:world.summary() for name,world in world_by_name.items()},
            scenario_metadata=hook.metadata(),feedback_receipts=budget.receipts,
            wall_seconds=time.monotonic()-started)
        with (out/"metrics.jsonl").open("w") as metric_file:
            for metric in hook.metrics:
                metric_file.write(json.dumps(metric)+"\n")
        if args.mode=="control":
            learner.save_policy_artifact(out/"policy.pt",extra=dict(seed=args.seed,arm=args.arm,
                scenario=args.scenario,contract_config=asdict(contract),target_records=total_packet_rows))
        write(out/"summary.json",summary)
        progress("complete",stream.now,target_records=total_packet_rows)
        return summary
    except Exception as error:
        write(out/"failure.json",dict(status="failed",type=type(error).__name__,message=str(error),
            arm=args.arm,mode=args.mode,seed=args.seed,scenario=args.scenario,
            target_time_seconds=stream.now if stream is not None and stream.episode_id>=0 else None,
            target_records=total_packet_rows,wall_seconds=time.monotonic()-started))
        raise
    finally:
        if hook is not None:
            hook.close()
        if env is not None:
            env.close()


if __name__=="__main__":
    run(parse_args())
