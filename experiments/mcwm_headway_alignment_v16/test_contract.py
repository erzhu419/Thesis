"""Check historical reward parity and the concrete transition failure cases."""

import ast
import subprocess
from types import SimpleNamespace
import unittest

import numpy as np
import torch

from headway_common import reward, raw_reward, state_reward
from vehicle_data import bookmark,settle,Sampler
from test_contract import example_state


class Contracts(unittest.TestCase):
    def test_historical_reward_exactly_matches(self):
        root = "/home/erzhu419/mine_code/sumo-rl"
        text = subprocess.check_output(["git","-C",root,"show","3416538:SUMO_ruiguang/online_control/rl_env.py"],text=True)
        tree = ast.parse(text)
        cls = next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=="SumoBusHoldingEnv")
        function = next(n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name=="_compute_reward_linear")
        namespace = {"np":np,"DecisionEvent":object}
        exec(compile(ast.Module(body=[function],type_ignores=[]),"historical_reward","exec"),namespace)
        rng = np.random.default_rng(901)
        for _ in range(1000):
            f,b,tf,tb = rng.uniform(0,3000,2).tolist()+rng.uniform(60,1200,2).tolist()
            fp,bp = rng.integers(0,2,2).astype(bool)
            event = SimpleNamespace(forward_headway=f,backward_headway=b,target_forward_headway=tf,
                target_backward_headway=tb,forward_bus_present=fp,backward_bus_present=bp)
            expected = namespace["_compute_reward_linear"](None,event,tf)
            self.assertAlmostEqual(float(reward(f,b,tf,tb,fp,bp)),expected,places=9)

    def test_only_same_vehicle_later_station_closes(self):
        s = example_state()
        raw = np.array([0,2,900,360,360,360,360,0,10,0,0,1,1,50,0,10],np.float32)
        p = bookmark(s,np.array([-.5,.3]),raw)
        same_time = raw.copy()
        same_time[1] = 3
        with self.assertRaises(ValueError):
            settle(p,s,same_time)
        wrong_line = raw.copy()
        wrong_line[0],wrong_line[1],wrong_line[2] = 1,3,1200
        with self.assertRaises(ValueError):
            settle(p,s,wrong_line)
        next_raw = raw.copy()
        next_raw[1],next_raw[2],next_raw[3] = 3,1200,400
        row = settle(p,s,next_raw)
        self.assertTrue(np.array_equal(row[1],[-.5,.3]))
        self.assertEqual(row[5],300)
        self.assertAlmostEqual(row[3],raw_reward(next_raw))
        self.assertEqual(settle(p,s,next_raw,True)[3],0.)

    def test_perfect_headway_and_reward_threshold_continuity(self):
        self.assertAlmostEqual(float(state_reward(example_state()[None])[0]),0.,places=5)
        left,right = reward(540-1e-5,360,360,360),reward(540+1e-5,360,360,360)
        self.assertLess(abs(float(left-right)),1e-3)

    def test_simultaneous_other_vehicles_do_not_take_the_reward(self):
        raw = np.array([0,2,900,360,360,360,360,0,10,0,0,1,1,50,0,10],np.float32)
        b = raw.copy()
        b[0],b[3] = 1,700
        c = raw.copy()
        c[0],c[2] = 2,1000
        next_a = raw.copy()
        next_a[1],next_a[2],next_a[3] = 3,1200,400
        batches = [{("A",0):[raw],("B",1):[b]},{("C",2):[c]},{("A",0):[next_a]}]

        class Env:
            done = False
            def __init__(self):
                self.line_map = {line:SimpleNamespace(bus_all=[SimpleNamespace(bus_id=i,trip_id=0)])
                                 for i,line in enumerate(("A","B","C"))}
                self.index,self.commands = -1,[]
                self.state = {}
            def reset(self):
                pass
            def step_to_event(self,commands):
                self.commands.append(commands)
                self.index += 1
                self.state = batches[min(self.index,2)]

        class Replay:
            stations = np.full(12,46,np.float32)
            def __init__(self):
                self.rows = []
            def append(self,*row):
                self.rows.append(row)

        env,replay = Env(),Replay()
        sampler = Sampler(env,replay)
        def policy(s,deterministic=False):
            return np.array([[-.5,.3]],np.float32)
        sampler.collect(policy,2)
        self.assertEqual(len(replay.rows),0)
        self.assertTrue(np.allclose(env.commands[1]["A",0],[15.,1.06]))
        sampler.collect(policy,1)
        self.assertEqual(len(replay.rows),0)
        sampler.collect(policy,1)
        self.assertEqual(len(replay.rows),1)
        self.assertTrue(np.allclose(replay.rows[0][1],[-.5,.3]))
        self.assertEqual(replay.rows[0][5],300.)
        self.assertAlmostEqual(replay.rows[0][3],raw_reward(next_a))


if __name__=="__main__":
    torch.set_num_threads(1)
    unittest.main()
