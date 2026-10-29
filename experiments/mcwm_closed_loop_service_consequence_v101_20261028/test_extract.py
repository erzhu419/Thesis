"""Synthetic persisted-log checks; no simulator or historical outcomes are read."""
import copy
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

spec = importlib.util.spec_from_file_location('service_extract',Path(__file__).with_name('extract.py'))
e = importlib.util.module_from_spec(spec)
spec.loader.exec_module(e)


def point(now, wait=0, ride=0, waiting=2, onboard=3):
    return dict(time_seconds=now,accrued_wait_seconds=wait,
        accrued_in_vehicle_seconds=ride,waiting_passengers=waiting,in_vehicle_passengers=onboard)


def row(sequence, now, station='A', index=1, trip='bus', line='7X', onboard=3, stranded=2, action=.1):
    event = [0]*16
    event[1],event[2],event[7],event[8],event[10] = index,now-1,stranded,onboard,8
    return dict(sequence=sequence,started_at=now,trip_id=trip,line_id=line,station_id=station,
        issue_ledger=point(now),event=event,a=[action],s=[1,2],raw_s=[1,2],proposed_a=[action])


def summary(root, final=(100,200), tail=None):
    initial = root['issue_ledger']
    result = dict(root=copy.deepcopy(root),remaining=dict(
        wait=final[0]-initial['accrued_wait_seconds'],
        ride=final[1]-initial['accrued_in_vehicle_seconds'],observed_at=24000.),
        status='complete',integrity={'all_pass':True},final_clock=24000.,executed_hold_seconds=4,
        physical=dict(departure={'clock_seconds':root['started_at']+4},
            next_same_vehicle_arrival={'clock_seconds':root['started_at']+100}))
    if tail is not None:
        result['tail_boundary'] = tail
    return result


class ExtractTests(unittest.TestCase):
    def test_actual_common_clocks_and_missing_windows(self):
        ref,cand = e.Trace(),e.Trace()
        common = [12000,12304,12961,13830,15600,18300,20100,24000]
        for i,now in enumerate(common):
            ref.add_ledger(now,dict(cost=e.cost(i*100,i*50),counts=[2,3]))
            cand.add_ledger(now,dict(cost=e.cost(i*100+i,i*50-i),counts=[2+(i>0),3]))
        # Neither arm's nearby private clock can form a paired observation.
        ref.add_ledger(12905,dict(cost=e.cost(50,50),counts=[2,3]))
        cand.add_ledger(12906,dict(cost=e.cost(51,50),counts=[2,3]))
        result = e.decompose(ref,cand,12000)
        self.assertEqual(result['common_clocks'],8)
        self.assertEqual(result['first_common_divergence'],12304)
        self.assertEqual(result['first_common_cost_divergence'],12304)
        self.assertEqual(result['anchors'][0][:3],[12300.,12304,4.])
        self.assertEqual(result['anchors'][1],[12900.,None,None,None,None,None])
        self.assertEqual(result['anchors'][2][:3],[13800.,13830,30.])
        self.assertEqual(result['anchors'][2][4],e.cost(2,-2))
        self.assertEqual(result['interval_sum'],e.cost(7,-7))

    def test_finish_uses_true_endpoint_and_does_not_invent_terminal_inventory(self):
        root = row(5,12000)
        root['issue_ledger'] = point(12000,1000,2000)
        source = summary(root,(1700,2900),point(18300,1500,2600,8,9))
        trace = e.Trace()
        trace.add_ledger(23990,dict(cost=e.cost(1690,2890),counts=[20,1]))
        trace.finish(source)
        self.assertEqual(trace.ledgers[24000]['cost'],e.cost(1700,2900))
        self.assertEqual(trace.ledgers[24000]['counts'],[None,None])
        self.assertEqual(trace.ledgers[18300]['counts'],[8,9])
        self.assertEqual(trace.conflicts,0)

    def test_duplicate_station_visits_missing_sides_and_subsequent_actions(self):
        root = row(10,12000)
        ref,cand = e.Trace(),e.Trace()
        ref_rows = [root,row(11,12100,'B',2),row(12,12200,'A',3),
            row(13,12300,'B',2),row(14,12400,'C',4),
            row(15,12500,'only_ref',5,trip='other',line='102X')]
        cand_rows = [copy.deepcopy(root),row(11,12104,'B',2,onboard=4,stranded=1,action=.2),
            row(12,12204,'A',3),row(13,12304,'B',2),row(14,12404,'C',4),
            row(15,12500,'only_cand',5,trip='other',line='705X')]
        # The substituted root command is excluded from subsequent action counts.
        cand_rows[0]['a'] = [.9]
        for value in ref_rows:
            ref.add(value)
        for value in cand_rows:
            cand.add(value)
        result = e.match_events(ref,cand,root)
        self.assertEqual(result['root_vehicle_matched_events'],4)
        self.assertEqual([value[2:5] for value in result['root_next_three']],
            [['B',2,1],['A',3,1],['B',2,2]])
        line = result['by_line']['7X']
        self.assertEqual(line['matched'],4)
        self.assertEqual(line['issue_shift'],[16,4.,4,4,4])
        self.assertEqual(line['onboard'],[1,1])
        self.assertEqual(line['stranded'],[-1,1])
        self.assertEqual(line['action_changes'],1)
        self.assertEqual(line['first_action'],[12100,12104,.1,.2])
        self.assertEqual(result['by_line']['102X']['missing_candidate'],1)
        self.assertEqual(result['by_line']['705X']['missing_reference'],1)
        self.assertEqual(result['first_network_change'][5:7],[12100,12104])

    def reference_fixture(self, directory, groups):
        paths,summaries = [],[]
        for i,rows in enumerate(groups):
            path = Path(directory)/f'{i}.jsonl'
            path.write_text(''.join(json.dumps(value)+'\n' for value in rows))
            paths.append(path)
            summaries.append(summary(rows[0]))
        return e.reference_group(paths,summaries)

    def test_reference_compares_all_fields_after_common_start(self):
        rows = [row(n,12000+n) for n in range(1,7)]
        groups = [copy.deepcopy(rows),copy.deepcopy(rows[1:]),copy.deepcopy(rows[2:])]
        with tempfile.TemporaryDirectory() as directory:
            _,result = self.reference_fixture(directory,groups)
            self.assertTrue(result['all_pass'])
            self.assertEqual(result['compared_rows'],4)
            self.assertEqual(result['common_start_sequence'],3)
            groups[2][1]['raw_s'][0] = 999
            _,result = self.reference_fixture(directory,groups)
        self.assertFalse(result['all_pass'])
        self.assertEqual(result['unequal_rows'],1)
        self.assertEqual(result['first_difference'],{'sequence':4,'fields':['raw_s']})
        self.assertTrue(result['same_endpoint'])

    def test_unshared_eof_is_not_reported_as_field_divergence(self):
        rows = [row(n,12000+n) for n in range(1,7)]
        with tempfile.TemporaryDirectory() as directory:
            _,result = self.reference_fixture(directory,[rows,rows[:-2],rows])
        self.assertFalse(result['all_pass'])
        self.assertEqual(result['unequal_rows'],0)
        self.assertIsNone(result['first_difference'])
        self.assertTrue(result['same_endpoint'])
        self.assertEqual(result['unshared_tail_rows'],{'7X':2,'102X':0,'705X':2})
        self.assertEqual(result['last_common_sequence'],4)

    def test_same_clock_conflicts_and_nonexact_saved_tail_remain_visible(self):
        trace = e.Trace()
        trace.add_ledger(12000,dict(cost=e.cost(1,2),counts=[3,4]))
        trace.add_ledger(12000,dict(cost=e.cost(2,2),counts=[3,5]))
        self.assertEqual(trace.conflicts,2)
        self.assertEqual(trace.first_conflict,{'kind':'same_clock_cost_conflict','time':12000})
        trace = e.Trace()
        trace.finish(summary(row(1,12000),tail=point(18301)))
        self.assertEqual(trace.first_conflict,{'kind':'nonexact_saved_tail','time':18301})
        self.assertNotIn(18300,trace.ledgers)

    def test_pair_cost_is_endpoint_difference_and_interval_increments_telescope(self):
        root = row(1,12000)
        low,high = summary(root,(100,200)),summary(root,(103,198))
        ref,cand = e.Trace(),e.Trace()
        ref.add(root)
        cand.add(root)
        ref.finish(low)
        cand.finish(high)
        result = e.summarize_pair(ref,cand,low,high)
        self.assertTrue(result['all_pass'])
        self.assertEqual(result['D'],[3,-2,4])
        self.assertEqual(result['decomposition']['anchors'][-1],
            [24000.,24000.,0.,[3,-2,4],[3.,-2.,4.],None])


if __name__ == '__main__':
    unittest.main()
