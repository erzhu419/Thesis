from copy import deepcopy
import unittest

from experiments.mcwm_full_receiver_observation_v127_20261029.selection import bind_selection


def fixture():
    manifest = [dict(meta=dict(source_group='fixed/stable', sequence=i,
        observed_at=float(1000+i), receiver='bus', station='stop'),
        selection=dict(sequence=i, window_start=float(1000+i), window_end=float(1060+i),
            feedback_available_at=float(1090+i), packet_deadline=1200)) for i in (7,8)]
    records = [dict(context=dict(meta=deepcopy(r['meta'])), selection=deepcopy(r['selection']))
               for r in manifest]
    case = dict(roots=2, packets=1, final_root=deepcopy(manifest[-1]['meta']),
                stop_before_apply_clock=1008., stop_before_apply_sequence=8)
    return manifest, records, case


class SelectionTests(unittest.TestCase):
    def test_full_selection_is_sorted_bound_and_unchanged(self):
        manifest, records, case = fixture()
        manifest.reverse()
        original = deepcopy((manifest, records, case))
        targets, contexts = bind_selection(manifest, records, case)
        self.assertEqual(list(targets), [7,8])
        self.assertEqual(targets[8]['issue_clock'], 1008.)
        self.assertEqual(contexts[8]['meta'], case['final_root'])
        self.assertEqual((manifest, records, case), original)

    def test_duplicate_mismatched_or_wrong_final_root_cannot_launch(self):
        manifest, records, case = fixture()
        with self.assertRaises(ValueError):
            bind_selection(manifest, [records[0], records[0]], case)
        records[0]['context']['meta']['source_group'] = 'fixed/local_speed_drop'
        with self.assertRaises(ValueError):
            bind_selection(manifest, records, case)
        manifest, records, case = fixture()
        case['stop_before_apply_sequence'] = 7
        with self.assertRaises(ValueError):
            bind_selection(manifest, records, case)
