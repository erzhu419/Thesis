"""Synthetic saved-state XML checks; no SUMO import or simulation."""
from pathlib import Path
import tempfile
import unittest
import xml.etree.ElementTree as ET

from experiments.mcwm_closed_loop_future_rng_v102_20261028.native_probe import read_snapshot


class SnapshotPartitionTest(unittest.TestCase):
    def setUp(self):
        self.scratch = tempfile.TemporaryDirectory()
        self.addCleanup(self.scratch.cleanup)
        self.path = Path(self.scratch.name) / 'state.xml'

    def snapshot(self):
        root = ET.Element('snapshot', time='20.00', version='1.25.0')
        rng = ET.SubElement(root, 'rngState', default='17', routeHandler='23')
        for index in range(64):
            ET.SubElement(rng, 'rngLane', index=str(index), state=str(index + 1))
        ET.SubElement(root, 'vehicle', id='bus', pos='12.25', speed='3.5')
        return root

    def read(self, root):
        ET.ElementTree(root).write(self.path)
        return read_snapshot(self.path)

    def test_nested_lane_replacement_does_not_mask_other_rng(self):
        root = self.snapshot()
        before = self.read(root)
        for lane in root.find('rngState'):
            lane.set('state', '1000000 future_engine_' + lane.get('index'))
        after = self.read(root)
        self.assertEqual(len(after['lane']), 64)
        self.assertNotEqual(before['lane'], after['lane'])
        self.assertEqual(before['other_rng'], after['other_rng'])
        self.assertEqual(before['physical'], after['physical'])
        self.assertEqual(before['other_rng'][0][1],
                         (('default', '17'), ('routeHandler', '23')))

    def test_other_rng_and_vehicle_changes_remain_detectable(self):
        root = self.snapshot()
        before = self.read(root)
        root.find('rngState').set('default', '18')
        other_changed = self.read(root)
        self.assertNotEqual(before['other_rng'], other_changed['other_rng'])
        self.assertEqual(before['physical'], other_changed['physical'])
        self.assertEqual(before['lane'], other_changed['lane'])
        root.find('vehicle').set('pos', '12.50')
        vehicle_changed = self.read(root)
        self.assertNotEqual(other_changed['physical'], vehicle_changed['physical'])
        self.assertEqual(other_changed['other_rng'], vehicle_changed['other_rng'])
        self.assertEqual(other_changed['lane'], vehicle_changed['lane'])


if __name__ == '__main__':
    unittest.main()
