import ast
from pathlib import Path
import unittest

import numpy as np

from rules import hold_seconds


class FixedRulesTest(unittest.TestCase):
    def test_matches_transitduet_formula(self):
        source = Path(__file__).resolve().parents[3] / "TransitDuet/transit_duet/run_baseline_rule.py"
        tree = ast.parse(source.read_text())
        function = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "rule_holding_action")
        namespace = dict(np=np, HMAX_HOLD=60.0)
        exec(compile(ast.Module(body=[function], type_ignores=[]), str(source), "exec"), namespace)
        original = namespace["rule_holding_action"]
        for front in (0, 60, 240, 360, 480, 2400):
            for back in (0, 60, 240, 360, 480, 2400):
                expected = original([0, 0, 0, 0, front, back], 360, "daganzo")
                self.assertEqual(hold_seconds("daganzo_duet", front, back, 360, 360, 1, 1), expected)

    def test_not_equivalent_with_unequal_targets(self):
        self.assertEqual(hold_seconds("daganzo_duet", 300, 380, 300, 420, 1, 1), 40)
        adjusted = max(0, min(60, .5 * ((380 - 420) - (300 - 300))))
        self.assertEqual(adjusted, 0)

    def test_missing_neighbor_uses_current_target(self):
        self.assertEqual(hold_seconds("daganzo_duet", 999, 400, 300, 420, 0, 1), 50)
        self.assertEqual(hold_seconds("daganzo_duet", 300, 999, 300, 420, 1, 0), 60)
        self.assertEqual(hold_seconds("forward_proportional", 999, 400, 300, 420, 0, 1), 0)

    def test_forward_gain_and_action_limits(self):
        self.assertEqual(hold_seconds("forward_proportional", 300, 400, 360, 360, 1, 1), 36)
        self.assertEqual(hold_seconds("forward_proportional", 500, 400, 360, 360, 1, 1), 0)
        self.assertEqual(hold_seconds("forward_proportional", 0, 400, 360, 360, 1, 1), 60)


if __name__ == "__main__":
    unittest.main()
