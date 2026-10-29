"""Resident runtime; all v43 outputs stay in the new experiment directory."""
from pathlib import Path
import sys
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))
from experiments.mcwm_closed_loop_action_local_v3_20261020.bootstrap import formal
