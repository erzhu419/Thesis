"""Reuse the established resident import setup for v40 read-only inference."""
from pathlib import Path
import sys
HERE = Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parents[1]))
from experiments.mcwm_closed_loop_action_local_v3_20261020.bootstrap import formal, write
