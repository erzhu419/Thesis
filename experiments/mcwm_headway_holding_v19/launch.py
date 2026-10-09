"""Reuse the existing independent-task launch pattern."""

from pathlib import Path

HERE = Path(__file__).resolve().parent
source = HERE.parent / "mcwm_action_matrix_v18/launch.py"
# The launcher uses its file location for staging and isolated output paths.
code = compile(source.read_text(), str(source), "exec")
exec(code, dict(__name__="__main__", __file__=str(HERE / "launch.py")))
