"""Replace only the resident SUMO 1.25.0 lane RNGs in a fork child."""
from __future__ import annotations

import ctypes
from pathlib import Path
import subprocess


BURN_PER_LANE = 1_000_000
LOADER_SYMBOL = "_ZN6MSLane12loadRNGStateEiRKSs"


def build_adapter(output_path: str | Path) -> Path:
    """Compile the small adapter; this does not import or start SUMO."""
    output = Path(output_path).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    source = Path(__file__).with_name("native_rng.cpp")
    subprocess.run(
        ["g++", "-std=c++11", "-O2", "-shared", "-fPIC",
         "-D_GLIBCXX_USE_CXX11_ABI=0", str(source), "-o", str(output), "-ldl"],
        check=True,
    )
    return output


class LaneRNG:
    """Bind an adapter to the already loaded libsumo shared object.

    ``apply(None)`` validates the native entry point and leaves original RNGs
    untouched. An integer seed replaces each lane pool with an independently
    seeded standard MT19937 after exactly one million discarded draws. Other
    SUMO streams, physical state, subscriptions and Python objects are untouched.
    """

    def __init__(self, adapter_path: str | Path, libsumo_library: str | Path,
                 lane_count: int = 64):
        if type(lane_count) is not int or lane_count <= 0:
            raise ValueError("lane_count must be a positive integer")
        self.lane_count = lane_count
        self.library = str(Path(libsumo_library).resolve()).encode()
        self.adapter = ctypes.CDLL(str(Path(adapter_path).resolve()))
        self.function = self.adapter.mcwm_apply_lane_rng
        self.function.argtypes = [ctypes.c_char_p, ctypes.c_int, ctypes.c_int,
                                  ctypes.c_uint32, ctypes.c_char_p, ctypes.c_size_t]
        self.function.restype = ctypes.c_int

    def apply(self, seed: int | None = None) -> dict:
        """Run in prepare_child, after fork and before the first env.step."""
        if seed is not None and (
            type(seed) is not int or not 0 <= seed <= 2**31 - self.lane_count
        ):
            raise ValueError("seed plus lane index must fit SUMO's nonnegative int range")
        error = ctypes.create_string_buffer(2048)
        result = self.function(self.library, self.lane_count, int(seed is None),
                               0 if seed is None else seed, error, len(error))
        if result != 0:
            raise RuntimeError(error.value.decode())
        return {
            "mode": "identity" if seed is None else "lane_rng_replacement",
            "seed": seed,
            "lane_count": self.lane_count,
            "burn_per_lane": 0 if seed is None else BURN_PER_LANE,
            "loader_symbol": LOADER_SYMBOL,
            "other_native_streams": "unchanged",
        }
