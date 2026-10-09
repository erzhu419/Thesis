"""Stage just the Python runtime, SUMO inputs, and two dataset attributes."""

import shutil
import xml.etree.ElementTree as ET
from pathlib import Path

import h5py

HERE = Path(__file__).resolve().parent
MCWM = HERE.parent / "mcwm_bus"
SOURCE = HERE.parents[2] / "sumo-rl"
H2O = SOURCE / "H2Oplus"
SUMO = SOURCE / "SUMO_ruiguang"


def copy(source, target):
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, target)


for name in ("evaluate_control.py", "data.py", "protocol_v1.json"):
    copy(MCWM / name, HERE / name)
for folder in ("SimpleSAC", "bus_h2o", "collect_policy"):
    for source in (H2O / folder).rglob("*.py"):
        copy(source, HERE / "runtime/H2Oplus" / source.relative_to(H2O))
copy(H2O / "reproducibility/build_passenger_scenarios.py",
     HERE / "runtime/H2Oplus/reproducibility/build_passenger_scenarios.py")
copy(H2O / "bus_h2o/network_data/a_sorted_busline_edge.xml",
     HERE / "runtime/H2Oplus/bus_h2o/network_data/a_sorted_busline_edge.xml")
cfg = SUMO / "online_control/control_sim_traci_period.sumocfg"
copy(cfg, HERE / "assets/SUMO_ruiguang" / cfg.relative_to(SUMO))
for element in ET.parse(cfg).getroot().find("input"):
    if element.tag not in ("net-file", "additional-files", "route-files"):
        continue
    for name in element.attrib["value"].split(","):
        source = (cfg.parent / name.strip()).resolve()
        copy(source, HERE / "assets/SUMO_ruiguang" / source.relative_to(SUMO))
for folder, pattern in (("initialize_obj", "*.xml"), ("sim_obj", "*.py")):
    for source in (SUMO / "online_control" / folder).rglob(pattern):
        copy(source, HERE / "assets/SUMO_ruiguang" / source.relative_to(SUMO))
with h5py.File(H2O / "bus_h2o/datasets_v7/formal_v2_full/transit_smdp_v7_formal_v2.h5") as source:
    with h5py.File(HERE / "contract.h5", "w") as target:
        for key in ("config", "station_max_by_line"):
            target.attrs[key] = source.attrs[key]
size = sum(p.stat().st_size for p in HERE.rglob("*") if p.is_file())
print(f"Evaluation pack: {size / 1024**2:.1f} MiB; no replay or full checkpoints")
