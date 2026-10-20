"""Collect the existing compact v4 summary, preserving its missing-before status."""

import gzip
import json
from pathlib import Path
import shlex
import subprocess
import sys

HERE = Path(__file__).resolve().parent
V4 = HERE.parent / "mcwm_closed_loop_head_diagnostic_v4_20261020"
sys.path.insert(0, str(V4))
import infra


def main():
    script = '''import gzip,json
from pathlib import Path
source=Path(__SOURCE__)
out=Path(__OUT__)
value=json.loads(source.read_text())
integrity=value['integrity']
errors=integrity['errors']
expected=lambda e: e.get('scope')=='fixed_clock_cost' and e.get('window')=='before' and e.get('status')=='missing_endpoint' and e.get('missing_times')==[0.0]
assert integrity['complete_new_runs']==integrity['expected_new_runs']==20
assert all(expected(e) for e in errors), 'Unexpected integrity error; stop analysis'
assert all(x['exact_cost_and_budget'] and x['cost_delta']==0 for x in value['passive_reproduction'])
assert len(value['passive_reproduction'])==8
cells=value['new_control_cells']+value['stability_control_cells']
assert len(value['new_control_cells'])==20 and len(value['stability_control_cells'])==30
assert all(c['windows'][w]['status']=='complete' for c in cells for w in ('adaptation','post_training_frozen'))
out.parent.mkdir(parents=True,exist_ok=True)
with gzip.open(out,'wb') as f: f.write(source.read_bytes())
print(json.dumps(dict(source=str(source),original_status=value['status'],complete_new_runs=20,exact_reproductions=8,
 before_missing_endpoint_records=len(errors),requested_cost_windows_complete=True,raw_bytes=source.stat().st_size,
 transferred_bytes=out.stat().st_size,remote_archive=str(out))))
'''.replace("__SOURCE__", repr(infra.remote(V4 / "head_diagnostic_summary.json"))).replace(
        "__OUT__", repr(infra.remote(HERE / "v4_summary.json.gz")))
    _, output, _ = infra.scheduler.run_on(
        "node001", shlex.join([infra.PYTHON, "-c", script]), timeout=45, check=True)
    receipt = json.loads(output)
    subprocess.run([
        "rsync", "-a", "--timeout=30", "-e", infra.scheduler._ssh_rsync_shell_for_node("node001"),
        infra.scheduler._ssh_target_for_node("node001") + ":" + receipt["remote_archive"],
        str(HERE / "v4_summary.json.gz"),
    ], check=True, timeout=60)
    with gzip.open(HERE / "v4_summary.json.gz", "rb") as archive:
        (HERE / "v4_summary.json").write_bytes(archive.read())
    (HERE / "collection_receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps(receipt))


if __name__ == "__main__":
    main()
