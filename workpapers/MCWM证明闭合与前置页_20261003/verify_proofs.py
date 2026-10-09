"""Compile and audit the Appendix E module without importing the old scaffold."""
import json
from pathlib import Path
import re
import subprocess

HERE = Path(__file__).resolve().parent
source = HERE / 'MCWMClosed.lean'
completed = subprocess.run(
    ['/home/erzhu419/.elan/bin/lake', 'env', 'lean', str(source)],
    cwd='/home/erzhu419/mine_code/proof', text=True, capture_output=True,
)
log = completed.stdout + completed.stderr
(HERE / 'lean_closed.log').write_text(log)
entries = re.findall(r"'([^']+)' depends on axioms: \[([^\]]*)\]", log)
allowed = {'propext', 'Classical.choice', 'Quot.sound'}
extra = {name: [v.strip() for v in axioms.split(',') if v.strip() not in allowed]
         for name, axioms in entries}
extra = {name: values for name, values in extra.items() if values}
report = {'exit_code': completed.returncode, 'audited_roots': len(entries),
          'custom_or_placeholder_dependencies': extra,
          'source_placeholder_declarations': re.findall(r'^\s*(?:axiom|opaque|sorry|admit)\b.*$', source.read_text(), re.M),
          'historical_project_modified': False,
          'hypotheses': 'Finite kernels, valid fixed policy, discount below one; full update contraction is an explicit premise.'}
(HERE / 'proof_status.json').write_text(json.dumps(report, ensure_ascii=False, indent=2))
print(json.dumps(report, ensure_ascii=False))
assert completed.returncode == 0 and len(entries) == source.read_text().count('#print axioms')
assert not extra and not report['source_placeholder_declarations']
