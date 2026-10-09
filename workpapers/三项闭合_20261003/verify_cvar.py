"""Compile the measure-space CVaR proof and check recursive axiom dependencies."""
from pathlib import Path
import json
import re
import subprocess

here = Path(__file__).resolve().parent
source = here / 'CVaRMeasure.lean'
run = subprocess.run(['/home/erzhu419/.elan/bin/lake', 'env', 'lean', str(source)],
                     cwd='/home/erzhu419/mine_code/proof', text=True, capture_output=True)
log = run.stdout + run.stderr
(here / 'cvar_measure.log').write_text(log)
entries = re.findall(r"'([^']+)' depends on axioms: \[([^\]]*)\]", log)
allowed = {'propext', 'Classical.choice', 'Quot.sound'}
bad = {name: [v.strip() for v in values.split(',') if v.strip() not in allowed]
       for name, values in entries}
bad = {name: values for name, values in bad.items() if values}
placeholders = re.findall(r'\b(?:sorry|admit)\b|^\s*(?:axiom|opaque)\b', source.read_text(), re.M)
result = {'exit_code': run.returncode, 'roots': [name for name, _ in entries],
          'custom_or_placeholder_dependencies': bad, 'source_placeholders': placeholders,
          'scope': 'Arbitrary probability spaces, measurable integrable real cost, 0 < beta < 1; '
                   'bounded measurable costs also cover beta = 1. No finite trajectory assumption.'}
(here / 'cvar_status.json').write_text(json.dumps(result, ensure_ascii=False, indent=2)+'\n')
print(json.dumps(result, ensure_ascii=False))
assert run.returncode == 0 and len(entries) == source.read_text().count('#print axioms')
assert not bad and not placeholders
