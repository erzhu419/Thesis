"""Audit theorem dependencies; a successful build alone is not a closed proof."""
import json
from pathlib import Path
import re
import subprocess

HERE = Path(__file__).resolve().parent


def audit(filename, stem, scope):
    source = HERE / filename
    run = subprocess.run(['/home/erzhu419/.elan/bin/lake', 'env', 'lean', str(source)],
                         cwd='/home/erzhu419/mine_code/proof', text=True, capture_output=True)
    log = run.stdout + run.stderr
    (HERE/(stem+'.log')).write_text(log)
    entries = re.findall(r"'([^']+)' depends on axioms: \[([^\]]*)\]", log)
    allowed = {'propext', 'Classical.choice', 'Quot.sound'}
    dependencies = {name: [v.strip() for v in values.split(',')
                          if v.strip() and v.strip() not in allowed]
                    for name, values in entries}
    bad = {name: values for name, values in dependencies.items() if values}
    result = {'exit_code': run.returncode, 'audited_roots': len(entries),
              'custom_dependencies': bad,
              'placeholders': re.findall(r'\b(?:sorry|admit)\b|^\s*(?:axiom|opaque)\b',
                                        source.read_text(), re.M),
              'roots': [name for name, _ in entries], 'scope': scope}
    name = 'proof_status' if filename == 'AppendixExtensions.lean' else 'existing_proof_status'
    (HERE/(name+'.json')).write_text(json.dumps(result, ensure_ascii=False, indent=2)+'\n')
    print(json.dumps({k: v for k, v in result.items() if k != 'roots'}, ensure_ascii=False), flush=True)
    assert run.returncode == 0 and len(entries) == source.read_text().count('#print axioms')
    assert not bad and not result['placeholders']


audit('AppendixExtensions.lean', 'lean_extensions',
      'CVaR is finite-trajectory; other state-probability results use finite state spaces, '
      'not finite uncertainty sets. General vector derivative composition and finite-history '
      'KL chain rule are included.')
audit('ExistingAppendixAudit.lean', 'lean_existing',
      'Dependency audit of existing RE-SAC, BAPR and CS-BAPR appendix roots; '
      'historical MC-WM axioms are not imported.')
