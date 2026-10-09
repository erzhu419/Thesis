"""Display cited claims next to primary abstracts; keep review judgments manual."""
import argparse
import json
from pathlib import Path
import re

HERE = Path(__file__).resolve().parent
p = argparse.ArgumentParser()
p.add_argument('start', type=int)
p.add_argument('end', type=int)
p.add_argument('--abstract', type=int, default=1400)
args = p.parse_args()
refs = json.loads((HERE/'current_references.json').read_text())
sources = json.loads((HERE/'primary_sources.json').read_text())
printed = set()
for r in refs[args.start-1:args.end]:
    s = sources[r['key']]
    a = s.get('primary') or s.get('crossref') or s.get('inherited_primary', {})
    print('\n###', r['number'], r['key'])
    print(r['text'])
    print('PRIMARY:', a.get('title', ''), '|', a.get('id', a.get('URL', '')))
    print('ABSTRACT:', re.sub('<[^>]+>', '', a.get('abstract', ''))[:args.abstract])
    for u in r['uses']:
        if u['block'] in printed:
            print('USE', u['block'], '(paragraph already displayed)')
        else:
            print('USE', u['block'], u['section'], u['text'])
            printed.add(u['block'])
