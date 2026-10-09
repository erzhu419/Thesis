"""Read current Word blocks with stable pre-edit anchors for manual review."""
import argparse
import json
from pathlib import Path

p = argparse.ArgumentParser()
p.add_argument('start',type=int)
p.add_argument('end',type=int)
p.add_argument('--mode',choices=['all','text','math','headings'],default='all')
args=p.parse_args()
b=json.loads((Path(__file__).resolve().parent/'current_blocks.json').read_text())
for x in b[args.start:args.end+1]:
    if not x['text'] or x['style'].startswith('TOC'):
        continue
    if args.mode=='text' and x.get('rows'):
        continue
    if args.mode=='math' and not x.get('rows'):
        continue
    if args.mode=='headings' and x['style'] not in ['chapter','h2','h3','h4']:
        continue
    print(str(x['block'])+' ['+x['style']+'] '+x['text'])
