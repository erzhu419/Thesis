from pathlib import Path
import json,sys
here=Path(__file__).resolve().parent
root=Path(sys.argv[1])
result=[]
for item in json.loads((here/'source_expected.json').read_text()):
    actual=(root/item['path']).read_text().splitlines()[item['low']-1:item['high']]
    result.append(dict(path=item['path'],low=item['low'],high=item['high'],equal=actual==item['lines']))
print(json.dumps(dict(all_pass=all(r['equal'] for r in result),segments=result),separators=(',',':')))
