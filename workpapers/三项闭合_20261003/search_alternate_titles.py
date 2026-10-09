"""Check author plus distinctive title words for versions with changed titles."""
import json
from pathlib import Path
import re
import time
from urllib.request import urlopen
from urllib.parse import urlencode
import xml.etree.ElementTree as E

HERE = Path(__file__).resolve().parent
path = HERE/'primary_sources.json'
data = json.loads(path.read_text())
refs = json.loads((HERE/'current_references.json').read_text())
queries = {
    1:'au:Daganzo AND ti:headway', 2:'au:Xuan AND ti:holding',
    3:'au:Delgado AND ti:boarding', 4:'au:Wang AND ti:holding AND ti:bunching',
    6:'au:Wang AND ti:multi AND ti:corridor', 7:'au:Lopez AND ti:microscopic',
    11:'au:Rojas AND ti:bus AND ti:review', 12:'au:Hickman AND ti:holding',
    13:'au:Andres AND ti:bunching', 14:'au:Daganzo AND ti:cooperation',
    15:'au:Bartholdi AND ti:bus', 16:'au:Cats AND ti:holding',
    17:'au:Dai AND ti:holding', 18:'au:Martinez AND ti:holding',
    19:'au:Wu AND ti:overtaking AND ti:bus', 20:'au:Gkiotsalitis AND ti:control AND ti:review',
    21:'au:Rezazada AND ti:bunching', 22:'au:He AND ti:holding AND ti:transfer',
    23:'au:Zhao AND ti:bus AND ti:overlapping', 24:'au:Li AND ti:bus AND ti:aggregation',
    25:'au:Liu AND ti:bus AND ti:priority', 27:'au:Alesiani AND ti:holding',
    30:'au:Iyengar AND ti:robust AND ti:dynamic',
    35:'au:Rodriguez AND ti:bus AND ti:cooperative',
    41:'au:Yu AND ti:bus AND ti:hierarchical',
    42:'au:Li AND ti:bus AND ti:graph',
    46:'au:Xu AND ti:bunching', 48:'au:Wei AND ti:pressure',
    130:'au:Todorov AND ti:mujoco', 131:'au:Sutton AND ti:temporal AND ti:abstraction',
}
ns={'a':'http://www.w3.org/2005/Atom'}
for r in refs:
    if r['number'] not in queries or data[r['key']].get('alternate_search'):
        continue
    query=queries[r['number']]
    url='https://export.arxiv.org/api/query?'+urlencode({'search_query':query,'max_results':30})
    feed=E.fromstring(urlopen(url,timeout=90).read())
    candidates=[{'id':v.findtext('a:id','',ns).replace('http:','https:'),
                 'title':' '.join(v.findtext('a:title','',ns).split()),
                 'authors':[a.findtext('a:name','',ns) for a in v.findall('a:author',ns)],
                 'abstract':' '.join(v.findtext('a:summary','',ns).split())}
                for v in feed.findall('a:entry',ns)]
    data[r['key']]['alternate_search']={'query':query,'url':url,'candidates':candidates,
                                       'checked':'2026-10-03'}
    path.write_text(json.dumps(data,ensure_ascii=False,indent=2)+'\n')
    print(r['number'],[(x['id'],x['title']) for x in candidates],flush=True)
    time.sleep(3)
