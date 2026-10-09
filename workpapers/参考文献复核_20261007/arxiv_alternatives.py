"""Search official arXiv for matching editions of remaining journal references."""
import json
import re
import time
from pathlib import Path
import xml.etree.ElementTree as E

import requests

HERE = Path(__file__).resolve().parent
refs = json.loads((HERE / 'references.json').read_text())
online = json.loads((HERE / '在线元数据.json').read_text())
ns = {'a': 'http://www.w3.org/2005/Atom', 'ar': 'http://arxiv.org/schemas/atom'}
titles = []
for ref in refs:
    if ref['arxiv'] or ref['number'] > 158 or ref['number'] in [22, 23, 24, 25]:
        continue
    title = re.split(r'\[(?:J/OL|C/OL)\]', ref['text'].split('. ', 1)[1])[0]
    titles.append((ref['number'], title))
results = []
for start in range(0, len(titles), 8):
    batch = titles[start:start + 8]
    query = ' OR '.join('ti:"' + title.replace('"', '') + '"' for _, title in batch)
    response = requests.get('https://export.arxiv.org/api/query', params={
        'search_query': query, 'max_results': 100}, timeout=45)
    response.raise_for_status()
    feed = E.fromstring(response.content)
    entries = []
    for entry in feed.findall('a:entry', ns):
        entries.append({'id': entry.findtext('a:id', '', ns).replace('http:', 'https:'),
                        'title': ' '.join(entry.findtext('a:title', '', ns).split()),
                        'authors': [v.findtext('a:name', '', ns) for v in entry.findall('a:author', ns)],
                        'published': entry.findtext('a:published', '', ns),
                        'updated': entry.findtext('a:updated', '', ns),
                        'doi': entry.findtext('ar:doi', '', ns)})
    results.append({'requested': batch, 'results': entries, 'query_url': response.url})
    print(batch, entries, flush=True)
    time.sleep(3)
(HERE / '期刊文献arXiv检索.json').write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding='utf-8')
