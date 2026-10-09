"""Refresh small arXiv/Crossref records without downloading papers."""
import json
import time
from pathlib import Path
import re
from urllib.parse import quote
import xml.etree.ElementTree as E

import requests

HERE = Path(__file__).resolve().parent
refs = json.loads((HERE / 'references.json').read_text())
path = HERE / '在线元数据.json'
records = json.loads(path.read_text()) if path.exists() else {}
session = requests.Session()
session.headers['User-Agent'] = 'Thesis reference review (metadata only)'
NS = {'a': 'http://www.w3.org/2005/Atom', 'ar': 'http://arxiv.org/schemas/atom'}


def save():
    path.write_text(json.dumps(records, ensure_ascii=False, indent=2), encoding='utf-8')


pending = [r for r in refs if r['arxiv'] and str(r['number']) not in records]
if pending:
    response = session.get('https://export.arxiv.org/api/query', params={
        'id_list': ','.join(r['arxiv'] for r in pending), 'max_results': len(pending)}, timeout=90)
    response.raise_for_status()
    feed = E.fromstring(response.content)
    by_id = {}
    for entry in feed.findall('a:entry', NS):
        identifier = entry.findtext('a:id', '', NS).replace('http:', 'https:')
        by_id[re.sub(r'v\d+$', '', identifier.rsplit('/', 1)[-1])] = {
            'kind': 'arxiv', 'source': identifier,
            'title': ' '.join(entry.findtext('a:title', '', NS).split()),
            'authors': [v.findtext('a:name', '', NS) for v in entry.findall('a:author', NS)],
            'published': entry.findtext('a:published', '', NS),
            'updated': entry.findtext('a:updated', '', NS),
            'abstract': ' '.join(entry.findtext('a:summary', '', NS).split()),
            'doi': entry.findtext('ar:doi', '', NS), 'checked': '2026-10-07'}
    for ref in pending:
        key = re.sub(r'v\d+$', '', ref['arxiv'])
        records[str(ref['number'])] = by_id.get(key, {'error': 'No matching arXiv API record', 'requested': ref['arxiv']})
    save()
    print(f'arXiv: {len(by_id)} primary records', flush=True)
    time.sleep(3)

for ref in refs:
    if not ref['doi'] or str(ref['number']) in records:
        continue
    url = 'https://api.crossref.org/works/' + quote(ref['doi'], safe='')
    try:
        response = session.get(url, timeout=15)
        response.raise_for_status()
        data = response.json()['message']
        record = {k: data.get(k) for k in ['title', 'author', 'container-title', 'volume', 'issue',
                  'page', 'article-number', 'published', 'published-online', 'published-print',
                  'DOI', 'URL', 'abstract', 'type']}
        record.update(kind='crossref', source=url, checked='2026-10-07')
    except (requests.RequestException, ValueError, KeyError) as error:
        record = {'error': str(error), 'source': url, 'checked': '2026-10-07'}
    records[str(ref['number'])] = record
    save()
    print(f"{ref['number']}: {'lookup failed' if 'error' in record else record['title']}", flush=True)
print(f'{len(records)} online records attempted', flush=True)
