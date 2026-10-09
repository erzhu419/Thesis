"""Fetch small primary metadata and abstracts, never paper PDFs or checkpoints."""
import json
from pathlib import Path
import re
import time
import unicodedata
from urllib.parse import urlencode
from urllib.request import urlopen
import xml.etree.ElementTree as E

HERE = Path(__file__).resolve().parent
refs = json.loads((HERE/'current_references.json').read_text())
old = json.loads((HERE/'inherited_metadata.json').read_text())
ATOM = {'a': 'http://www.w3.org/2005/Atom', 'ar': 'http://arxiv.org/schemas/atom'}


def title(ref):
    m = old.get(ref['key'], {})
    return m.get('entry', {}).get('title') or m.get('title') or re.split(
        r'\[(?:EB/OL|J/OL|C/OL|D)\]', ref['text'].split('. ', 1)[1])[0]


def normalize(s):
    return ''.join(c for c in unicodedata.normalize('NFKD', s).lower() if c.isalnum())


def arxiv(params):
    url = 'https://export.arxiv.org/api/query?' + urlencode(params)
    feed = E.fromstring(urlopen(url, timeout=90).read())
    entries = []
    for e in feed.findall('a:entry', ATOM):
        entries.append({
            'id': e.findtext('a:id', '', ATOM).replace('http:', 'https:'),
            'title': ' '.join(e.findtext('a:title', '', ATOM).split()),
            'authors': [v.findtext('a:name', '', ATOM) for v in e.findall('a:author', ATOM)],
            'published': e.findtext('a:published', '', ATOM),
            'updated': e.findtext('a:updated', '', ATOM),
            'abstract': ' '.join(e.findtext('a:summary', '', ATOM).split()),
            'doi': e.findtext('ar:doi', '', ATOM), 'api_url': url})
    return entries


path = HERE/'primary_sources.json'
data = json.loads(path.read_text()) if path.exists() else {}
ids = []
for r in refs:
    match = re.search(r'https://arxiv.org/abs/(\d{4}\.\d{4,5}(?:v\d+)?)', r['text'])
    if match and not data.get(r['key'], {}).get('primary'):
        ids.append((r, match.group(1)))
if ids:
    entries = arxiv({'id_list': ','.join(v for _, v in ids), 'max_results': 200})
    by_id = {re.sub(r'v\d+$', '', v['id'].rsplit('/', 1)[1]): v for v in entries}
    for r, aid in ids:
        result = by_id.get(re.sub(r'v\d+$', '', aid))
        data[r['key']] = {'number': r['number'], 'requested_id': aid,
                          'primary': result, 'title_match': bool(result) and
                          normalize(result['title']) == normalize(title(r)), 'checked': '2026-10-03'}
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2)+'\n')
    print('arXiv current entries fetched: '+str(len(entries)), flush=True)
    time.sleep(3)
for r in refs:
    if r['key'] in data:
        continue
    t = title(r)
    query = 'ti:"' + t.replace('"', '') + '"'
    entries = arxiv({'search_query': query, 'max_results': 30})
    matches = [v for v in entries if normalize(v['title']) == normalize(t)]
    data[r['key']] = {'number': r['number'], 'arxiv_title_query': query,
                      'arxiv_candidates': entries, 'matched_arxiv': matches,
                      'inherited_primary': old.get(r['key'], {}), 'checked': '2026-10-03'}
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2)+'\n')
    print(str(r['number'])+' '+r['key']+': '+str(len(matches))+' exact-title arXiv matches', flush=True)
    time.sleep(3)
print(json.dumps({'records': len(data), 'existing_title_mismatches':
    [k for k, v in data.items() if 'title_match' in v and not v['title_match']],
    'new_arxiv_matches': [k for k, v in data.items() if v.get('matched_arxiv')]}, ensure_ascii=False))
