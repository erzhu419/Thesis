"""Refresh small publisher-deposited records for the journal/conference entries."""
import json
from pathlib import Path
import re
from urllib.parse import quote
from urllib.request import Request, urlopen
from urllib.error import HTTPError
import time

HERE = Path(__file__).resolve().parent
path = HERE/'primary_sources.json'
sources = json.loads(path.read_text())
refs = json.loads((HERE/'current_references.json').read_text())
for r in refs:
    s = sources[r['key']]
    if s.get('primary') or s.get('crossref') or s.get('crossref_http_status'):
        continue
    doi = re.search(r'DOI:([^\s]+)', r['text'])
    if not doi:
        continue
    url = 'https://api.crossref.org/works/' + quote(doi.group(1).rstrip('.'), safe='')
    req = Request(url, headers={'User-Agent': 'ThesisCitationReview/1.0'})
    try:
        record = json.loads(urlopen(req, timeout=60).read())['message']
    except HTTPError as e:
        if e.code != 404:
            raise
        s['crossref_http_status'] = 404
        s['crossref_checked'] = '2026-10-03'
        s['crossref_url'] = url
        path.write_text(json.dumps(sources, ensure_ascii=False, indent=2)+'\n')
        print(r['number'], 'DOI not indexed in Crossref; publisher lookup required', flush=True)
        continue
    s['crossref'] = record
    s['crossref_checked'] = '2026-10-03'
    path.write_text(json.dumps(sources, ensure_ascii=False, indent=2)+'\n')
    print(r['number'], record['title'], 'abstract='+str(bool(record.get('abstract'))), flush=True)
    time.sleep(1)
