"""Show metadata discrepancies and source passages for editorial judgment."""
import json
from pathlib import Path
import re
import sys
import unicodedata

HERE = Path(__file__).resolve().parent
PREV = HERE.parent / '参考文献复核_20261007'


def norm(s):
    return ''.join(c for c in unicodedata.normalize('NFKD', s).lower() if c.isalnum())


refs = json.loads((HERE / '当前文献与引用.json').read_text())
records = json.loads((HERE / '在线原始记录.json').read_text())
previous = json.loads((PREV / '在线元数据.json').read_text())
chinese = json.loads((PREV / '期刊网页复核.json').read_text())


def record(r):
    n = str(r['number'])
    s = records.get(n, {})
    if 'error' in s or not s:
        s = previous.get(n, chinese.get(n, {}))
        s = dict(s, cache_only=True)
    return s


def title(r):
    body = re.sub(r'^\[\d+\]\s*', '', r['text'])
    return body.split('. ', 1)[1].split('[', 1)[0]


def metadata(r, s):
    if 'version_page' in s:
        m = s['version_page']['metadata']
        return m.get('citation_title', [''])[0], m.get('citation_author', []), m
    if s.get('kind') == 'arxiv':
        return s['title'], s['authors'], {'published': s['published'], 'updated': s['updated']}
    if s.get('kind') == 'crossref':
        return s['title'][0], [' '.join([a.get('given', ''), a.get('family', '')]) for a in s['author']], {k: s.get(k) for k in ('container-title','published','published-print','published-online','volume','issue','page','article-number')}
    m = s.get('metadata', {})
    return (m.get('citation_title') or [''])[0], (m.get('citation_author') or m.get('DC.Creator') or []), m


start, end = map(int, sys.argv[2:4]) if len(sys.argv) > 3 else (1,214)
mode = sys.argv[1]
for r in refs[start-1:end]:
    s = record(r)
    t, authors, m = metadata(r, s)
    if mode == 'metadata':
        print('\nREF', r['number'], 'CACHE' if s.get('cache_only') else 'LIVE', r['text'])
        print('TITLE', t, 'MATCH', norm(t) == norm(title(r)))
        print('AUTHORS', authors)
        keys = ('citation_journal_title','citation_authors','DC.Contributor','dc.creator','citation_date','citation_online_date','dc.date','DC.Date','citation_volume','citation_issue','citation_firstpage','citation_lastpage','citation_doi','container-title','published','published-print','published-online','volume','issue','page','article-number','updated')
        print('DETAIL', {k:m[k] for k in keys if k in m})
        if s.get('kind') in ('publisher','publisher-pdf') and not t:
            print('TEXT', s.get('text', '')[:1700])
    elif mode == 'differences':
        if t and norm(t) != norm(title(r)):
            print(r['number'], '\nLISTED', title(r), '\nSOURCE', t)
        elif not t:
            print(r['number'], 'NO STRUCTURED TITLE', s.get('kind'), 'title_in_text', norm(title(r)) in norm(s.get('text','')))
    elif mode == 'claims':
        print('\nREF', r['number'], title(r))
        print('SOURCE', s.get('source',''), 'CACHE' if s.get('cache_only') else 'LIVE')
        abstract = s.get('abstract','')
        if not abstract:
            text = s.get('text','')
            at = text.find('摘要')
            if at < 0:
                at = text.find('Abstract')
            abstract = text[max(0,at):max(0,at)+1500]
        print('EVIDENCE', re.sub('<[^>]+>', '', abstract)[:1800])
        for u in r['uses']:
            print('USE',u['index'],u['section'],u['text'])
    elif mode == 'authors':
        print(r['number'], r['text'].split('. ', 1)[0], '|', '; '.join(authors[:3]))
