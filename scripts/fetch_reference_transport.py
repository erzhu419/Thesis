"""Import selected transport references from their registered DOI records."""

import json
from pathlib import Path
from urllib.parse import quote
from urllib.request import Request, urlopen

ROOT = Path(__file__).resolve().parents[1]
SELECTED = {
    'hickman2001analytic': '10.1287/trsc.35.3.215.10150',
    'dai2019predictive': '10.1016/j.trb.2019.05.001',
    'bartholdi2012self': '10.1016/j.trb.2011.11.001',
    'gkiotsalitis2021atstop': '10.1016/j.tre.2020.102176',
    'rezazada2024review': '10.1080/01441647.2024.2313969',
    'cats2012strategies': '10.3141/2274-11',
    'ibarra2015planning': '10.1016/j.trb.2015.03.002',
    'sanchezmartinez2016realtime': '10.1016/j.trb.2015.11.013',
    'daganzo2011cooperation': '10.1016/j.trb.2010.06.005',
    'wu2017overtaking': '10.1016/j.trb.2017.06.019',
    'cats2019frequency': '10.1177/0361198118822292',
    'xu2025systematic': '10.1109/TITS.2025.3545642',
}


def main():
    dest = ROOT/'manuscript/reference_transport.json'
    records = {r['id']: r for r in json.loads(dest.read_text())} if dest.exists() else {}
    for key, doi in SELECTED.items():
        if key in records:
            continue
        url = 'https://api.crossref.org/works/' + quote(doi, safe='/')
        with urlopen(Request(url, headers={'User-Agent': 'ThesisReferenceCheck/1.0'}), timeout=30) as response:
            work = json.load(response)['message']
        entry = dict(id=key, type='article-journal', title=work['title'][0],
                     author=[{k: a[k] for k in ('family','given') if k in a} for a in work['author']],
                     issued=work.get('published-print', work['issued']),
                     **{'container-title': work['container-title'][0]}, DOI=doi, language='en')
        for field in ('volume', 'issue', 'page'):
            if work.get(field):
                entry[field] = work[field]
        if not entry.get('page') and work.get('article-number'):
            entry['page'] = work['article-number']
        records[key] = dict(id=key, entry=entry, verification_url=url,
                            checked='2026-09-28', abstract=work.get('abstract',''))
        dest.write_text(json.dumps([records[k] for k in SELECTED if k in records], ensure_ascii=False, indent=2)+'\n')
        print(key, entry['title'], flush=True)
    dest.write_text(json.dumps([records[k] for k in SELECTED], ensure_ascii=False, indent=2)+'\n')


if __name__ == '__main__':
    main()
