"""Read arXiv abstract metadata for the selected thesis literature."""

import csv
import json
import re
import time
from pathlib import Path
from urllib.request import Request, urlopen

from lxml import html


ROOT = Path(__file__).resolve().parents[1]
DEST = ROOT / 'manuscript/reference_preprints.json'


def normalized(text):
    return ''.join(c for c in text.casefold() if c.isalnum())


def main():
    with (ROOT/'manuscript/references_to_add.tsv').open() as stream:
        selected = list(csv.DictReader(stream, delimiter='\t'))
    cached = {r['id']:r for r in json.loads(DEST.read_text())} if DEST.exists() else {}
    failures = []
    for item in selected:
        key, number = item['id'], item['arxiv']
        assert re.fullmatch(r'[a-z0-9]+', key), key
        url = 'https://arxiv.org/abs/' + number
        if key in cached and cached[key]['verification_url'] == url:
            continue
        try:
            request = Request(url, headers={'User-Agent':'Mozilla/5.0'})
            with urlopen(request, timeout=20) as response:
                page = html.fromstring(response.read())
            def meta(name):
                return page.xpath('//meta[@name=$name]/@content', name='citation_'+name)
            title = meta('title')[0]
            if normalized(title) != normalized(item['expected_title']):
                failures.append(dict(item, actual_title=title, reason='title differs; review before inclusion'))
                print('TITLE', key, title, flush=True)
                continue
            authors = []
            for author in meta('author'):
                if ',' in author:
                    family, given = author.split(',',1)
                    authors.append(dict(family=family.strip(),given=given.strip()))
                else:
                    authors.append(dict(literal=author))
            abstract = meta('abstract')
            if not abstract:
                abstract = [page.xpath('//blockquote[contains(@class,"abstract")]')[0].text_content().strip().removeprefix('Abstract:')]
            entry = dict(id=key, type='article-journal', title=title, author=authors,
                         issued={'date-parts':[[int(meta('date')[0][:4])]]},
                         **{'container-title':'arXiv preprint arXiv:'+number}, URL=url, language='en')
            cached[key] = dict(id=key, entry=entry, verification_url=url,
                               checked='2026-09-28', abstract=' '.join(abstract[0].split()))
            DEST.write_text(json.dumps(list(cached.values()),ensure_ascii=False,indent=2)+'\n')
            print('OK', key, title, flush=True)
        except Exception as error:
            failures.append(dict(item, reason=str(error)))
            print('FAILED', key, str(error), flush=True)
        time.sleep(.3)
    (ROOT/'workpapers/reference_inventory/preprint_fetch_issues.json').write_text(json.dumps(failures,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(dict(selected=len(selected),cached=len(cached),failures=len(failures))))


if __name__=='__main__':
    main()
