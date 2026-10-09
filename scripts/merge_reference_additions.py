"""Merge verified, actually cited additions into the thesis bibliography."""

import json
from pathlib import Path

import pypandoc

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT/'manuscript'


def citation_ids(node):
    if isinstance(node, dict):
        if node.get('t') == 'Cite':
            yield from (c['citationId'] for c in node['c'][0])
        for value in node.values():
            yield from citation_ids(value)
    elif isinstance(node, list):
        for value in node:
            yield from citation_ids(value)


def main():
    text = '\n'.join(p.read_text() for p in sorted((SOURCE/'chapters').glob('*.md')))
    ast = json.loads(pypandoc.convert_text(text, 'json', format='markdown'))
    cited = set(citation_ids(ast))
    candidates = sum((json.loads((SOURCE/f).read_text()) for f in
                      ('reference_preprints.json', 'reference_transport.json',
                       'reference_chinese.json')), [])
    additions = []
    for record in candidates:
        if record['id'] not in cited:
            continue
        entry = dict(record['entry'])
        if record['verification_url'].startswith('https://arxiv.org/'):
            entry['type'] = 'article'
            entry['archive'] = 'arXiv'
            entry['number'] = entry['URL'].rsplit('/', 1)[1]
            entry.pop('container-title', None)
        additions.append(entry)
    existing = {r['id']: r for r in json.loads((SOURCE/'references.json').read_text())}
    existing.update({r['id']: r for r in additions})
    assert cited <= existing.keys(), sorted(cited - existing.keys())
    titles = [''.join(c for c in r['title'].casefold() if c.isalnum()) for r in existing.values()]
    assert len(titles) == len(set(titles)), 'Duplicate work title in bibliography'
    (SOURCE/'reference_additions.json').write_text(json.dumps(additions, ensure_ascii=False, indent=2)+'\n')
    (SOURCE/'references.json').write_text(json.dumps(list(existing.values()), ensure_ascii=False, indent=2)+'\n')
    print(json.dumps(dict(total=len(existing), additions=len(additions),
                         arxiv_additions=sum(r['type']=='article' for r in additions))))


if __name__ == '__main__':
    main()
