"""Flag author and publication fields needing manual inspection, not fuzzy corrections."""
import json
import re
import unicodedata

from audit import HERE, read, write


def norm(value):
    return ''.join(c for c in unicodedata.normalize('NFKC', str(value)).lower() if c.isalnum())


refs = read(HERE / '当前文献与引用.json')
records = read(HERE / '在线原始记录.json')
extra = read(HERE / '疑点一手来源.json')
web = read(HERE / '专项证据补记.json')
flags = []
for ref in refs:
    n = ref['number']
    rec = records[str(n)]
    if str(n) + '-crossref' in extra:
        rec = extra[str(n) + '-crossref']
    key = {172: '172-http', 185: '185-http', 187: '187-http'}.get(n)
    if key:
        rec = extra[key]
    if n in (168,191,193):
        rec = web[str(n)]
    if ref['arxiv']:
        if 'version_page' in rec:
            date = rec['version_page']['metadata']['citation_online_date'][0]
        else:
            date = rec['published']
        year = re.search(r'arXiv, (\d{4})', ref['text'])[1]
        accessed = re.search(r'\[(\d{4}-\d{2}-\d{2})\]', ref['text'])[1]
        if year != date[:4]:
            flags.append(dict(number=n, field='arxiv-year', listed=year, actual=date))
        if accessed < date[:10].replace('/', '-'):
            flags.append(dict(number=n, field='access-before-version', accessed=accessed, actual=date))
    elif re.search('[\u4e00-\u9fff]', ref['text'].split('. ')[0]) and n != 25:
        source = norm(json.dumps(rec, ensure_ascii=False)) + norm(rec.get('text', ''))
        authors = re.sub(r'^\[\d+\]\s*', '', ref['text']).split('. ')[0].split(', ')
        for author in authors:
            if author == '等':
                continue
            if norm(author) not in source:
                flags.append(dict(number=n, field='author-not-found', author=author))
        match = re.search(r', (\d{4}), (\d+)\((\d+)\): (\d+)-(\d+)', ref['text'])
        if match:
            for label, val in zip(('year','volume','issue','firstpage','lastpage'), match.groups()):
                if val not in source:
                    flags.append(dict(number=n, field=label + '-not-found', value=val))
        if ref['doi'] and norm(ref['doi']) not in source:
            flags.append(dict(number=n, field='doi-not-found', value=ref['doi']))
write('著录字段候选问题.json', flags)
print(json.dumps(flags, ensure_ascii=False, indent=2))
