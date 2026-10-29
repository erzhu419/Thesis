"""Resolve specific primary-source gaps, retaining small bibliographic evidence."""
from concurrent.futures import ThreadPoolExecutor, as_completed
import importlib.util
import json
from pathlib import Path
from urllib.parse import quote

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('audit', HERE / 'audit.py')
audit = importlib.util.module_from_spec(spec)
spec.loader.exec_module(audit)

targets = {
    '7-author-order': 'https://elib.dlr.de/124092/',
    '39-author-name': 'https://proceedings.mlr.press/v80/rashid18a.html',
    '141-author-name': 'https://proceedings.mlr.press/v48/mniha16.html',
    '165-volume': 'https://glyqy.csust.edu.cn/glyqy/article/abstract/20240306',
    '168-primary': 'https://www.ort.shu.edu.cn/CN/10.15960/j.cnki.issn.1007-6093.2023.03.005',
    '172-issue': 'https://www.tseit.org.cn/CN/Y2023/V23/I6',
    '185-issue': 'https://www.tseit.org.cn/CN/Y2024/V24/I5',
    '187-primary': 'https://kzyjc.org/kzyjc/article/abstract/2022-0044',
    '191-primary': 'https://www.ort.shu.edu.cn/CN/Y2023/V27/I2/49',
    '193-primary-pdf': 'https://infocn.scichina.com/2025/SSI-2023-0335.pdf',
    '193-crossref': 'https://api.crossref.org/works/' + quote('10.1360/SSI-2023-0335', safe=''),
    '172-issue-no-www': 'https://tseit.org.cn/CN/Y2023/V23/I6',
    '185-issue-no-www': 'https://tseit.org.cn/CN/Y2024/V24/I5',
    '187-http': 'http://kzyjc.alljournals.cn/html/2022/12/20221202.htm',
    '193-sciengine': 'https://www.sciengine.com/doi/10.1360/SSI-2023-0335',
    '193-author-published': 'https://www.lamda.nju.edu.cn/lilh/file/openmarl.pdf',
    '187-primary-alternate': 'https://kzyjc.alljournals.cn/kzyjc/article/abstract/2022-0044',
    '172-http': 'http://www.tseit.org.cn/CN/Y2023/V23/I6',
    '185-http': 'http://www.tseit.org.cn/CN/Y2024/V24/I5',
    '193-official-pdf': 'https://scis.scichina.com/cn/2025/SSI-2023-0335.pdf',
    '193-primary-pdf-tls': 'https://infocn.scichina.com/2025/SSI-2023-0335.pdf',
}
refs = audit.read(HERE / '当前文献与引用.json')
for n in [2, 3, 14, 15, 20]:
    targets[str(n) + '-crossref'] = 'https://api.crossref.org/works/' + quote(refs[n-1]['doi'], safe='')


def fetch(key, url):
    try:
        if 'api.crossref.org' in url:
            data = audit.get(url).json()['message']
            rec = {k: data.get(k) for k in ('title', 'subtitle', 'author', 'container-title', 'volume', 'issue', 'page', 'published', 'published-print', 'DOI', 'abstract')}
            rec.update(kind='crossref', source=url)
        else:
            try:
                rec = audit.html_record(url)
            except audit.requests.exceptions.SSLError:
                # These public journal sites returned expired/mismatched certificates.
                rec = audit.html_record(url, verify=False)
                rec['tls_note'] = 'Public metadata read with certificate verification disabled after an SSL error.'
        rec['checked'] = audit.DATE
    except Exception as exc:
        rec = dict(source=url, error=str(exc), checked=audit.DATE)
    return key, rec


path = HERE / '疑点一手来源.json'
records = audit.read(path) if path.exists() else {}
with ThreadPoolExecutor(max_workers=3) as pool:
    for future in as_completed([pool.submit(fetch, k, v) for k, v in targets.items() if k not in records]):
        key, rec = future.result()
        records[key] = rec
        audit.write(path.name, records)
        print(key, rec.get('kind', rec.get('error')), flush=True)
