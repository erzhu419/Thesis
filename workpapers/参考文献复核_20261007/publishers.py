"""Recheck Chinese publisher pages and six public repository identities."""
import json
import re
from pathlib import Path

from lxml import html
import requests

ROOT = Path('/home/erzhu419/mine_code/Thesis')
HERE = Path(__file__).resolve().parent
refs = json.loads((ROOT / 'workpapers/综述新增中文文献_20261006/references.json').read_text())
path = HERE / '期刊网页复核.json'
results = json.loads(path.read_text()) if path.exists() else {}
overrides = {
    171: 'https://zgglxb.chd.edu.cn/CN/10.19721/j.cnki.1001-7372.2023.06.020',
    172: 'https://www.tseit.org.cn/CN/Y2023/V23/I6/165',
    176: 'https://computing.hit.edu.cn/znjsjyy/article/abstract/20231128',
    180: 'https://zgglxb.chd.edu.cn/CN/10.19721/j.cnki.1001-7372.2023.10.022',
    185: 'https://www.tseit.org.cn/CN/Y2024/V24/I5/173',
    193: 'https://www.sciengine.com/SSI/doi/10.1360/SSI-2023-0335',
    199: 'https://cjc.ict.ac.cn/online/onlinepaper/wl-20241212082856.pdf',
    208: 'https://fcst.ceaj.org/CN/10.3778/j.issn.1673-9418.2211113',
}
session = requests.Session()
session.headers['User-Agent'] = 'Mozilla/5.0 (Thesis bibliography metadata review)'
for ref in refs:
    number = str(ref['number'])
    if number in results:
        continue
    url = overrides.get(ref['number'], ref['source'])
    if '.pdf' in url:
        results[number] = {'source': url, 'note': 'Use existing small source extract or official article record; no PDF fetched'}
        continue
    try:
        response = session.get(url, timeout=15)
        response.raise_for_status()
        if 'charset' not in response.headers.get('Content-Type', '').lower():
            response.encoding = response.apparent_encoding
        dom = html.fromstring(response.text)
        metadata = {}
        for meta in dom.xpath('//meta[@content]'):
            key = meta.get('name') or meta.get('property') or ''
            if any(v in key.lower() for v in ('citation', 'dc.', 'title', 'description', 'keywords')):
                metadata.setdefault(key, []).append(meta.get('content'))
        for node in dom.xpath('//script|//style'):
            node.drop_tree()
        text = re.sub(r'[ \t]+', ' ', dom.text_content())
        text = re.sub(r'\n\s*\n', '\n', text).strip()
        record = {'source': response.url, 'metadata': metadata, 'text': text,
                  'title_present': ref['title'] in text or ref['title'] in str(metadata), 'checked': '2026-10-07'}
    except (requests.RequestException, ValueError) as error:
        record = {'source': url, 'error': str(error), 'checked': '2026-10-07'}
    results[number] = record
    path.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding='utf-8')
    print(number, record.get('title_present', record.get('error', '')), flush=True)
path.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding='utf-8')

codes = {}
for repo in ['Categorical-Feature-SAC-in-bus-simulation', 'RE-SAC', 'BAPR', 'CS-BAPR', 'sumo-rl', 'MC-WM']:
    url = 'https://api.github.com/repos/erzhu419/' + repo
    response = session.get(url, timeout=20)
    if response.ok:
        data = response.json()
        codes[repo] = {k: data.get(k) for k in ['full_name', 'html_url', 'description', 'default_branch', 'archived', 'private']}
    else:
        codes[repo] = {'source': url, 'http_status': response.status_code}
(HERE / '仓库复核.json').write_text(json.dumps(codes, ensure_ascii=False, indent=2), encoding='utf-8')
print('repository identities:', len(codes), flush=True)
