"""Read the journal's embedded structured metadata, not its Vue placeholders."""
import json
from pathlib import Path
import requests

here = Path(__file__).resolve().parent
refs = json.loads((here / 'current_references.json').read_text())
path = here / 'primary_sources.json'
sources = json.loads(path.read_text())
url = 'https://zgglxb.chd.edu.cn/CN/10.19721/j.cnki.1001-7372.2021.07.003'
response = requests.get(url, timeout=30)
response.raise_for_status()
payload = response.text.split('window.metaData = ', 1)[1]
metadata, _ = json.JSONDecoder().raw_decode(payload)
article = metadata['article']
assert article['title_cn'] == '车路协同环境下重叠线路公交车速诱导策略'
sources[refs[22]['key']]['publisher_article'] = article
sources[refs[22]['key']]['publisher_url'] = url
path.write_text(json.dumps(sources, ensure_ascii=False, indent=2), encoding='utf-8')
print(article['zhaiyao_cn'])
