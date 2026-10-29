"""Extract current citations and retrieve small primary bibliographic records."""

from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
import importlib.util
import json
from pathlib import Path
import re
import sys
from urllib.parse import quote
from zipfile import ZipFile

from lxml import etree as ET, html
import requests

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
PREV = ROOT / 'workpapers/参考文献复核_20261007'
SOURCE = ROOT / 'output/排版完成_七章重写稿整合_20261010.docx'
NS = {'w': 'http://schemas.openxmlformats.org/wordprocessingml/2006/main',
      'm': 'http://schemas.openxmlformats.org/officeDocument/2006/math'}
W = '{' + NS['w'] + '}'
DATE = '2026-10-10'


def write(name, value):
    (HERE / name).write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')


def read(path):
    return json.loads(path.read_text())


def extract():
    spec = importlib.util.spec_from_file_location('extract_helper', ROOT / 'workpapers/全文深度校审_20261003/extract_word.py')
    helper = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(helper)
    original_content = helper.content
    def content(node):
        if ET.QName(node).namespace not in NS.values():
            return ''
        return original_content(node)
    helper.content = content
    with ZipFile(SOURCE) as archive:
        doc = ET.fromstring(archive.read('word/document.xml'))
    blocks, refs = [], []
    uses = {}
    section = chapter = '前置部分'
    for i, node in enumerate(doc.find('w:body', NS)):
        styles = node.xpath('./w:pPr/w:pStyle/@w:val', namespaces=NS)
        style = styles[0] if styles else ''
        text = helper.content(node)
        if style in ('chapter', 'bibliographytitle'):
            chapter = section = text
        elif re.match(r'^[1-7]\.\d', text) and style not in ('body', 'caption'):
            section = text
        anchors = node.xpath('.//w:hyperlink/@w:anchor', namespaces=NS)
        bookmarks = node.xpath('.//w:bookmarkStart/@w:name', namespaces=NS)
        block = dict(index=i, style=style, chapter=chapter, section=section,
                     text=text, anchors=anchors)
        blocks.append(block)
        keys = [b for b in bookmarks if b.startswith('ref-')]
        if keys:
            n = int(re.match(r'\[(\d+)\]', text)[1])
            refs.append(dict(number=n, key=keys[0], text=text, block=i))
        else:
            for key in dict.fromkeys(anchors):
                if key.startswith('ref-'):
                    uses.setdefault(key, []).append(block)
    old = {r['number']: r for r in read(PREV / '逐项文献核对.json')}
    for ref in refs:
        ref['uses'] = uses.get(ref['key'], [])
        a = re.search(r'arxiv.org/(?:abs|pdf)/(\d{4}\.\d{4,5}(?:v\d+)?)', ref['text'])
        d = re.search(r'(?:DOI:|doi.org/)(10\.[^\s]+)', ref['text'], re.I)
        ref['arxiv'] = a[1] if a else ''
        ref['doi'] = d[1].rstrip('.') if d else ''
        ref['prior_source'] = old[ref['number']]['source']
        ref['prior_note'] = old[ref['number']]['citation_note']
        ref['bibliography_unchanged'] = ref['text'].strip() == old[ref['number']]['entry'].strip()
    write('当前文献与引用.json', refs)
    write('当前正文块.json', blocks)
    print(json.dumps(dict(total=len(refs), arxiv=sum(bool(r['arxiv']) for r in refs),
                          uncited=[r['number'] for r in refs if not r['uses']],
                          unique_cited_blocks=len({u['index'] for r in refs for u in r['uses']}),
                          changed_bibliography=[r['number'] for r in refs if not r['bibliography_unchanged']]), ensure_ascii=False))


def get(url, **kwargs):
    r = requests.get(url, headers={'User-Agent': 'Mozilla/5.0 (doctoral bibliography metadata verification)'},
                     timeout=35, **kwargs)
    r.raise_for_status()
    return r


def html_record(url, **kwargs):
    r = get(url, **kwargs)
    if r.content[:4] == b'%PDF':
        import fitz
        p = fitz.open(stream=r.content, filetype='pdf')
        return dict(source=r.url, kind='publisher-pdf', text='\n'.join(page.get_text() for page in p), bytes=len(r.content))
    if 'charset' not in r.headers.get('Content-Type', '').lower():
        r.encoding = r.apparent_encoding
    dom = html.fromstring(r.text)
    meta = {}
    for e in dom.xpath('//meta[@content]'):
        key = e.get('name') or e.get('property') or ''
        if any(k in key.lower() for k in ('citation', 'dc.', 'title', 'description', 'keywords')):
            meta.setdefault(key, []).append(e.get('content'))
    for e in dom.xpath('//script|//style'):
        e.drop_tree()
    text = re.sub(r'[ \t\r]+', ' ', dom.text_content())
    text = re.sub(r'\n\s*\n', '\n', text).strip()
    return dict(source=r.url, kind='publisher', metadata=meta, text=text)


def fetch_one(ref):
    n = ref['number']
    url = ref['prior_source']
    try:
        if 'github.com' in url:
            url = url.replace('https://github.com/', 'https://api.github.com/repos/')
            data = get(url).json()
            rec = dict(kind='repository', source=data['html_url'], **{k: data.get(k) for k in ('full_name', 'owner', 'description', 'default_branch', 'private', 'license')})
        elif n == 25:
            import fitz
            files = list(ROOT.glob('不确定环境下多线路公交信号优先在途协同控制方法研究*.pdf'))
            p = fitz.open(files[0])
            rec = dict(kind='local-thesis', source=str(files[0]), text='\n'.join(p[i].get_text() for i in range(4)))
        elif ref['doi'] and not re.search('[\u4e00-\u9fff]', ref['text'].split('. ')[1]):
            url = 'https://api.crossref.org/works/' + quote(ref['doi'], safe='')
            data = get(url).json()['message']
            rec = {k: data.get(k) for k in ('title', 'author', 'container-title', 'volume', 'issue', 'page', 'article-number', 'published', 'published-online', 'published-print', 'DOI', 'URL', 'abstract', 'type')}
            rec.update(kind='crossref', source=url)
        else:
            rec = html_record(url)
        rec['checked'] = DATE
    except Exception as e:
        rec = dict(source=url, error=str(e), checked=DATE)
    return str(n), rec


def fetch():
    refs = read(HERE / '当前文献与引用.json')
    path = HERE / '在线原始记录.json'
    records = read(path) if path.exists() else {}
    arxiv = [r for r in refs if r['arxiv'] and str(r['number']) not in records]
    if arxiv:
        ns = {'a': 'http://www.w3.org/2005/Atom', 'ar': 'http://arxiv.org/schemas/atom'}
        try:
            r = get('https://export.arxiv.org/api/query', params={'id_list': ','.join(v['arxiv'] for v in arxiv), 'max_results': len(arxiv)})
            feed = ET.fromstring(r.content)
            by_id = {}
            for e in feed.findall('a:entry', ns):
                url = e.findtext('a:id', '', ns).replace('http:', 'https:')
                rec = dict(kind='arxiv', source=url,
                           title=' '.join(e.findtext('a:title', '', ns).split()),
                           authors=[a.findtext('a:name', '', ns) for a in e.findall('a:author', ns)],
                           published=e.findtext('a:published', '', ns), updated=e.findtext('a:updated', '', ns),
                           abstract=' '.join(e.findtext('a:summary', '', ns).split()), checked=DATE)
                by_id[re.sub(r'v\d+$', '', url.rsplit('/', 1)[-1])] = rec
            for ref in arxiv:
                records[str(ref['number'])] = by_id.get(re.sub(r'v\d+$', '', ref['arxiv']), dict(error='No arXiv entry', requested=ref['arxiv']))
            write(path.name, records)
            print('arXiv records', len(by_id), flush=True)
        except Exception as e:
            print('arXiv batch failed', str(e), flush=True)
    pending = [r for r in refs if not r['arxiv'] and str(r['number']) not in records]
    with ThreadPoolExecutor(max_workers=6) as pool:
        for future in as_completed([pool.submit(fetch_one, r) for r in pending]):
            n, rec = future.result()
            records[n] = rec
            write(path.name, records)
            print(n, rec.get('kind', rec.get('error', '')), flush=True)
    # Pinned versions require their own page's authors and version date.
    for ref in refs:
        if not re.search(r'v\d+$', ref['arxiv']) or 'version_page' in records.get(str(ref['number']), {}):
            continue
        n = str(ref['number'])
        try:
            rec = html_record('https://arxiv.org/abs/' + ref['arxiv'])
            records.setdefault(n, {})['version_page'] = rec
        except Exception as e:
            records.setdefault(n, {})['version_error'] = str(e)
        write(path.name, records)
    print('Records', len(records), 'errors', [n for n,r in records.items() if 'error' in r], flush=True)


if __name__ == '__main__':
    HERE.mkdir(exist_ok=True)
    {'extract': extract, 'fetch': fetch}[sys.argv[1]]()
