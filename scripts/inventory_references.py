"""Inventory current paper citations without modifying their bibliographies."""

import csv
import json
import re
import unicodedata
from pathlib import Path
from urllib.parse import unquote

import fitz
import pypandoc


ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT.parent
OUT = ROOT / 'workpapers/reference_inventory'
PAPERS = {
    'Categorical Feature': None,
    'RE-SAC': 'RE-SAC/paper/paper.tex',
    'BAPR': 'BAPR/paper/paper.tex',
    'CS-BAPR': 'CS-BAPR/paper/paper.tex',
    'TransitDuet': 'TransitDuet/paper/main.tex',
    'offline-sumo': 'offline-sumo/paper/main.tex',
    'H2O+': 'sumo-rl/H2Oplus/paper/main.tex',
    'MC-WM': 'MC-WM/paper/main.tex',
    'BAMOR': 'BAMOR/paper/paper.tex',
    'BAPR-HRO': 'BAPR-HRO/paper/paper_opre.tex',
    'CFCMT': 'CFCMT/paper/main.tex',
    'MEIS': 'MEIS/paper/main.tex',
    'Asumption Agent': 'Asumption Agent/paper/main.tex',
}
CORE = set(list(PAPERS)[:8])
ITEM = re.compile(r'\\bibitem(?:\[[\s\S]*?\])?\s*\{([^{}]+)\}')
CITE = re.compile(r'\\(?:cite|citep|citet|citealp|citeauthor|citeyear|citeyearpar|parencite|textcite)\*?(?:\[[^\]]*\]){0,2}\s*\{([^{}]+)\}')


def plain(text):
    return ' '.join(pypandoc.convert_text(text, 'plain', format='latex').split())


def normalized(text):
    text = re.sub(r'<[^>]+>', '', text)
    text = text.replace('\\phi', 'phi').replace('\u03d5', 'phi').replace('\u03c6', 'phi')
    if text.startswith('H2O+:'):
        text = re.sub(r'\bRL\b', 'reinforcement learning', text)
    return ''.join(c for c in unicodedata.normalize('NFKD', text).casefold() if c.isalnum())


def doi(text):
    match = re.search(r'10\.\d{4,9}/[^\s<>"{}]+', unquote(text))
    return match.group().lower().rstrip('.,;') if match else ''


def manual_entry(key, raw):
    quoted = re.search(r"``([\s\S]+?)''", raw)
    if quoted:
        title = plain(quoted[1]).rstrip(',.')
    elif re.search(r'\(\d{4}\)', raw):
        title = plain(re.split(r'\(\d{4}\)', raw, maxsplit=1)[1].split('\\emph')[0]).rstrip('.')
    elif '\\newblock' in raw:
        title = plain(raw.split('\\newblock')[1]).rstrip('.')
    else:
        emph = re.search(r'\\emph\{(.+?)\}\.\\hskip', raw, re.S)
        assert emph, (key, raw)
        title = plain(emph[1])
    entry = {'id': key, 'title': title, 'raw_reference': plain(raw)}
    if found := doi(raw):
        entry['DOI'] = found
    return entry


def paper_records(name, relative):
    main = BASE / relative
    visited = set()
    texts = []

    def visit(path):
        path = path.resolve()
        if path in visited:
            return
        visited.add(path)
        text = re.sub(r'(?<!\\)%[^\n]*', '', path.read_text())
        texts.append(text)
        for target in re.findall(r'\\(?:input|include)\s*\{([^{}]+)\}', text):
            target = main.parent / target
            if not target.suffix:
                target = target.with_suffix('.tex')
            visit(target)
    visit(main)
    text = '\n'.join(texts)
    cited = {key.strip() for group in CITE.findall(text) for key in group.split(',')}
    entries = {}
    for group in re.findall(r'\\bibliography\{([^{}]+)\}', text):
        for filename in group.split(','):
            path = main.parent / filename
            if not path.suffix:
                path = path.with_suffix('.bib')
            entries.update({r['id']:r for r in json.loads(pypandoc.convert_file(str(path), 'csljson', format='biblatex'))})
    # Manually included bibliographies are authoritative; the main .bbl is a fallback.
    bbl = main.with_suffix('.bbl')
    item_texts = [t for t in texts if '\\bibitem' in t]
    if not item_texts and bbl.exists():
        item_texts = [bbl.read_text()]
    listed = set()
    for item_text in item_texts:
        matches = list(ITEM.finditer(item_text))
        for i, match in enumerate(matches):
            key = match[1]
            listed.add(key)
            raw = item_text[match.end():matches[i+1].start() if i+1<len(matches) else len(item_text)]
            raw = raw.split('\\end{thebibliography}')[0]
            if key in cited and key not in entries:
                entries[key] = manual_entry(key, raw)
    missing = cited - entries.keys()
    assert not missing, (name, sorted(missing))
    rows = [dict(entries[key], project=name, local_key=key, source=str(main.relative_to(BASE))) for key in sorted(cited)]
    return rows, dict(project=name, cited=len(cited), listed=len(listed),
                      uncited_bibliography_keys=sorted(listed-cited))


def categorical_records():
    source = Path('/mnt/d/Paper/Single Agent Robust Deep Reinforcement Learning for Bus Fleet Control.pdf')
    pdf = fitz.open(source)
    text = '\n'.join(p.get_text(flags=fitz.TEXT_DEHYPHENATE) for p in pdf)
    text = text.split('\nReferences\n')[-1]
    text = re.sub(r'(?m)^\d+\s*$', '', text)
    text = re.sub(r'(?<=[A-Za-z])-\s*\n\s*(?=[a-z])', '', text)
    matches = list(re.finditer(r'(?m)^\[(\d+)\]\s*', text))
    rows = []
    for i, match in enumerate(matches):
        raw = ' '.join(text[match.end():matches[i+1].start() if i+1<len(matches) else len(text)].split())
        quoted = re.search('\u201c(.+?)\u201d', raw)
        if quoted:
            title = quoted[1].rstrip(',')
        else:
            assert match[1] == '6', raw
            title = 'Public Transit Planning and Operation: Modeling, Practice and Behavior'
        rows.append(dict(id='pdf'+match[1], local_key='pdf'+match[1], project='Categorical Feature',
                         title=title, DOI=doi(raw), raw_reference=raw, source=str(source)))
    assert len(rows) == 27
    return rows, dict(project='Categorical Feature', cited=len(rows), listed=len(rows), uncited_bibliography_keys=[])


def main():
    OUT.mkdir(exist_ok=True)
    records, counts = [], []
    for name, relative in PAPERS.items():
        rows, info = paper_records(name, relative) if relative else categorical_records()
        records.extend(rows)
        counts.append(info)
    thesis = json.loads((ROOT/'manuscript/references.json').read_text())
    all_rows = records + [dict(r, project='Thesis', local_key=r['id']) for r in thesis]
    groups = []
    for row in all_rows:
        title = normalized(row['title'])
        identifier = doi(row.get('DOI', ''))
        matches = [i for i,g in enumerate(groups) if title in g['titles'] or (identifier and identifier in g['dois'])]
        group = dict(titles={title}, dois={identifier} if identifier else set(), rows=[row])
        for i in reversed(matches):
            old = groups.pop(i)
            for k in group:
                group[k].update(old[k]) if isinstance(group[k],set) else group[k].extend(old[k])
        groups.append(group)
    library = []
    for index, group in enumerate(sorted(groups, key=lambda g:normalized(g['rows'][0]['title'])), 1):
        members = group['rows']
        projects = sorted({r['project'] for r in members})
        selected = next((r for r in members if r['project']=='Thesis'), members[0])
        library.append(dict(number=index, title=selected['title'], dois=sorted(group['dois']),
                            projects=projects, in_thesis='Thesis' in projects,
                            core_source=bool(CORE.intersection(projects)), members=members))
    summary = dict(papers=counts, paper_citation_occurrences=len(records),
                   unique_paper_references=sum(any(p!='Thesis' for p in r['projects']) for r in library),
                   unique_core_references=sum(r['core_source'] for r in library),
                   current_thesis_references=len(thesis),
                   paper_references_not_in_thesis=sum(not r['in_thesis'] for r in library),
                   core_references_not_in_thesis=sum(r['core_source'] and not r['in_thesis'] for r in library),
                   matching='DOI or normalized title, including phi typography and the observed H2O+ RL abbreviation; metadata not reverified; remaining variant titles require review',
                   chapter_unique_citations={p.name:len(set(re.findall(r'@([A-Za-z0-9_:.-]+)',p.read_text())))
                                            for p in sorted((ROOT/'manuscript/chapters').glob('*.md'))})
    (OUT/'inventory.json').write_text(json.dumps(dict(summary=summary, references=library),ensure_ascii=False,indent=2)+'\n')
    with (OUT/'references.csv').open('w',newline='',encoding='utf-8-sig') as stream:
        writer=csv.DictWriter(stream,fieldnames=['number','title','DOI','projects','in_thesis','core_source'])
        writer.writeheader()
        for r in library:
            writer.writerow({k:r[k] for k in ['number','title','in_thesis','core_source']} |
                            dict(DOI='; '.join(r['dois']),projects='; '.join(r['projects'])))
    print(json.dumps(summary,ensure_ascii=False,indent=2))


if __name__=='__main__':
    main()
