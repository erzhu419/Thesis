"""Build eight display-only cases; never launch Word or change the source."""
from copy import deepcopy
from pathlib import Path
from zipfile import ZipFile
import json
from lxml import etree as E

T = Path(__file__).resolve().parent
SOURCE = T.parents[1] / '调整完格式的.docx'
OUTPUT = T / 'ampersand_nor_probe.docx'
W = '{http://schemas.openxmlformats.org/wordprocessingml/2006/main}'
M = '{http://schemas.openxmlformats.org/officeDocument/2006/math}'
labels = ['B-48', 'D-12', 'D-40', 'E-10']

with ZipFile(SOURCE) as source:
    infos = source.infolist()
    parts = {i.filename: source.read(i.filename) for i in infos}
root = E.fromstring(parts['word/document.xml'])
original_nsmap = dict(root.nsmap)
body = root.find(W + 'body')
section = deepcopy(body.find(W + 'sectPr'))
tables = {}
for table in body.findall(W + 'tbl'):
    for row in table.findall(W + 'tr'):
        cells = row.findall(W + 'tc')
        if not cells:
            continue
        label = ''.join(t.text or '' for t in cells[-1].iter(W + 't')).strip()
        if label[1:-1] in labels and label.startswith(('(', '（')):
            assert len(table.findall(W + 'tr')) == 1, label
            tables[label[1:-1]] = table
assert set(tables) == set(labels)
for child in list(body):
    body.remove(child)

pages = []
for label in labels:
    for variant in ['original', 'without_m_nor_on_ampersand_runs']:
        page = len(pages) + 1
        title = E.SubElement(body, W + 'p')
        ppr = E.SubElement(title, W + 'pPr')
        if page > 1:
            E.SubElement(ppr, W + 'pageBreakBefore')
        E.SubElement(ppr, W + 'keepNext')
        E.SubElement(ppr, W + 'ind', {W + 'firstLine': '0'})
        E.SubElement(ppr, W + 'spacing', {W + 'after': '240'})
        run = E.SubElement(title, W + 'r')
        rp = E.SubElement(run, W + 'rPr')
        E.SubElement(rp, W + 'rFonts', {W + 'ascii': 'Times New Roman', W + 'hAnsi': 'Times New Roman'})
        E.SubElement(rp, W + 'sz', {W + 'val': '22'})
        caption = f'Case {page}: {label} - ' + ('Original' if variant == 'original' else 'Only m:nor removed from ampersand runs')
        E.SubElement(run, W + 't').text = caption
        table = deepcopy(tables[label])
        # A paired copy needs distinct internal bookmark identities. Its text,
        # cached number and SEQ field are preserved, and do not target bookmarks.
        id_map = {}
        for mark in table.iter(W + 'bookmarkStart'):
            old_id = mark.get(W + 'id')
            new_id = str(10000 + page * 100 + len(id_map))
            id_map[old_id] = new_id
            mark.set(W + 'id', new_id)
            mark.set(W + 'name', f'probe_{page}_' + mark.get(W + 'name'))
        for mark in table.iter(W + 'bookmarkEnd'):
            mark.set(W + 'id', id_map[mark.get(W + 'id')])
        before_math = [''.join(t.text or '' for t in math.iter(M + 't')) for math in table.iter(M + 'oMath')]
        changed = []
        for math_run in table.iter(M + 'r'):
            text = ''.join(t.text or '' for t in math_run.iter(M + 't'))
            if '&' not in text:
                continue
            props = math_run.find(M + 'rPr')
            normal = props.find(M + 'nor') if props is not None else None
            if variant != 'original' and normal is not None:
                props.remove(normal)
                changed.append(text)
        assert before_math == [''.join(t.text or '' for t in math.iter(M + 't')) for math in table.iter(M + 'oMath')]
        body.append(table)
        pages.append({'expected_page': page, 'label': label, 'variant': variant,
                      'english_title': caption, 'removed_m_nor_count': len(changed),
                      'affected_math_run_texts': changed,
                      'ampersand_run_count': sum('&' in ''.join(t.text or '' for t in r.iter(M + 't')) for r in table.iter(M + 'r')),
                      'eqArr_row_count': [len(eq.findall(M + 'e')) for eq in table.iter(M + 'eqArr')]})
body.append(section)
assert dict(root.nsmap) == original_nsmap
parts['word/document.xml'] = E.tostring(root, encoding='UTF-8', xml_declaration=True, standalone=True)
with ZipFile(OUTPUT, 'w') as output:
    for info in infos:
        output.writestr(info, parts[info.filename])
with ZipFile(SOURCE) as source, ZipFile(OUTPUT) as output:
    changed_parts = [name for name in source.namelist() if source.read(name) != output.read(name)]
assert changed_parts == ['word/document.xml']
report = {'source': str(SOURCE), 'probe': str(OUTPUT), 'expected_page_count': 8,
          'changed_package_parts': changed_parts, 'source_namespace_map_preserved': True,
          'source_math_text_preserved': True, 'Word_started': False,
          'bookmark_note': 'Probe copies use page-specific bookmark names/IDs; original cached formula labels and SEQ codes are preserved.',
          'preview_note': 'Render without updating fields. Page titles identify source formula even if Word updates SEQ caches.',
          'pages': pages}
(T / 'ampersand_nor_probe_pages.json').write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n')
print(json.dumps({'probe': str(OUTPUT), 'pages': len(pages), 'removed_m_nor_total': sum(p['removed_m_nor_count'] for p in pages)}, ensure_ascii=False))
