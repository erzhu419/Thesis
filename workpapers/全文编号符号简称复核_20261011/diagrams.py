"""Correct native Visio labels without changing diagram geometry."""
from copy import deepcopy
from io import BytesIO
from zipfile import ZipFile
import json

from lxml import etree as E

from scan import HERE, W, content
from repair import OUT

V = '{http://schemas.microsoft.com/office/visio/2012/main}'
R = '{http://schemas.openxmlformats.org/officeDocument/2006/relationships}'
O = '{urn:schemas-microsoft-com:office:office}'
VML = '{urn:schemas-microsoft-com:vml}'

EDITS = {
    '图1-1': {70: [('H20/H20+', 'H2O/H2O+'), ('Important Sampling', 'Importance Sampling'), ('跨域真移', '跨域迁移')]},
    '图2-1': {145: [('S', 's')], 150: [('S', 's')]},
    '图4-1': {53: [('S', 's')], 57: [('S', 's')], 132: [('τ', 'r')],
              260: [('桶温度', '熵温度')], 355: [('t', 'eff')],
              292: [('Q1', 'Qϕ1')], 293: [('Q2', 'Qϕ2')], 294: [('QK', 'QϕK')]},
    '图5-2': {2: [('iQL', 'IQL')], 79: [('H20+', 'H2O+')],
              66: [('α', 'a')], 67: [('α', 'a'), ('φ', 'ϕ')]},
}


def replace_label(node, old, new):
    slots = [(node, 'text')] + [(child, 'tail') for child in node]
    text = ''.join(getattr(n, slot) or '' for n, slot in slots)
    assert old in text, (old, text)
    start, end = text.index(old), text.index(old) + len(old)
    offset, inserted = 0, False
    for n, slot in slots:
        value = getattr(n, slot) or ''
        right = offset + len(value)
        if right > start and offset < end:
            left_cut, right_cut = max(0, start - offset), min(len(value), end - offset)
            setattr(n, slot, value[:left_cut] + (new if not inserted else '') + value[right_cut:])
            inserted = True
        offset = right


def prepare():
    destination = HERE / '图内符号校订'
    destination.mkdir(exist_ok=True)
    manifest = []
    with ZipFile(OUT) as docx:
        doc = E.fromstring(docx.read('word/document.xml'))
        body = doc.find(W + 'body')
        rels = {r.get('Id'):r.get('Target') for r in E.fromstring(docx.read('word/_rels/document.xml.rels'))}
        for i, p in enumerate(body):
            obj = p.find('.//' + W + 'object')
            if obj is None:
                continue
            number = content(body[i + 1]).split()[0]
            if number not in EDITS:
                continue
            embedding = 'word/' + rels[obj.find(O + 'OLEObject').get(R + 'id')]
            image = 'word/' + rels[obj.find('.//' + VML + 'imagedata').get(R + 'id')]
            original = docx.read(embedding)
            (destination / (number + '_修改前.vsdx')).write_bytes(original)
            with ZipFile(BytesIO(original)) as archive:
                page = E.fromstring(archive.read('visio/pages/page1.xml'))
                shapes = {int(s.get('ID')):s for s in page.iter(V + 'Shape')}
                changes = []
                for shape_id, edits in EDITS[number].items():
                    shape = shapes[shape_id]
                    text = shape.find(V + 'Text')
                    before = ''.join(text.itertext())
                    for old, new in edits:
                        replace_label(text, old, new)
                    if number == '图4-1' and shape_id in (292, 293, 294):
                        text.clear()
                        for ix, value in ((0, 'Q'), (1, 'ϕ' + {292:'1',293:'2',294:'K'}[shape_id]), (2, '\n')):
                            cp = E.SubElement(text, V+'cp', IX=str(ix))
                            cp.tail = value
                        for row in shape.findall(V+'Section[@N="Character"]/'+V+'Row'):
                            cell = row.find(V+'Cell[@N="Pos"]')
                            if cell is None:
                                cell = E.SubElement(row, V+'Cell', N='Pos')
                            cell.set('V', '2' if row.get('IX') == '1' else '0')
                            cell.attrib.pop('F', None)
                            style = row.find(V+'Cell[@N="Style"]')
                            if style is not None:
                                style.set('V', '2')
                                style.attrib.pop('F', None)
                    if number == '图4-1' and shape_id == 260:
                        section = shape.find(V + 'Section[@N="Character"]')
                        row = deepcopy(section.find(V + 'Row[@IX="1"]'))
                        row.set('IX', '3')
                        row.find(V + 'Cell[@N="Pos"]').set('V', '2')
                        section.append(row)
                        cp = E.Element(V + 'cp', IX='3')
                        cp.tail = 'ent'
                        text.find(V + 'cp[@IX="1"]').addnext(cp)
                    changes.append({'shape': shape_id, 'before': before, 'after': ''.join(text.itertext())})
                path = destination / (number + '_校订.vsdx')
                with ZipFile(path, 'w') as output:
                    for info in archive.infolist():
                        output.writestr(info, E.tostring(page, encoding='utf-8', xml_declaration=True) if info.filename == 'visio/pages/page1.xml' else archive.read(info.filename))
            manifest.append({'figure': number, 'embedding_part': embedding, 'preview_part': image,
                             'source': str(path), 'changes': changes})
    (HERE / '图内符号修改记录.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2))
    print(json.dumps([{'figure': m['figure'], 'shapes':len(m['changes'])} for m in manifest], ensure_ascii=False))


def integrate():
    manifest = json.loads((HERE / '图内符号修改记录.json').read_text())
    replacement = {}
    for row in manifest:
        path = HERE / '图内符号校订' / (row['figure'] + '_校订.vsdx')
        replacement[row['embedding_part']] = path.read_bytes()
        replacement[row['preview_part']] = path.with_suffix('.emf').read_bytes()
    with ZipFile(OUT) as archive:
        parts = [(deepcopy(info), replacement.get(info.filename, archive.read(info.filename))) for info in archive.infolist()]
    with ZipFile(OUT, 'w') as archive:
        for info, data in parts:
            archive.writestr(info, data)
    print('Updated four embedded native diagrams and their EMF previews.')


if __name__ == '__main__':
    import sys
    integrate() if len(sys.argv)>1 and sys.argv[1]=='integrate' else prepare()
