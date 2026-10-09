"""Replace six Word figures without changing prose, captions or equations."""
import copy
import io
import json
import re
import shutil
from pathlib import Path
from xml.etree import ElementTree as ET
from zipfile import ZipFile

from PIL import Image

ROOT = Path('/home/erzhu419/mine_code/Thesis')
WORK = Path(__file__).resolve().parent
SOURCE = ROOT / 'output/排版完成_F2联动校订_20261007.docx'
OUTPUT = ROOT / 'output/排版完成_六图替换_20261007.docx'
NUMBERS = ['1-1', '2-1', '2-3', '4-1', '5-2', '6-1']
NS = {
    'w': 'http://schemas.openxmlformats.org/wordprocessingml/2006/main',
    'wp': 'http://schemas.openxmlformats.org/drawingml/2006/wordprocessingDrawing',
    'a': 'http://schemas.openxmlformats.org/drawingml/2006/main',
    'pic': 'http://schemas.openxmlformats.org/drawingml/2006/picture',
    'r': 'http://schemas.openxmlformats.org/officeDocument/2006/relationships',
}
W, R = '{' + NS['w'] + '}', '{' + NS['r'] + '}'


def text(node):
    return ''.join(t.text or '' for t in node.iter(W + 't'))


def bounds(image, number):
    if number == '1-1':
        return (0, 0, image.width, image.height)
    # Ignore the scan's outer page frame, not the diagram's own panel borders.
    mask = image.convert('L').point(lambda p: 255 if p < 230 else 0)
    border = round(min(image.size) * .02)
    for box in [(0, 0, image.width, border),
                (0, image.height - border, image.width, image.height),
                (0, 0, border, image.height),
                (image.width - border, 0, image.width, image.height)]:
        mask.paste(0, box)
    left, top, right, bottom = mask.getbbox()
    return (max(0, left - 12), max(0, top - 12),
            min(image.width, right + 12), min(image.height, bottom + 12))


shutil.copy2(SOURCE, WORK / '修改前.docx')
with ZipFile(SOURCE) as z:
    infos = z.infolist()
    parts = {i.filename: z.read(i.filename) for i in infos}
original_xml = parts['word/document.xml']
document = ET.fromstring(original_xml)
before = copy.deepcopy(document)
body = document.find('w:body', NS)
relationships = ET.fromstring(parts['word/_rels/document.xml.rels'])
rel_by_id = {node.get('Id'): node for node in relationships}
for _, (prefix, uri) in ET.iterparse(io.BytesIO(original_xml), events=['start-ns']):
    ET.register_namespace(prefix, uri)

raw_drawings = list(re.finditer(rb'<w:drawing\b[\s\S]*?</w:drawing>', original_xml))
drawings = list(document.iter(W + 'drawing'))
assert len(raw_drawings) == len(drawings)
substitutions, renamed, records = {}, {}, []

for number in NUMBERS:
    captions = [(i, p) for i, p in enumerate(body)
                if text(p).startswith('图' + number + ' ')]
    assert len(captions) == 1, number
    index, caption = captions[0]
    candidates = body[index - 1].findall('.//w:drawing', NS)
    assert len(candidates) == 1, number
    drawing = candidates[0]
    blip = drawing.find('.//a:blip', NS)
    rid = blip.get(R + 'embed')
    assert sum(b.get(R + 'embed') == rid for b in document.findall('.//a:blip', NS)) == 1
    old_target = rel_by_id[rid].get('Target')
    assert sum(n.get('Target') == old_target for n in relationships) == 1
    jpg = ROOT / 'output/figure_redesign' / ('图' + number + '.jpg')
    jpg_bytes = jpg.read_bytes()
    image = Image.open(io.BytesIO(jpg_bytes))
    box = bounds(image, number)
    left, top, right, bottom = box
    crop = {'l': round(left / image.width * 100000),
            't': round(top / image.height * 100000),
            'r': round((image.width - right) / image.width * 100000),
            'b': round((image.height - bottom) / image.height * 100000)}
    extent = drawing.find('.//wp:extent', NS)
    width = int(extent.get('cx'))
    height = round(width * (bottom - top) / (right - left))
    extent.set('cy', str(height))
    drawing.find('.//a:xfrm/a:ext', NS).set('cy', str(height))
    fill = drawing.find('.//pic:blipFill', NS)
    old_crop = fill.find('a:srcRect', NS)
    if old_crop is not None:
        fill.remove(old_crop)
    if any(crop.values()):
        fill.insert(1, ET.Element('{' + NS['a'] + '}srcRect',
                                  {key: str(value) for key, value in crop.items()}))
    fill.find('a:stretch/a:fillRect', NS).attrib.clear()
    drawing.find('.//wp:docPr', NS).set('descr', text(caption))
    drawing.find('.//pic:cNvPr', NS).set('descr', text(caption))
    new_target = str(Path(old_target).with_suffix('.jpg'))
    assert 'word/' + new_target not in parts
    rel_by_id[rid].set('Target', new_target)
    renamed['word/' + old_target] = ('word/' + new_target, jpg_bytes)
    substitutions[drawings.index(drawing)] = ET.tostring(drawing, encoding='utf-8')
    records.append({'figure': '图' + number, 'caption': text(caption),
                    'image': str(jpg), 'source_pixels': image.size,
                    'visible_box_pixels': box, 'word_crop': crop,
                    'width_cm': width / 360000, 'height_cm': height / 360000})

xml = original_xml
for index in sorted(substitutions, reverse=True):
    match = raw_drawings[index]
    xml = xml[:match.start()] + substitutions[index] + xml[match.end():]
parts['word/document.xml'] = xml
ET.register_namespace('', 'http://schemas.openxmlformats.org/package/2006/relationships')
parts['word/_rels/document.xml.rels'] = ET.tostring(relationships, encoding='utf-8', xml_declaration=True)
types = ET.fromstring(parts['[Content_Types].xml'])
CT = 'http://schemas.openxmlformats.org/package/2006/content-types'
assert not any(t.get('PartName') in ['/' + p for p in renamed] for t in types)
if not any(t.get('Extension') == 'jpg' for t in types):
    types.insert(0, ET.Element('{' + CT + '}Default', {'Extension': 'jpg', 'ContentType': 'image/jpeg'}))
ET.register_namespace('', CT)
parts['[Content_Types].xml'] = ET.tostring(types, encoding='utf-8', xml_declaration=True)
with ZipFile(OUTPUT, 'w') as z:
    for info in infos:
        if info.filename in renamed:
            path, data = renamed[info.filename]
            copied = copy.copy(info)
            copied.filename = path
            z.writestr(copied, data)
        else:
            z.writestr(info, parts[info.filename])

after = ET.fromstring(xml)
assert text(before) == text(after), 'Manuscript text changed'
for tag in ['{http://schemas.openxmlformats.org/officeDocument/2006/math}oMath',
            W + 'sectPr', W + 'instrText', W + 'bookmarkStart', W + 'bookmarkEnd']:
    assert [ET.tostring(n) for n in before.iter(tag)] == [ET.tostring(n) for n in after.iter(tag)], tag
with ZipFile(OUTPUT) as z:
    assert sum(p.startswith('word/media/') for p in z.namelist()) == 63
    for old_path, (new_path, data) in renamed.items():
        assert z.read(new_path) == data
    for path, data in parts.items():
        if path not in renamed and path not in {'word/document.xml', 'word/_rels/document.xml.rels', '[Content_Types].xml'}:
            assert z.read(path) == data, path
(WORK / '替换记录.json').write_text(json.dumps({'source': str(SOURCE), 'output': str(OUTPUT),
    'figures': records, 'prose_captions_equations_fields_preserved': True}, ensure_ascii=False, indent=2), encoding='utf-8')
print(json.dumps({'output': str(OUTPUT), 'figures': records}, ensure_ascii=False, indent=2))
