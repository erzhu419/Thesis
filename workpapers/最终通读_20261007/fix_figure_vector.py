"""Correct only two text records in the existing vector preview and Visio source."""
import struct
from pathlib import Path
from zipfile import ZipFile

from lxml import etree as E

WORK = Path(__file__).resolve().parent
old = '源结构复用与模型选择'
new = '模型生成与置信度加权'
with ZipFile(WORK/'图6-1修改前.vsdx') as source:
    infos = source.infolist()
    parts = {n.filename: source.read(n.filename) for n in infos}
page = E.fromstring(parts['visio/pages/page1.xml'])
ns = {'v': page.nsmap[None]}
label = page.xpath('//v:Shape[@ID="458"]/v:Text', namespaces=ns)[0]
assert ''.join(label.itertext()).strip() == '(a)'+old
label[1].tail = '(b)'
label[2].tail = new+'\n'
parts['visio/pages/page1.xml'] = E.tostring(page, xml_declaration=True, encoding='UTF-8', standalone=True)
with ZipFile(WORK/'图6-1标记校订.vsdx', 'w') as output:
    for n in infos:
        output.writestr(n, parts[n.filename])

data = bytearray((WORK/'图6-1当前嵌入.emf').read_bytes())
matches = {'(a)': [], old: []}
position = 0
while position < len(data):
    kind, size = struct.unpack_from('<II', data, position)
    if kind == 84:  # EMR_EXTTEXTOUTW stores the Unicode text and glyph advances.
        count, offset = struct.unpack_from('<II', data, position+44)
        value = data[position+offset:position+offset+count*2].decode('utf-16le')
        if value in matches:
            matches[value].append((position+offset, count))
    position += size
for before, after in [('(a)', '(b)'), (old, new)]:
    assert len(matches[before]) == 2
    start, count = matches[before][1]
    encoded = after.encode('utf-16le')
    assert len(encoded) == count*2
    data[start:start+len(encoded)] = encoded
(WORK/'图6-1标记校订.emf').write_bytes(data)
print('Lower-panel label corrected; all vector geometry and text metrics retained.')
