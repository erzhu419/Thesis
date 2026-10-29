"""Rename the eleven displacement symbols; preserve the user's saved XML layout."""
from pathlib import Path
from copy import copy
from zipfile import ZipFile
from lxml import etree as E
import json
import re

A = Path(__file__).resolve().parent
SOURCE = A / 'source_user_layout_snapshot.docx'
LATEST = A.parents[1] / '调整完格式的_全公式显示终校_20261019.docx'
OUTPUT = A.parents[1] / '调整完格式的_公式符号复核_20261019.docx'
M = '{http://schemas.openxmlformats.org/officeDocument/2006/math}'
NS = {'m': M[1:-1]}
IDS = [1794, 1799, 1811, 1812, 1813, 1814, 1817, 1822, 1823, 1826]

assert LATEST.read_bytes() == SOURCE.read_bytes(), 'The user saved a newer version; refresh the snapshot.'
with ZipFile(SOURCE) as z:
    original = z.read('word/document.xml')
    xml = original.decode('utf-8')
    root = E.fromstring(original)
    maths = list(root.iter(M + 'oMath'))
    runs = list(root.iter(M + 'r'))
    raw_runs = list(re.finditer(r'<m:r\b[^>]*>.*?</m:r>', xml, flags=re.S))
    assert len(maths) == 4755 and len(runs) == len(raw_runs)
    replacements, records, added_text_nodes = [], [], 0
    for ident in IDS:
        for run in maths[ident - 1].iter(M + 'r'):
            text = ''.join(run.xpath('./m:t/text()', namespaces=NS))
            if 'd' not in text:
                continue
            style = run.find(M + 'rPr/' + M + 'sty')
            if style is not None and style.get(M + 'val') == 'p':
                assert ident == 1812 and text == 'd'  # Preserve the differential in du.
                continue
            assert text.count('d') == 1 and style is None
            ordinal = runs.index(run)
            match = raw_runs[ordinal]
            raw = match.group()
            value = re.search(r'(<m:t\b[^>]*>)(.*?)(</m:t>)', raw, flags=re.S)
            assert value and value.group(2) == text and run.find(M + 'rPr') is None
            head, tail = raw[:value.start(2)], raw[value.end(2):]
            position = text.index('d')
            prefix, remainder = text[:position], text[position:]
            delta_head = re.sub(r'^(<m:r\b[^>]*>)',
                                r'\1<m:rPr><m:sty m:val="p"/></m:rPr>', head, count=1)
            new = (head + prefix + tail if prefix else '')
            new += delta_head + 'Δ' + tail + head + remainder + tail
            added_text_nodes += 1 + bool(prefix)
            replacements.append((match.start(), match.end(), raw, new))
            records.append({'math_index': ident, 'run_index': ordinal + 1,
                            'before': text, 'after': prefix + 'Δ' + remainder})
    assert len(records) == 11
    for start, end, _, new in reversed(replacements):
        xml = xml[:start] + new + xml[end:]
    # Removing exactly the inserted fragments must recover every source byte.
    offset, undo = 0, []
    for start, end, old, new in replacements:
        position = start + offset
        assert xml[position:position + len(new)] == new
        undo.append((position, position + len(new), old))
        offset += len(new) - len(old)
    reverted = xml
    for start, end, old in reversed(undo):
        reverted = reverted[:start] + old + reverted[end:]
    assert reverted.encode('utf-8') == original
    changed_root = E.fromstring(xml.encode('utf-8'))
    assert len(list(changed_root.iter(M + 'oMath'))) == 4755
    assert len(list(changed_root.iter(M + 't'))) - len(list(root.iter(M + 't'))) == added_text_nodes
    with ZipFile(OUTPUT, 'w') as out:
        for item in z.infolist():
            out.writestr(copy(item), xml.encode('utf-8') if item.filename == 'word/document.xml'
                         else z.read(item.filename))
    with ZipFile(OUTPUT) as out:
        assert out.namelist() == z.namelist()
        assert all(out.read(n) == z.read(n) for n in z.namelist() if n != 'word/document.xml')

(A / 'displacement_changes.json').write_text(json.dumps({
    'source': str(SOURCE), 'output': str(OUTPUT), 'math_objects': 4755,
    'changed_objects': IDS, 'changes': records,
    'scope': 'Only displacement d becomes upright delta plus italic d; differential du retained.',
    'paragraph_layout': 'Unchanged XML outside the eleven replaced math runs.',
}, ensure_ascii=False, indent=2), encoding='utf-8')
print(OUTPUT)
