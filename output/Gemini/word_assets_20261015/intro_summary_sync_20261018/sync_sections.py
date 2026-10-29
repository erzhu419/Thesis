"""Sync only the user's ten revised regions, retaining native Word formatting."""
from pathlib import Path
from zipfile import ZipFile, ZIP_DEFLATED
from lxml import etree as E
from difflib import SequenceMatcher
from collections import defaultdict, deque
import ast, copy, importlib.util, json, re
import pypandoc

A = Path(__file__).resolve().parent
R = A.parents[1]
PREV = A.parent / 'md_sync_latest_20261016'
SOURCE = Path('/mnt/d/Paper/Thesis/Gemini/重构论文_替换库二次修订_20261016.md')
WORD = Path('/mnt/d/Paper/Thesis/Gemini/重构论文_格式恢复_20261016.docx')
OUT = R / '重构论文_导言小结同步_20261018.docx'
spec = importlib.util.spec_from_file_location('native_helpers', PREV / 'native_helpers.py')
h = importlib.util.module_from_spec(spec)
spec.loader.exec_module(h)
W, M, P = h.W, h.M, h.P
text = h.text
def norm(s):
    return h.norm(s).replace('ϕ', 'φ')

before = (PREV / '最新MD.md').read_text()
after = SOURCE.read_text()
nums = json.loads((A.parent / 'terminal_cleanup_20261015/reference_numbers_final.json').read_text())

def region(s, key):
    if key.startswith('intro'):
        chapter = key[-1]
        m = re.search(r'^# 第' + chapter + r'章[^\n]*\n', s, re.M)
        end = re.search(r'^## ', s[m.end():], re.M)
        return s[m.end():m.end() + end.start()]
    if key.startswith('summary'):
        chapter = key[-1]
        m = re.search(r'^## ' + chapter + r'\.5 本章小结[^\n]*\n', s, re.M)
        end = re.search(r'^#{1,2} ', s[m.end():], re.M)
        return s[m.end():m.end() + end.start()]
    if key == 'B29':
        m = re.search(r'\\tag\{B-29\}\$\$\n\n([^\n]+)', s)
        return m[1]
    if key == 'E31':
        m = re.search(r'([^\n]+)\n\n\$\$[^\n]*\\tag\{E-31\}', s)
        return m[1]
    raise AssertionError(key)

keys = [f'{k}{n}' for n in range(3, 7) for k in ('intro', 'summary')] + ['B29', 'E31']
regions = [(k, region(before, k), region(after, k)) for k in keys]
assert all(a != b for _, a, b in regions)
# This detects unrelated source edits before any Word mutation.
covered = before
for key, old, new in regions:
    assert covered.count(old) == 1, key
    covered = covered.replace(old, new, 1)
assert re.sub(r'\s+', '', covered) == re.sub(r'\s+', '', after), 'Source changed outside requested regions'

def fragment_docx(which):
    pieces = []
    for k, old, new in regions:
        s = old if which == 'before' else new
        cites = re.compile(r'\[([A-Za-z]+\d{4}[\w-]*(?:\s*,\s*[A-Za-z]+\d{4}[\w-]*)*)\]')
        s = cites.sub(lambda m: '^\\[' + ','.join('[' + str(nums[key.strip()]) + '](#latest_ref_' + str(nums[key.strip()]) + ')' for key in m[1].split(',')) + '\\]^', s)
        s = re.sub(r'^---\s*$', '', s, flags=re.M)
        pieces += ['SYNC_SLOT_' + k, s]
    target = A / (which + '_donor.docx')
    pypandoc.convert_text('\n\n'.join(pieces), 'docx', format='markdown+tex_math_dollars-smart', outputfile=str(target))
    result, current = {}, None
    for e in h.read(target).find(W + 'body'):
        value = text(e)
        if value.startswith('SYNC_SLOT_'):
            current = value.removeprefix('SYNC_SLOT_')
            result[current] = []
        elif norm(value):
            assert e.tag == W + 'p', (current, e.tag)
            result[current].append(e)
    return result

old_sections = fragment_docx('before')
new_sections = fragment_docx('after')
with ZipFile(WORD) as z:
    parts = {n: z.read(n) for n in z.namelist()}
root = E.fromstring(parts['word/document.xml'])
body = root.find(W + 'body')
blocks = list(body)
original_xml = [E.tostring(e) for e in blocks]
mapping, cursor = {}, 0
for key in keys:
    mapping[key] = []
    for donor in old_sections[key]:
        matches = [i for i in range(cursor, len(blocks)) if blocks[i].tag == W + 'p' and norm(text(blocks[i])) == norm(text(donor))]
        assert matches, ('Old paragraph not present in latest Word', key, text(donor)[:180])
        i = matches[0]
        mapping[key].append(i)
        cursor = i + 1

inventory = []
for key in keys:
    inventory.append({'key': key, 'old_paragraphs': len(mapping[key]), 'new_paragraphs': len(new_sections[key]), 'word_indices': mapping[key], 'bookmarks': [x.get(W + 'name') for i in mapping[key] for x in blocks[i].iter(W + 'bookmarkStart')], 'field_instructions': [x.text for i in mapping[key] for x in blocks[i].iter(W + 'instrText')]})
(A / 'inventory.json').write_text(json.dumps(inventory, ensure_ascii=False, indent=2) + '\n')
print(json.dumps(inventory, ensure_ascii=False, indent=2))

if __name__ == '__main__':
    import sys
    if '--apply' not in sys.argv:
        raise SystemExit(0)
    # Reuse the previously exercised native paragraph/REF-preservation code.
    module = ast.parse((PREV / 'build_word_md_latest.py').read_text())
    functions = {'fields', 'split_run', 'splice', 'keep_fields', 'sync_p'}
    field_log = []
    def translate_num(p):
        assert p.find(W + 'pPr/' + W + 'numPr') is None
        return None
    exec(compile(ast.Module(body=[n for n in module.body if isinstance(n, ast.FunctionDef) and n.name in functions], type_ignores=[]), str(PREV / 'build_word_md_latest.py'), 'exec'), globals())

    changed, removed, supplied = set(), set(), {}
    for key in keys:
        native = [blocks[i] for i in mapping[key]]
        donors = new_sections[key]
        assert len(donors) <= len(native), (key, len(native), len(donors))
        assert not any(list(p.iter(W + 'drawing')) for p in native)
        # No internal bookmark is discarded when shortening a revised summary.
        assert not any(list(p.iter(W + 'bookmarkStart')) or list(p.iter(W + 'bookmarkEnd')) for p in native[len(donors):]), key
        for p, donor in zip(native, donors):
            old_ppr = E.tostring(p.find(W + 'pPr'))
            sync_p(p, donor)
            assert E.tostring(p.find(W + 'pPr')) == old_ppr, key
            changed.add(blocks.index(p))
        for p in native[len(donors):]:
            removed.add(blocks.index(p))
            body.remove(p)
        supplied[key] = native[:len(donors)]

    # QA: supplied text is exact, and every unrelated native block is unchanged.
    for key in keys:
        assert [norm(text(e)) for e in supplied[key]] == [norm(text(e)) for e in new_sections[key]], key
    for i, p in enumerate(blocks):
        if i not in changed | removed:
            assert E.tostring(p) == original_xml[i], ('Unexpected native change', i)
    assert not any('それぞれの' in text(p) or '統計信息' in text(p) for group in supplied.values() for p in group)
    parts['word/document.xml'] = E.tostring(root, xml_declaration=True, encoding='UTF-8', standalone=True)
    # Keep the exact source Word and Markdown used for this sync.
    (A / '修改前.docx').write_bytes(WORD.read_bytes())
    (A / '本次来源MD.md').write_bytes(SOURCE.read_bytes())
    with ZipFile(OUT, 'w', ZIP_DEFLATED) as z:
        for name, value in parts.items():
            z.writestr(name, value)
    with ZipFile(OUT) as z:
        assert set(z.namelist()) == set(parts)
        assert all(z.read(n) == v for n, v in parts.items())
    report = {'source': str(SOURCE), 'baseline_word': str(WORD), 'output': str(OUT), 'regions': inventory, 'changed_paragraphs': len(changed), 'removed_paragraphs': len(removed), 'unrelated_native_blocks_unchanged': True, 'all_non_document_xml_parts_unchanged': True, 'paragraph_properties_unchanged': True, 'field_log': field_log}
    (A / 'sync_result.json').write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n')
    print('SAVED', OUT)
