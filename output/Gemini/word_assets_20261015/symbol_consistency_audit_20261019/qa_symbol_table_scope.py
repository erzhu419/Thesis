"""Read-only check: symbol section and automatic TOC digits are the only exclusions.

CLI: .venv/bin/python qa_symbol_table_scope.py new.docx [--report report.json]
"""
from pathlib import Path
from zipfile import ZipFile
from lxml import etree as E
import argparse
import json
import re

A = Path(__file__).resolve().parent
SOURCE = A / 'source_snapshot.docx'
W = '{http://schemas.openxmlformats.org/wordprocessingml/2006/main}'
M = '{http://schemas.openxmlformats.org/officeDocument/2006/math}'


def plain(node):
    return ''.join(t.text or '' for t in node.iter() if t.tag in (W + 't', M + 't'))


def exclude_authorized(root):
    body = root.find(W + 'body')
    starts = [i for i, n in enumerate(body) if n.tag == W + 'p' and plain(n).strip() == '主要符号说明']
    assert len(starts) == 1, 'Symbol heading must occur exactly once'
    start = starts[0]
    stops = [i for i, n in enumerate(body) if i > start and n.tag == W + 'p'
             and re.sub(r'\s', '', plain(n)) == '目录']
    assert stops, 'TOC heading after symbol section is missing'
    stop = stops[0]
    excluded = list(body)[start:stop]
    section_info = {'body_blocks': len(excluded), 'tables': sum(n.tag == W + 'tbl' for n in excluded),
                    'math_objects': sum(len(list(n.iter(M + 'oMath'))) for n in excluded)}
    for node in excluded:
        body.remove(node)
    stack, pages = [], {}
    for node in root.iter():
        if node.tag == W + 'fldChar':
            kind = node.get(W + 'fldCharType')
            if kind == 'begin':
                stack.append({'code': '', 'result': False, 'texts': []})
            elif kind == 'separate' and stack:
                stack[-1]['result'] = True
            elif kind == 'end' and stack:
                field = stack.pop()
                match = re.search(r'\bPAGEREF\s+(_Toc\d+)\b', field['code'], re.I)
                if match:
                    key = match[1].casefold()
                    assert len(field['texts']) == 1 and key not in pages, ('Unexpected TOC cache', key)
                    value = field['texts'][0].text or ''
                    assert value.isdigit(), ('TOC cache is not a page number', key, value)
                    pages[key] = value
                    field['texts'][0].text = 'AUTHORIZED_TOC_PAGE_CACHE'
        elif node.tag == W + 'instrText' and stack and not stack[-1]['result']:
            stack[-1]['code'] += node.text or ''
        elif node.tag == W + 't' and stack and stack[-1]['result']:
            stack[-1]['texts'].append(node)
    return section_info, pages


def xml_differences(old, new, limit=20):
    found = []
    def visit(a, b, path):
        if len(found) >= limit:
            return
        for what, x, y in [('tag', a.tag, b.tag), ('attributes', dict(a.attrib), dict(b.attrib)),
                           ('text', a.text, b.text), ('tail', a.tail, b.tail)]:
            if x != y:
                found.append({'path': path, 'kind': what, 'before': x, 'after': y})
                if len(found) >= limit:
                    return
        if len(a) != len(b):
            found.append({'path': path, 'kind': 'child_count', 'before': len(a), 'after': len(b)})
        for i, (x, y) in enumerate(zip(a, b), 1):
            visit(x, y, f'{path}/{E.QName(x).localname}[{i}]')
    visit(old, new, '/document')
    return found


def check_docx_scope(new_docx, source=SOURCE):
    """Return a compact report; neither DOCX is modified. No hashes are used."""
    source, new_docx = Path(source).resolve(), Path(new_docx).resolve()
    with ZipFile(source) as old_zip, ZipFile(new_docx) as new_zip:
        old_names, new_names = set(old_zip.namelist()), set(new_zip.namelist())
        changed = [n for n in sorted(old_names & new_names) if old_zip.read(n) != new_zip.read(n)]
        media = [n for n in old_names | new_names if n.startswith('word/media/')]
        media_same = all(n in old_names & new_names and old_zip.read(n) == new_zip.read(n) for n in media)
        old = E.fromstring(old_zip.read('word/document.xml'))
        new = E.fromstring(new_zip.read('word/document.xml'))
    old_section, old_pages = exclude_authorized(old)
    new_section, new_pages = exclude_authorized(new)
    differences = xml_differences(old, new)
    same_targets = old_pages.keys() == new_pages.keys()
    other_parts_same = old_names == new_names and not set(changed) - {'word/document.xml'}
    return {'passed': not differences and same_targets and other_parts_same and media_same,
            'read_only': True, 'source': str(source), 'new_docx': str(new_docx),
            'excluded_symbol_section_before_after': [old_section, new_section],
            'TOC_cache_count_before_after': [len(old_pages), len(new_pages)],
            'TOC_bookmarks_identical': same_targets,
            'changed_TOC_page_caches': sum(old_pages[k] != new_pages[k] for k in old_pages.keys() & new_pages.keys()),
            'retained_document_XML_exact': not differences,
            'retained_math_objects_before_after': [len(list(r.iter(M + 'oMath'))) for r in (old, new)],
            'all_image_data_byte_identical': media_same, 'image_parts': len(media),
            'other_ZIP_part_data_byte_identical': other_parts_same,
            'changed_ZIP_parts': changed, 'added_parts': sorted(new_names - old_names),
            'removed_parts': sorted(old_names - new_names), 'actual_XML_differences_first_20': differences}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('new_docx', type=Path)
    parser.add_argument('--source', type=Path, default=SOURCE)
    parser.add_argument('--report', type=Path, default=A / 'symbol_table_scope_qa.json')
    args = parser.parse_args()
    result = check_docx_scope(args.new_docx, args.source)
    args.report.write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps(result, ensure_ascii=False))
    raise SystemExit(0 if result['passed'] else 1)
