"""Check preserved objects, numbered targets, and cached cross-references."""
from collections import defaultdict
from pathlib import Path
from zipfile import ZipFile
import json
import re

from lxml import etree as E

from scan import SOURCE, HERE, W, M, NS, content
from repair import OUT


def compact(text):
    return re.sub(r'\s+', '', text).translate(str.maketrans('（）', '()'))


def reference_fields(document):
    active, targets, fields, stack = {}, defaultdict(str), [], []
    for node in document.iter():
        if node.tag == W + 'bookmarkStart':
            active[node.get(W + 'id')] = node.get(W + 'name')
        elif node.tag == W + 'bookmarkEnd':
            active.pop(node.get(W + 'id'), None)
        elif node.tag in (W + 't', M + 't'):
            for name in active.values():
                targets[name] += node.text or ''
            for field in stack:
                if field['result']:
                    field['cache'] += node.text or ''
        elif node.tag == W + 'fldChar':
            kind = node.get(W + 'fldCharType')
            if kind == 'begin':
                stack.append({'code': '', 'cache': '', 'result': False})
            elif kind == 'separate' and stack:
                stack[-1]['result'] = True
            elif kind == 'end' and stack:
                fields.append(stack.pop())
        elif node.tag == W + 'instrText' and stack:
            stack[-1]['code'] += node.text or ''
    errors, checked = [], 0
    for field in fields:
        match = re.search(r'^\s*REF\s+([^\s\\]+)', field['code'])
        if not match:
            continue
        checked += 1
        name = match[1]
        if name not in targets or compact(field['cache']) != compact(targets[name]):
            errors.append({'target': name, 'cache': field['cache'], 'target_text': targets.get(name)})
    return checked, errors


def main():
    with ZipFile(SOURCE) as source, ZipFile(OUT) as revised:
        a = E.fromstring(source.read('word/document.xml'))
        b = E.fromstring(revised.read('word/document.xml'))
        diagram_changes = json.loads((HERE / '图内符号修改记录.json').read_text())
        diagram_parts = {row[key] for row in diagram_changes for key in ('embedding_part', 'preview_part')}
        changed_parts = {n for n in source.namelist() if source.read(n) != revised.read(n)}
        unexpected_parts = changed_parts - diagram_parts - {'word/document.xml'}
        assert not unexpected_parts, unexpected_parts
        assert diagram_parts <= changed_parts, diagram_parts - changed_parts
        aa, bb = a.find(W + 'body'), b.find(W + 'body')
        references_preserved = all(E.tostring(aa[i]) == E.tostring(bb[i]) for i in range(1786, 2002))
        assert references_preserved
        protected = {}
        for tag in ('drawing', 'object', 'sectPr'):
            original = [E.tostring(n) for n in a.iter(W + tag)]
            current = [E.tostring(n) for n in b.iter(W + tag)]
            assert original == current, tag
            protected[tag] = len(current)
        for tag in ('bookmarkStart', 'bookmarkEnd'):
            assert [dict(n.attrib) for n in a.iter(W + tag)] == [dict(n.attrib) for n in b.iter(W + tag)], tag
        field_count, field_errors = reference_fields(b)
        audit = json.loads((HERE / '校订后/编号与引用检查.json').read_text())
        ranges = {}
        for kind in ('equations', 'captions'):
            groups = defaultdict(set)
            for key in audit[kind]:
                prefix, number = key.rsplit('-', 1)
                groups[prefix].add(int(re.match(r'\d+', number)[0]))
            for group, values in groups.items():
                assert values == set(range(1, max(values) + 1)), (kind, group)
            ranges[kind] = {k: max(v) for k, v in groups.items()}
        assert not audit['errors'] and not audit['duplicate_bookmarks']
        assert set(audit['captions']) <= set(audit['caption_uses'])
        result = {
            'package_changes_limited_to_document_and_four_diagrams': True,
            'changed_diagram_parts': sorted(diagram_parts),
            'bibliography_content_and_format_preserved': references_preserved,
            'protected_object_counts': protected,
            'bookmark_attributes_preserved': True,
            'equation_count': len(audit['equations']),
            'figure_count': sum(k.startswith('图') for k in audit['captions']),
            'table_count': sum(k.startswith('表') for k in audit['captions']),
            'all_figures_and_tables_mentioned_in_text': True,
            'numbering_ranges': ranges,
            'REF_caches_checked': field_count,
            'REF_cache_errors': field_errors,
            'structural_reference_errors': audit['errors'],
        }
        (HERE / '成品结构验收.json').write_text(json.dumps(result, ensure_ascii=False, indent=2))
        print(json.dumps(result, ensure_ascii=False, indent=2))
        assert not field_errors


if __name__ == '__main__':
    main()
