"""Check verbatim replacements and the explicitly limited OOXML edit surface."""
from collections import Counter
import json
from pathlib import Path
import re
from zipfile import ZipFile

from lxml import etree as E

from revise import SOURCE, OUT, HERE, NS, W, helper, visible


def tree(node):
    return (node.tag, tuple(sorted(node.attrib.items())), node.text,
            tuple(tree(child) for child in node))


def main():
    log = json.loads((HERE/'修改记录.json').read_text())
    changed = set(log['replaced_blocks'])
    counts = Counter(item['before_index'] for item in log['insertions'])
    with ZipFile(SOURCE) as source, ZipFile(OUT) as output:
        a, b = source.read('word/document.xml'), output.read('word/document.xml')
        old = E.fromstring(a).find('w:body', NS)
        new = E.fromstring(b).find('w:body', NS)
        spans_a, spans_b = helper.body_spans(a), helper.body_spans(b)
        assert len(new) == len(old)+sum(counts.values())
        preserved, toc_changes, correspondence = 0, [], {}
        cursor = 0
        for i, node in enumerate(old):
            cursor += counts[i]
            correspondence[i] = cursor
            target = new[cursor]
            style = node.find('w:pPr/w:pStyle', NS)
            toc = style is not None and style.get(W+'val') in ('TOC1','TOC2','TOC3')
            if i not in changed:
                x, y = spans_a[i], spans_b[cursor]
                same = a[x[0]:x[1]] == b[y[0]:y[1]]
                if not same and toc:
                    values_a = node.xpath('.//w:t/text()', namespaces=NS)
                    values_b = target.xpath('.//w:t/text()', namespaces=NS)
                    assert values_a[:-1] == values_b[:-1]
                    assert values_b[-1].isdigit()
                    toc_changes.append(i)
                else:
                    assert same, ('unexpected edit',i)
                    preserved += 1
            if node.tag == W+'tbl' and node.find('.//m:oMath',NS) is not None and i != 2652:
                assert [tree(m) for m in node.findall('.//m:oMath',NS)] == [tree(m) for m in target.findall('.//m:oMath',NS)], ('changed equation', i)
            cursor += 1
        abstract_lines = (HERE.parents[1]/'markdown/摘要第2至4项_中英文替换稿_20261011.md').read_text().splitlines()
        expected = [line for line in abstract_lines if re.match(r'^（[234]）',line)]
        expected += [line for line in abstract_lines if re.match(r'^\([234]\)',line)]
        for index, text in zip(log['abstract_replacements'],expected):
            assert visible(new[correspondence[index]]) == text
            assert new[correspondence[index]].find('.//w:numPr',NS) is None
        assert log['body_changed_indices'] == [996,1022,1535,1543]
        assert all(i in log['abstract_replacements'] or i==220 or i in (996,1022,1535,1543) or i>=2002 for i in changed)
        modified_parts = [name for name in source.namelist() if source.read(name)!=output.read(name)]
        assert set(modified_parts) == {'word/document.xml','word/header16.xml'}
        assert source.read('word/header16.xml').replace('探索性比较'.encode(),'补充实验'.encode()) == output.read('word/header16.xml')
        bookmarks = new.xpath('.//w:bookmarkStart/@w:name',namespaces=NS)
        new_targets = ['supp_'+number.replace('-','_') for number in log['new_equations']]
        assert all(bookmarks.count(name)==1 for name in new_targets)
        instructions = new.xpath('.//w:instrText/text()',namespaces=NS)
        references = [m.group(1) for value in instructions for m in re.finditer(r'REF\s+(supp_\w+)',value)]
        assert references and set(references)<=set(new_targets)
        assert set(new_targets)<=set(bookmarks)
        assert len(new.xpath('.//w:sectPr',namespaces=NS)) == len(old.xpath('.//w:sectPr',namespaces=NS))
        result = {
            'abstracts_verbatim':6, 'original_body_blocks':len(old),
            'preserved_blocks_byte_identical':preserved,
            'toc_page_cache_changes':len(toc_changes),
            'main_body_paragraphs_changed':log['body_changed_indices'],
            'only_modified_package_parts':modified_parts,
            'existing_display_equations_unchanged_except':['E-27'],
            'new_native_equations':log['new_equations'],
            'supplementary_reference_fields':len(references),
            'pictures_tables_references_styles_sections_preserved_except_declared_edits':True,
        }
    (HERE/'内容与范围验收.json').write_text(json.dumps(result,ensure_ascii=False,indent=2))
    print(json.dumps(result,ensure_ascii=False,indent=2))


if __name__=='__main__':
    main()
