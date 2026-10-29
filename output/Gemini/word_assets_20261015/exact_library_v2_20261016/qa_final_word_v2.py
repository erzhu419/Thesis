"""第二轮替换库的最终原生Word只读回归；收到最终路径后显式运行。"""
from argparse import ArgumentParser
from collections import Counter
from pathlib import Path
from zipfile import ZipFile
from xml.etree import ElementTree as E
import json
import re
import sys

BASE = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE.parent / 'exact_library_20261015'))
from qa_final_word import read_docx, text, style, equation_tables, bibliography, check_links, fields, W, M

R = '{http://schemas.openxmlformats.org/officeDocument/2006/relationships}'
P = '{http://schemas.openxmlformats.org/package/2006/relationships}'


def package_parts(path):
    with ZipFile(path) as archive:
        rels = E.fromstring(archive.read('word/_rels/document.xml.rels'))
        parts = {name: E.fromstring(archive.read(name)) for name in archive.namelist() if re.fullmatch(r'word/(?:header|footer)\d+\.xml', name)}
    return {r.get('Id'): 'word/' + r.get('Target') for r in rels.findall(P + 'Relationship')}, parts


def section_signature(section, rels, parts):
    settings = {}
    for name in ['type', 'pgSz', 'pgMar', 'pgNumType', 'cols', 'titlePg']:
        node = section.find(W + name)
        settings[name] = dict(node.attrib) if node is not None else None
    headers, footers = {}, {}
    for node in section:
        if node.tag not in (W + 'headerReference', W + 'footerReference'):
            continue
        partname = rels.get(node.get(R + 'id'))
        part = parts.get(partname)
        kind = node.get(W + 'type')
        if node.tag == W + 'headerReference':
            headers[kind] = text(part) if part is not None else None
        else:
            footers[kind] = fields(part)[0] if part is not None else None
    return {'settings': settings, 'headers': headers, 'footer_field_codes': footers}


def sequence_fields(element):
    """只取SEQ指令及其缓存结果；同时支持简单和复杂字段。"""
    result, stack = [], []
    for node in element.iter():
        if node.tag == W + 'fldSimple':
            code = node.get(W + 'instr', '')
            if re.match(r'\s*SEQ\s', code):
                result.append({'code': re.sub(r'\s+', ' ', code).strip(), 'result': text(node).strip()})
        elif node.tag == W + 'fldChar':
            kind = node.get(W + 'fldCharType')
            if kind == 'begin':
                stack.append({'code': [], 'result': [], 'phase': 'code'})
            elif kind == 'separate' and stack:
                stack[-1]['phase'] = 'result'
            elif kind == 'end' and stack:
                field = stack.pop()
                code = ''.join(field['code'])
                if re.match(r'\s*SEQ\s', code):
                    result.append({'code': re.sub(r'\s+', ' ', code).strip(), 'result': ''.join(field['result']).strip()})
        elif stack and node.tag == W + 'instrText' and stack[-1]['phase'] == 'code':
            stack[-1]['code'].append(node.text or '')
        elif stack and node.tag in (W + 't', M + 't') and stack[-1]['phase'] == 'result':
            stack[-1]['result'].append(node.text or '')
    return result


def captions(root, kind):
    wanted = 'figurecaption' if kind == '图' else 'tablecaption'
    result = []
    for block in root.find(W + 'body'):
        if style(block) == wanted:
            match = re.match(r'^' + kind + r'((?:\d+|[A-F])-\d+)\s', text(block))
            result.append({'id': match[1] if match else None, 'text': text(block)})
    return result


def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument('final_docx', type=Path)
    args = parser.parse_args()
    before = BASE.parents[1] / '重构论文_替换库修订_20261015.docx'
    mdpath = BASE.parents[1] / '重构论文_替换库二次修订_20261016.md'
    old, old_media = read_docx(before)
    new, new_media = read_docx(args.final_docx)
    ob, nb = list(old.find(W + 'body')), list(new.find(W + 'body'))
    old_rels, old_parts = package_parts(before)
    new_rels, new_parts = package_parts(args.final_docx)

    remaining = list(old_media.items())
    changed_images = []
    for name, data in new_media.items():
        index = next((i for i, (_, olddata) in enumerate(remaining) if olddata == data), None)
        if index is None:
            changed_images.append(name)
        else:
            remaining.pop(index)
    images_ok = len(old_media) == len(new_media) == 53 and not remaining and not changed_images
    old_refs, new_refs = bibliography(old), bibliography(new)
    refs_ok = len(old_refs) == len(new_refs) == 218 and old_refs == new_refs
    links = check_links(new)
    links_ok = not any(links[k] for k in ['duplicate_names', 'unpaired_start_ids', 'unpaired_end_ids', 'unterminated_complex_fields', 'broken_targets', 'citation_label_target_mismatches'])

    oe, ne = equation_tables(old), equation_tables(new)
    old_labels = [label for labels, _ in oe for label in labels]
    new_labels = [label for labels, _ in ne for label in labels]
    differences = [{'ordinal': i + 1, 'old': a, 'new': b} for i, (a, b) in enumerate(zip(old_labels, new_labels)) if a != b]
    old_seq, new_seq = sequence_fields(old), sequence_fields(new)
    seq_table_issues = []
    for labels, table in ne:
        seqs = sequence_fields(table)
        if not seqs:
            continue
        expected = [re.search(r'-(\d+)', label)[1] for label in labels]
        actual = [seq['result'] for seq in seqs]
        if expected != actual:
            seq_table_issues.append({'labels': labels, 'SEQ_results': actual})
    seq_ok = len(old_seq) == len(new_seq) == 234 and old_seq == new_seq and not seq_table_issues
    equations_ok = len(oe) == len(ne) == 368 and len(old_labels) == len(new_labels) == 376 and old_labels == new_labels

    old_f = next(i for i, block in enumerate(ob) if style(block) == 'chapter' and text(block).startswith('附录F '))
    old_g = next(i for i, block in enumerate(ob) if style(block) == 'chapter' and text(block).startswith('附录G '))
    original_f_section = [block.find(W + 'pPr/' + W + 'sectPr') for block in ob[old_f:old_g] if block.find(W + 'pPr/' + W + 'sectPr') is not None][-1]
    final_section = nb[-1]
    old_f_signature = section_signature(original_f_section, old_rels, old_parts)
    final_signature = section_signature(final_section, new_rels, new_parts)
    old_section_count = len(list(old.iter(W + 'sectPr')))
    new_section_count = len(list(new.iter(W + 'sectPr')))
    g_main = bool(re.search(r'附录\s*G', text(new)))
    g_parts = [name for name, part in new_parts.items() if re.search(r'附录\s*G', text(part))]
    f_chapters = [text(block) for block in nb if style(block) == 'chapter' and re.match(r'附录\s*F', text(block))]
    f_toc = [text(block) for block in nb if style(block).lower().startswith('toc') and re.match(r'附录\s*F', text(block))]
    sections_ok = final_section.tag == W + 'sectPr' and old_f_signature == final_signature and new_section_count == old_section_count - 1 and not g_main and not g_parts and len(f_chapters) == len(f_toc) == 1

    figs, tables = captions(new, '图'), captions(new, '表')
    old_fig_ids = {x['id'] for x in captions(old, '图')}
    old_table_ids = {x['id'] for x in captions(old, '表')}
    fake_f1 = [x for x in figs if x['text'].startswith('图F-1 比较')]
    captions_ok = len(figs) == 53 and len(tables) == 46 and len({x['id'] for x in figs}) == 53 and len({x['id'] for x in tables}) == 46 and {x['id'] for x in figs} == old_fig_ids and {x['id'] for x in tables} == old_table_ids and not fake_f1 and all(x['id'] for x in figs + tables)

    md = mdpath.read_text()
    expected_f2 = re.search(r'^\*\*(F\.2\s+.*?)\*\*', md, re.M)[1]
    f2_definitions = [text(block) for block in nb if re.match(r'^F\.2\s', text(block))]
    body_text = ''.join(text(block) for block in nb if not style(block).lower().startswith('toc'))
    word_f2_refs = len(re.findall(r'附录\s*F\.\s*2', body_text))
    md_f2_refs = md.count('附录F.2')
    f2_ok = len(f2_definitions) == 1 and re.sub(r'\s+', '', f2_definitions[0]).startswith(re.sub(r'\s+', '', expected_f2)) and word_f2_refs == md_f2_refs == 3
    report = {'baseline': str(before), 'final': str(args.final_docx), 'images': {'old_count': len(old_media), 'new_count': len(new_media), 'all_53_bytes_identical': images_ok, 'changed': changed_images, 'missing': [x[0] for x in remaining]}, 'bibliography': {'old_count': len(old_refs), 'new_count': len(new_refs), 'all_218_text_and_targets_identical': refs_ok}, 'links': links, 'equations': {'old_tables': len(oe), 'new_tables': len(ne), 'old_labels': len(old_labels), 'new_labels': len(new_labels), 'label_differences': differences, 'all_labels_and_order_unchanged': equations_ok, 'old_SEQ_count': len(old_seq), 'new_SEQ_count': len(new_seq), 'SEQ_codes_and_results_unchanged': old_seq == new_seq, 'SEQ_label_mismatches': seq_table_issues, 'all_SEQ_valid': seq_ok}, 'appendix_G_removal_and_F_section': {'old_section_count': old_section_count, 'new_section_count': new_section_count, 'G_in_body_or_TOC': g_main, 'G_in_headers_or_footers': g_parts, 'F_chapter_count': len(f_chapters), 'F_TOC_count': len(f_toc), 'final_F_section_signature_matches_original': old_f_signature == final_signature, 'valid': sections_ok}, 'captions': {'formal_figures': len(figs), 'formal_tables': len(tables), 'duplicate_figure_ids': [k for k, n in Counter(x['id'] for x in figs).items() if n > 1], 'duplicate_table_ids': [k for k, n in Counter(x['id'] for x in tables).items() if n > 1], 'fake_F1_captions': fake_f1, 'all_53_figures_46_tables_correspond': captions_ok}, 'appendix_F2': {'definition_count': len(f2_definitions), 'word_reference_count': word_f2_refs, 'md_reference_count': md_f2_refs, 'valid': f2_ok}}
    report['pass'] = all([images_ok, refs_ok, links_ok, equations_ok, seq_ok, sections_ok, captions_ok, f2_ok])
    (BASE / 'qa_final_word.json').write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps(report, ensure_ascii=False))


if __name__ == '__main__':
    main()
