"""格式恢复后的只读内容零改动比较；不重做数学语义审查。"""
from argparse import ArgumentParser
from collections import Counter
from pathlib import Path
import json
import sys

BASE = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE.parent / 'exact_library_20261015'))
from qa_final_word import read_docx, text, style, equation_tables, bibliography, check_links, W, M


def paragraph_texts(root):
    result = []
    for paragraph in root.find(W + 'body').iter(W + 'p'):
        if style(paragraph).lower().startswith('toc'):
            continue
        normal = ''.join(n.text or '' for n in paragraph.iter(W + 't'))
        math = ''.join(n.text or '' for n in paragraph.iter(M + 't'))
        instructions = ''.join(n.text or '' for n in paragraph.iter(W + 'instrText'))
        result.append((normal, math, text(paragraph), instructions))
    return result


def citation_links(root):
    return [(h.get(W + 'anchor'), text(h)) for h in root.find(W + 'body').iter(W + 'hyperlink') if h.get(W + 'anchor', '').startswith('latest_ref_')]


def bookmark_names(root):
    return Counter(n.get(W + 'name') for n in root.iter(W + 'bookmarkStart') if n.get(W + 'name') != '_GoBack' and not n.get(W + 'name', '').startswith('_Toc'))


def math_styles(root):
    return [n.get(M + 'val') for n in root.find(W + 'body').iter(M + 'sty')]


def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument('final_docx', type=Path)
    args = parser.parse_args()
    old, old_media = read_docx(BASE / '修改前.docx')
    new, new_media = read_docx(args.final_docx)
    old_text, new_text = paragraph_texts(old), paragraph_texts(new)
    text_changes = []
    for index, (before, after) in enumerate(zip(old_text, new_text), 1):
        if before != after:
            text_changes.append({'non_TOC_paragraph': index, 'old': before[2][:160], 'new': after[2][:160], 'normal_text_equal': before[0] == after[0], 'math_text_equal': before[1] == after[1], 'instruction_text_equal': before[3] == after[3]})
    old_equations, new_equations = equation_tables(old), equation_tables(new)
    old_labels = [label for labels, _ in old_equations for label in labels]
    new_labels = [label for labels, _ in new_equations for label in labels]
    old_refs, new_refs = bibliography(old), bibliography(new)
    old_bookmarks, new_bookmarks = bookmark_names(old), bookmark_names(new)
    links = check_links(new)
    checks = {
        'all_non_TOC_paragraph_text_and_field_instructions_identical': old_text == new_text,
        'all_53_image_bytes_identical': len(old_media) == len(new_media) == 53 and sorted(old_media.values()) == sorted(new_media.values()),
        'all_376_formula_labels_and_order_identical': len(old_labels) == len(new_labels) == 376 and old_labels == new_labels,
        'native_equation_table_count_unchanged': len(old_equations) == len(new_equations),
        'all_218_bibliography_text_and_targets_identical': len(old_refs) == len(new_refs) == 218 and old_refs == new_refs,
        'citation_links_and_labels_identical': citation_links(old) == citation_links(new),
        'body_bookmark_names_preserved_except_regenerated_TOC_and_cursor': old_bookmarks == new_bookmarks,
        'all_internal_targets_valid': not any(links[k] for k in ['duplicate_names', 'unpaired_start_ids', 'unpaired_end_ids', 'unterminated_complex_fields', 'broken_targets', 'citation_label_target_mismatches']),
        'native_math_styles_including_b_and_bi_identical': math_styles(old) == math_styles(new),
    }
    report = {'baseline': str(BASE / '修改前.docx'), 'final': str(args.final_docx), 'scope': '仅比较非TOC正文文字/数学字符/字段指令及资源引用；允许目录页码刷新，不重审数学语义。', 'checks': checks, 'counts': {'old_non_TOC_paragraphs': len(old_text), 'new_non_TOC_paragraphs': len(new_text), 'equation_tables': len(new_equations), 'equation_labels': len(new_labels), 'images': len(new_media), 'bibliography': len(new_refs), 'citation_links': len(citation_links(new)), 'bookmarks': links['bookmark_count'], 'REF': links['REF_count'], 'PAGEREF': links['PAGEREF_count']}, 'text_differences': text_changes, 'lost_bookmarks': list((old_bookmarks - new_bookmarks).elements()), 'added_bookmarks': list((new_bookmarks - old_bookmarks).elements()), 'broken_targets': links['broken_targets'], 'pass': all(checks.values())}
    (BASE / 'qa_format_content.json').write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps(report, ensure_ascii=False))


if __name__ == '__main__':
    main()
