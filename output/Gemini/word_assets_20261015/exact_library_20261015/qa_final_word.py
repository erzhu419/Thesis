"""独立只读回归；仅在收到原生Word最终版路径后显式运行。"""
from argparse import ArgumentParser
from collections import Counter
from pathlib import Path
from zipfile import ZipFile
from xml.etree import ElementTree as E
import json
import re

BASE = Path(__file__).resolve().parent
W = '{http://schemas.openxmlformats.org/wordprocessingml/2006/main}'
M = '{http://schemas.openxmlformats.org/officeDocument/2006/math}'
LABEL = re.compile(r'[（(]((?:\d+|[A-Z])-\d+[a-z]?)[）)]')


def text(element):
    return ''.join(n.text or '' for n in element.iter() if n.tag in (W + 't', M + 't'))


def read_docx(path):
    with ZipFile(path) as archive:
        root = E.fromstring(archive.read('word/document.xml'))
        media = {name: archive.read(name) for name in archive.namelist() if name.startswith('word/media/') and not name.endswith('/')}
    return root, media


def style(element):
    node = element.find(W + 'pPr/' + W + 'pStyle')
    return node.get(W + 'val', '') if node is not None else ''


def equation_tables(root):
    records = []
    for table in root.iter(W + 'tbl'):
        labels = []
        for row in table.findall(W + 'tr'):
            cells = row.findall(W + 'tc')
            if len(cells) < 2:
                continue
            raw = re.sub(r'\s+', '', text(cells[-1]))
            matches = LABEL.findall(raw)
            if matches and not LABEL.sub('', raw):
                labels.extend(matches)
        if labels:
            records.append((labels, table))
    return records


def appendix_content(root):
    blocks = list(root.find(W + 'body'))
    boundaries = []
    for index, block in enumerate(blocks):
        match = re.match(r'^附录([A-G])\s', text(block))
        if match and style(block) == 'chapter':
            boundaries.append((match[1], index))
    result = {}
    for n, (letter, start) in enumerate(boundaries):
        end = boundaries[n + 1][1] if n + 1 < len(boundaries) else len(blocks)
        result[letter] = [(block.tag.rsplit('}', 1)[-1], [text(p) for p in block.iter(W + 'p')]) for block in blocks[start:end]]
    return result


def bibliography(root):
    records = []
    for block in root.find(W + 'body'):
        targets = [b.get(W + 'name') for b in block.iter(W + 'bookmarkStart') if re.fullmatch(r'latest_ref_\d+', b.get(W + 'name', ''))]
        if targets:
            records.append({'targets': targets, 'text': text(block)})
    return records


def fields(root):
    found, stack = [], []
    for node in root.iter():
        if node.tag == W + 'fldSimple':
            found.append(node.get(W + 'instr', ''))
        elif node.tag == W + 'fldChar':
            kind = node.get(W + 'fldCharType')
            if kind == 'begin':
                stack.append({'parts': [], 'instruction': True})
            elif kind == 'separate' and stack:
                stack[-1]['instruction'] = False
            elif kind == 'end' and stack:
                found.append(''.join(stack.pop()['parts']))
        elif node.tag == W + 'instrText' and stack and stack[-1]['instruction']:
            stack[-1]['parts'].append(node.text or '')
    return found, len(stack)


def check_links(root):
    starts = list(root.iter(W + 'bookmarkStart'))
    names = Counter(b.get(W + 'name') for b in starts)
    start_ids = Counter(b.get(W + 'id') for b in starts)
    end_ids = Counter(b.get(W + 'id') for b in root.iter(W + 'bookmarkEnd'))
    codes, open_fields = fields(root)
    broken_links, bad_labels, ref_targets = [], [], []
    for node in root.iter(W + 'hyperlink'):
        anchor = node.get(W + 'anchor')
        if anchor and anchor not in names:
            broken_links.append({'kind': 'hyperlink', 'target': anchor, 'text': text(node)[:90]})
        match = re.fullmatch(r'latest_ref_(\d+)', anchor or '')
        if match and text(node).strip().strip('[]') != match[1]:
            bad_labels.append({'target': anchor, 'label': text(node)})
    for code in codes:
        match = re.search(r'\b(REF|PAGEREF)\s+(?:"([^"]+)"|([^\s\\]+))', code)
        if match:
            target = match[2] or match[3]
            ref_targets.append({'kind': match[1], 'target': target})
            if target not in names:
                broken_links.append({'kind': match[1], 'target': target})
    return {'bookmark_count': len(starts), 'duplicate_names': [k for k, n in names.items() if n > 1], 'unpaired_start_ids': list((start_ids - end_ids).elements()), 'unpaired_end_ids': list((end_ids - start_ids).elements()), 'unterminated_complex_fields': open_fields, 'REF_count': sum(x['kind'] == 'REF' for x in ref_targets), 'PAGEREF_count': sum(x['kind'] == 'PAGEREF' for x in ref_targets), 'broken_targets': broken_links, 'citation_label_target_mismatches': bad_labels}


def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument('final_docx', type=Path, help='已通知完成的原生Word最终版，不能传预生成工作稿')
    args = parser.parse_args()
    before = BASE / '修改前.docx'
    old, old_media = read_docx(before)
    new, new_media = read_docx(args.final_docx)

    unmatched = list(old_media.items())
    changed_media = []
    for name, data in new_media.items():
        match = next((i for i, (oldname, olddata) in enumerate(unmatched) if oldname == name and olddata == data), None)
        if match is None:
            match = next((i for i, (_, olddata) in enumerate(unmatched) if olddata == data), None)
        if match is None:
            changed_media.append(name)
        else:
            unmatched.pop(match)
    media_ok = len(old_media) == len(new_media) == 53 and not changed_media and not unmatched

    old_appendices, new_appendices = appendix_content(old), appendix_content(new)
    appendix_checks = {letter: letter in old_appendices and letter in new_appendices and old_appendices[letter] == new_appendices[letter] for letter in 'ABCDEF'}
    old_refs, new_refs = bibliography(old), bibliography(new)
    refs_ok = len(old_refs) == len(new_refs) == 218 and old_refs == new_refs
    old_equations, new_equations = equation_tables(old), equation_tables(new)
    old_labels = [label for labels, _ in old_equations for label in labels]
    new_labels = [label for labels, _ in new_equations for label in labels]
    md_tags = re.findall(r'\\tag\{([^}]+)\}', (BASE.parents[1] / '重构论文_替换库修订_20261015.md').read_text())
    md_tag_counts = Counter(new_labels)
    tag_issues = {tag: md_tag_counts[tag] for tag in md_tags if md_tag_counts[tag] != 1}
    expected_old_227 = next(table for labels, table in old_equations if '2-27' in labels)
    final_227 = next(table for labels, table in new_equations if '2-27' in labels)
    old_math = ''.join(n.text or '' for n in expected_old_227.iter(M + 't'))
    new_math = ''.join(n.text or '' for n in final_227.iter(M + 't'))
    theta_ok = new_math.count('θ') == 2 and not any(phi in new_math for phi in ('φ', 'ϕ'))
    only_phi_changed = re.sub(r'\s+', '', new_math) == re.sub(r'\s+', '', old_math.replace('φ', 'θ').replace('ϕ', 'θ'))
    links = check_links(new)
    links_ok = not any(links[k] for k in ['duplicate_names', 'unpaired_start_ids', 'unpaired_end_ids', 'unterminated_complex_fields', 'broken_targets', 'citation_label_target_mismatches'])

    report = {'baseline': str(before), 'final': str(args.final_docx), 'images': {'old_count': len(old_media), 'new_count': len(new_media), 'all_53_bytes_identical': media_ok, 'changed_or_added': changed_media, 'missing': [name for name, _ in unmatched]}, 'appendix_A_F_content_unchanged': appendix_checks, 'bibliography': {'old_count': len(old_refs), 'new_count': len(new_refs), 'all_218_text_and_targets_identical': refs_ok}, 'links': links, 'equations': {'old_table_count': len(old_equations), 'new_table_count': len(new_equations), 'old_label_count': len(old_labels), 'new_label_count': len(new_labels), 'all_labels_and_order_identical': old_labels == new_labels, 'md_tag_count': len(md_tags), 'all_142_md_tags_once_in_word': len(md_tags) == 142 and not tag_issues, 'md_tag_issues': tag_issues, 'formula_2_27_theta_count': new_math.count('θ'), 'formula_2_27_two_thetas_no_phi': theta_ok, 'formula_2_27_only_phi_to_theta': only_phi_changed}}
    report['pass'] = all([media_ok, all(appendix_checks.values()), refs_ok, links_ok, old_labels == new_labels, len(old_equations) == len(new_equations) == 368, len(new_labels) == 376, len(md_tags) == 142, not tag_issues, theta_ok, only_phi_changed])
    (BASE / 'qa_final_word.json').write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps(report, ensure_ascii=False))


if __name__ == '__main__':
    main()
