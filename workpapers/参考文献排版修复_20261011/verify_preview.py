"""Check the scope of the repair and render every bibliography page."""

from collections import Counter
from pathlib import Path
from zipfile import ZipFile
import json
import re

import fitz
from lxml import etree as E
from PIL import Image, ImageDraw

from repair import SOURCE, OUT, WORK, W, canonical_text, body_spans


def structure(element):
    return (element.tag, tuple(sorted(element.attrib.items())), element.text,
            tuple(structure(c) for c in element))


def reference_pages(doc):
    texts = [p.get_text().replace('\u200b', '') for p in doc]
    first = next(i for i, t in enumerate(texts) if re.search(r'\[1\]\s+DAGANZO', t))
    last = next(i for i in range(first, len(texts)) if re.search(r'\[214\]', texts[i]))
    return list(range(first, last + 1))


def line_counts(doc, pages):
    counts = Counter()
    current = None
    for index in pages:
        rows = {}
        for block in doc[index].get_text('dict')['blocks']:
            for line in block.get('lines', []):
                if not 70 < line['bbox'][1] < 765:
                    continue
                key = round(line['bbox'][1])
                rows.setdefault(key, []).append((line['bbox'][0], ''.join(s['text'] for s in line['spans'])))
        for key, items in sorted(rows.items()):
            value = ''.join(s for _, s in sorted(items)).replace('\u200b', '').strip()
            match = re.match(r'^\[(\d+)\]', value)
            if match:
                current = int(match[1])
            if current and value:
                counts[current] += 1
    assert len(counts) == 214, len(counts)
    return counts


def main():
    with ZipFile(SOURCE) as a, ZipFile(OUT) as b:
        assert a.namelist() == b.namelist()
        for name in a.namelist():
            if name != 'word/document.xml':
                assert a.read(name) == b.read(name), name
        raw_a, raw_b = a.read('word/document.xml'), b.read('word/document.xml')
        body_a = E.fromstring(raw_a).find(W + 'body')
        body_b = E.fromstring(raw_b).find(W + 'body')
        spans_a, spans_b = body_spans(raw_a), body_spans(raw_b)
        changed_toc = 0
        for i, (pa, pb) in enumerate(zip(body_a, body_b)):
            if 1787 <= i <= 2000:
                assert canonical_text(pa) == canonical_text(pb), i
                assert [structure(r) for r in pa.iter(W + 'rPr')] == [structure(r) for r in pb.iter(W + 'rPr')]
                for tag in ['hyperlink', 'bookmarkStart', 'bookmarkEnd', 'instrText']:
                    assert [e.attrib for e in pa.iter(W + tag)] == [e.attrib for e in pb.iter(W + tag)], (i, tag)
                continue
            sa, ea = spans_a[i]
            sb, eb = spans_b[i]
            if raw_a[sa:ea] == raw_b[sb:eb]:
                continue
            style = pa.find(W + 'pPr/' + W + 'pStyle')
            assert style is not None and style.get(W + 'val') in ('TOC1', 'TOC2', 'TOC3'), i
            ta, tb = list(pa.iter(W + 't')), list(pb.iter(W + 't'))
            assert ta[-1].text.isdigit() and tb[-1].text.isdigit()
            old = tb[-1].text
            tb[-1].text = ta[-1].text
            assert E.tostring(pa) == E.tostring(pb), i
            tb[-1].text = old
            changed_toc += 1
    original = fitz.open(SOURCE.with_suffix('.pdf'))
    final = fitz.open(OUT.with_suffix('.pdf'))
    old_pages, pages = reference_pages(original), reference_pages(final)
    old_urls = {link['uri'] for i in old_pages for link in original[i].get_links() if 'uri' in link}
    new_urls = {link['uri'] for i in pages for link in final[i].get_links() if 'uri' in link}
    assert old_urls == new_urls, (old_urls - new_urls, new_urls - old_urls)
    before, after = line_counts(original, old_pages), line_counts(final, pages)
    for start in range(0, len(pages), 4):
        sheet = Image.new('RGB', (1440, 2070), '#dddddd')
        draw = ImageDraw.Draw(sheet)
        for slot, index in enumerate(pages[start:start + 4]):
            pix = final[index].get_pixmap(matrix=fitz.Matrix(1.17, 1.17), alpha=False)
            preview = Image.frombytes('RGB', (pix.width, pix.height), pix.samples)
            x, y = (slot % 2) * 720, (slot // 2) * 1035
            draw.text((x + 8, y + 5), f'PDF {index+1}', fill='black')
            sheet.paste(preview, (x + 10, y + 25))
        sheet.save(WORK / f'参考文献逐页_{start//4+1}.png')
    for i in [pages[0], pages[3], pages[8], pages[-1]]:
        final[i].get_pixmap(matrix=fitz.Matrix(1.5, 1.5)).save(str(WORK / f'修复后_{i+1}.png'))
    result = {'all_non_document_package_parts_unchanged': True,
              'non_bibliography_blocks_changed_only_for_toc_page_cache': changed_toc,
              'references_preserved_in_order': 214, 'original_pdf_pages': len(original),
              'bibliography_pdf_link_targets_preserved': len(new_urls),
              'final_pdf_pages': len(final), 'bibliography_pages_before': len(old_pages),
              'bibliography_pages_after': len(pages),
              'all_bibliography_pages_rendered': [i + 1 for i in pages],
              'reference_line_distribution_before': dict(sorted(Counter(before.values()).items())),
              'reference_line_distribution_after': dict(sorted(Counter(after.values()).items())),
              'references_shorter': sum(after[n] < before[n] for n in before),
              'references_longer': sum(after[n] > before[n] for n in before)}
    (WORK / '成品验收.json').write_text(json.dumps(result, ensure_ascii=False, indent=2))
    print(json.dumps(result, ensure_ascii=False))


if __name__ == '__main__':
    main()
