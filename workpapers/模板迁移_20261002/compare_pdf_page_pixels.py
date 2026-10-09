#!/usr/bin/env python3
"""Compare actual RGB page pixels at the same scale as the contact-sheet renderer."""
import argparse
import json
from pathlib import Path
import subprocess
import sys

import fitz

HERE = Path(__file__).resolve().parent


def render_pixels(page):
    scale = min(96/72, 800/page.rect.width, 1131/page.rect.height)
    pix = page.get_pixmap(matrix=fitz.Matrix(scale, scale), alpha=False, colorspace=fitz.csRGB)
    return (pix.width, pix.height), pix.samples


def compare_chunk(old_path, new_path, start, stop):
    records = []
    with fitz.open(old_path) as old, fitz.open(new_path) as new:
        for index in range(start, stop):
            if index >= len(old) or index >= len(new):
                records.append({'physical_page': index+1, 'identical': False,
                                'reason': 'Page is present in only one PDF.'})
                continue
            old_size, old_pixels = render_pixels(old[index])
            new_size, new_pixels = render_pixels(new[index])
            records.append({'physical_page': index+1,
                            'identical': old_size == new_size and old_pixels == new_pixels,
                            'old_pixels': list(old_size), 'new_pixels': list(new_size)})
    return records


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--old', type=Path, default=HERE/'final_preview_word.pdf')
    parser.add_argument('--new', type=Path, default=HERE/'final_preview_symbol_fixed.pdf')
    parser.add_argument('--report', type=Path, default=HERE/'page_visual_diff.json')
    parser.add_argument('--chunk', nargs=2, type=int)
    args = parser.parse_args()
    if args.chunk:
        print(json.dumps(compare_chunk(args.old, args.new, *args.chunk)))
        return
    with fitz.open(args.old) as old, fitz.open(args.new) as new:
        old_count, new_count = len(old), len(new)
    records = []
    for start in range(0, max(old_count, new_count), 16):
        stop = min(start+16, max(old_count, new_count))
        raw = subprocess.check_output([sys.executable, __file__, '--old', str(args.old),
                  '--new', str(args.new), '--chunk', str(start), str(stop)], text=True)
        records.extend(json.loads(raw))
        print(f'Compared actual pixels on physical pages {start+1}-{stop}', flush=True)
    report = {'old_pdf': str(args.old.resolve()), 'new_pdf': str(args.new.resolve()),
              'old_page_count': old_count, 'new_page_count': new_count,
              'method': 'Direct RGB pixel byte comparison, no hashes; 96 dpi capped at 800 by 1131 pixels, matching render_full_preview.py.',
              'identical_pages': [r['physical_page'] for r in records if r['identical']],
              'changed_pages': [r['physical_page'] for r in records if not r['identical']],
              'pages': records,
              'review_rule': 'Reuse actual visual review only for identical pages; visually review every changed page.'}
    args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2)+'\n')
    print(json.dumps({'report': str(args.report), 'identical_pages': len(report['identical_pages']),
                      'changed_pages': report['changed_pages']}, ensure_ascii=False))


if __name__ == '__main__':
    main()
