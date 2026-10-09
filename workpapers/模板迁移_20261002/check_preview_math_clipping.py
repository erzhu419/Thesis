#!/usr/bin/env python3
"""Locate possible horizontal glyph clipping near Word equation-cell PDF clips."""
import argparse
from collections import Counter
import json
from pathlib import Path
import re
import subprocess
import sys
from xml.etree import ElementTree as ET
from zipfile import ZipFile

import fitz

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
NS = {'w': 'http://schemas.openxmlformats.org/wordprocessingml/2006/main'}
W = '{' + NS['w'] + '}'


def equation_geometry(docx):
    with ZipFile(docx) as archive:
        root = ET.fromstring(archive.read('word/document.xml'))
    margins = Counter((int(e.get(W+'left')), int(e.get(W+'right')))
                      for e in root.findall('.//w:sectPr/w:pgMar', NS))
    left_margin, right_margin = margins.most_common(1)[0][0]
    grids = Counter()
    for table in root.findall('.//w:tbl', NS):
        code = ''.join(e.text or '' for e in table.findall('.//w:instrText', NS))
        if 'SEQ Eq' in code:
            grid = tuple(int(e.get(W+'w')) for e in table.findall('./w:tblGrid/w:gridCol', NS))
            if len(grid) != 3:
                raise ValueError(f'Equation layout requires the observed three columns: {grid}')
            grids[grid] += 1
    return {'margin_twips': [left_margin, right_margin],
            'equation_grids': [{'widths_twips': list(g), 'count': n} for g,n in grids.items()]}


def inspect_pages(path, start, stop, geometry):
    findings = []
    with fitz.open(path) as pdf:
        for index in range(start, min(stop, len(pdf))):
            page = pdf[index]
            boundaries = [(geometry['margin_twips'][0]/20 + g['widths_twips'][0]/20,
                           page.rect.width - geometry['margin_twips'][1]/20 - g['widths_twips'][2]/20)
                          for g in geometry['equation_grids']]
            cells = {tuple(d['scissor']) for d in page.get_drawings(extended=True)
                     if d['type'] == 'clip' and any(abs(d['scissor'].x0-left) < .2
                     and abs(d['scissor'].x1-right) < .2 for left,right in boundaries)}
            if not cells:
                continue
            spans = [s for s in page.get_texttrace() if 'Math' in s['font']]
            for bounds in sorted(cells):
                cell = fitz.Rect(bounds)
                clipped = []
                for span in spans:
                    for code, _, origin, box in span['chars']:
                        # Baselines from the preceding paragraph can enter the clip by <1 pt.
                        # Require the glyph body, rather than its baseline, to occupy this row.
                        if (cell.y0 <= (box[1]+box[3])/2 <= cell.y1
                                and (box[0] < cell.x0-.5 or box[2] > cell.x1+.5)
                                and not chr(code).isspace()):
                            clipped.append(chr(code))
                if clipped:
                    number_box = fitz.Rect(cell.x1, cell.y0,
                                           page.rect.width-geometry['margin_twips'][1]/20, cell.y1)
                    numbers = re.findall(r'\([0-9A-Z]+-\d+\)', page.get_textbox(number_box))
                    findings.append({'physical_page': index+1,
                                     'equation': numbers[-1] if numbers else 'unknown',
                                     'cell': [round(v, 2) for v in bounds],
                                     'clipped': ''.join(clipped)})
    return findings


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('pdf', nargs='?', type=Path, default=ROOT/'output/排版预览.pdf')
    parser.add_argument('--docx', type=Path, default=ROOT/'output/排版完成.docx')
    parser.add_argument('--report', type=Path, default=HERE/'visual/math_clipping.json')
    parser.add_argument('--chunk', nargs=2, type=int)
    args = parser.parse_args()
    geometry = equation_geometry(args.docx)
    if args.chunk:
        print(json.dumps(inspect_pages(args.pdf, *args.chunk, geometry), ensure_ascii=False))
        return
    with fitz.open(args.pdf) as pdf:
        pages = len(pdf)
    findings = []
    # The installed texttrace build crashes after many pages in one process.
    for start in range(0, pages, 16):
        result = subprocess.check_output([sys.executable, __file__, str(args.pdf), '--docx',
                    str(args.docx), '--chunk', str(start), str(min(start+16, pages))], text=True)
        findings.extend(json.loads(result))
        print(f'Checked equation clips on physical pages {start+1}-{min(start+16, pages)}', flush=True)
    report = {'pdf': str(args.pdf.resolve()), 'docx': str(args.docx.resolve()),
              'physical_pages': pages, 'geometry': geometry, 'clipping': findings,
              'scope': 'Geometry candidates near matched equation-cell clips, not a clipping verdict. Nearby prose subscripts can share the vertical band; confirm each candidate visually.'}
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2)+'\n')
    print(json.dumps({'report': str(args.report), 'clipping_findings': len(findings)}, ensure_ascii=False))
    sys.exit(bool(findings))

if __name__ == '__main__':
    main()
