"""Inspect revised equations and newly integrated result pages at readable scale."""
from pathlib import Path
import fitz

HERE = Path(__file__).resolve().parent
doc = fitz.open(HERE.parents[1] / 'output/排版完成_20261003_全文深度校审_终校.pdf')
terms = ['(2-16)', '(3-21)', '(4-7)', '(4-15)', '(5-5)', '(D-59)', '(B-29)',
         'E.9 归一化对数置信备份的尺度性质', '表6-17']
selected = set()
for term in terms:
    pages = [p.number for p in doc if term in p.get_text()]
    if pages:
        selected.add(pages[-1])
for number in sorted(selected):
    doc[number].get_pixmap(matrix=fitz.Matrix(1.35, 1.35), alpha=False).save(
        HERE / 'visual' / f'focus_{number+1:03d}.png')
print('Focused pages:', [p+1 for p in sorted(selected)])
