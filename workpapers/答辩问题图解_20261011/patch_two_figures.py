"""Apply only the two figure comments to the delivered defense deck."""
import json
import math
import posixpath
from pathlib import Path
from shutil import copy2
from zipfile import ZipFile

from lxml import etree
from pptx import Presentation
from pptx.chart.data import XyChartData
from pptx.dml.color import RGBColor
from pptx.enum.chart import XL_MARKER_STYLE
from pptx.enum.dml import MSO_LINE_DASH_STYLE
from pptx.util import Inches, Pt

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
SOURCE = ROOT / 'output/博士毕业答辩_问题图解_20261011.pptx'
OUTPUT = ROOT / 'output/博士毕业答辩_问题图解_两图修订_20261011.pptx'
BACKUP = HERE / 'before_two_figure_comments'


def add_components(slide):
    mixture = next(s for s in slide.shapes if s.shape_id == 12)
    colors = [str(next(s for s in slide.shapes if s.shape_id == sid).line.color.rgb) for sid in [17, 20, 23]]
    # Each dashed curve contributes one third of its mode density to the sum.
    for mean, color in zip([-1.4, 0, 1.4], colors):
        points = []
        for i in range(121):
            z = -3 + 6 * i / 120
            density = math.exp(-.5 * ((z - mean) / .64) ** 2) / (.64 * math.sqrt(2 * math.pi)) / 3
            points.append((1.03 + 4.44 * i / 120, 3.84 - 1.19 * density / .36))
        builder = slide.shapes.build_freeform(Inches(points[0][0]), Inches(points[0][1]))
        builder.add_line_segments([(Inches(x), Inches(y)) for x, y in points[1:]], close=False)
        curve = builder.convert_to_shape()
        curve.name = f'Mode contribution {mean:g}'
        curve.fill.background()
        curve.line.color.rgb = RGBColor.from_string(color)
        curve.line.width = Pt(2.1)
        curve.line.dash_style = MSO_LINE_DASH_STYLE.DASH
        for effect in curve._element.xpath('.//a:effectRef'):
            effect.set('idx', '0')
        mixture._element.addprevious(curve._element)


def sparse_red_samples(chart):
    data = XyChartData()
    samples = data.add_series('示例样本')
    for x in [0, .1, .2, .3, .4, .8]:
        samples.add_data_point(x, 5 * x)
    for series in list(chart.series)[1:]:
        target = data.add_series(series.name)
        xs = [float(v.text) for v in series._element.xpath('./c:xVal/c:numRef/c:numCache/c:pt/c:v')]
        for x, y in zip(xs, series.values):
            target.add_data_point(x, y)
    chart.replace_data(data)
    samples = chart.series[0]
    samples.format.line.fill.background()
    samples.marker.style = XL_MARKER_STYLE.CIRCLE
    samples.marker.size = 11
    samples.marker.format.fill.solid()
    samples.marker.format.fill.fore_color.rgb = RGBColor.from_string('D72D3B')
    samples.marker.format.line.color.rgb = RGBColor.from_string('FFFFFF')
    samples.marker.format.line.width = Pt(.7)


def response_data(chart):
    return [
        (s.name, [v.text for v in s._element.xpath('./c:xVal/c:numRef/c:numCache/c:pt/c:v')], list(s.values))
        for s in list(chart.series)[1:]
    ]


def main():
    BACKUP.mkdir(exist_ok=True)
    backup = BACKUP / SOURCE.name
    if not backup.exists():
        copy2(SOURCE, backup)
    before = Presentation(SOURCE)
    after = Presentation(SOURCE)
    add_components(after.slides[13])
    chart = next(s.chart for s in after.slides[18].shapes if s.has_chart)
    sparse_red_samples(chart)
    notes = after.slides[18].notes_slide.notes_text_frame
    notes.text = notes.text.replace('灰点', '红色样本点')
    after.save(OUTPUT)

    # Retain original part names and relationships instead of repackaging the
    # reordered deck; only the drawing, sample chart and its workbook change.
    old_chart = next(s.chart for s in before.slides[18].shapes if s.has_chart)
    parts = [
        (old_chart.part, chart.part),
        (old_chart.part.chart_workbook.xlsx_part, chart.part.chart_workbook.xlsx_part),
        (before.slides[18].notes_slide.part, after.slides[18].notes_slide.part),
    ]
    with ZipFile(SOURCE) as original:
        presentation = etree.fromstring(original.read('ppt/presentation.xml'))
        relationships = etree.fromstring(original.read('ppt/_rels/presentation.xml.rels'))
        targets = {rel.get('Id'): rel.get('Target') for rel in relationships}
        slide_ids = presentation.find('{http://schemas.openxmlformats.org/presentationml/2006/main}sldIdLst')
        relation_id = slide_ids[13].get('{http://schemas.openxmlformats.org/officeDocument/2006/relationships}id')
        slide_part = posixpath.normpath('ppt/' + targets[relation_id])
    with ZipFile(OUTPUT) as edited:
        replacements = {str(old.partname).lstrip('/'): edited.read(str(new.partname).lstrip('/')) for old, new in parts}
        replacements[slide_part] = edited.read(str(after.slides[13].part.partname).lstrip('/'))
    with ZipFile(SOURCE) as source, ZipFile(OUTPUT, 'w') as target:
        for entry in source.infolist():
            target.writestr(entry, replacements.get(entry.filename, source.read(entry.filename)))

    # Detect unintended edits outside the requested slide and chart objects.
    reopened = Presentation(OUTPUT)
    unchanged_slides = all(
        etree.tostring(a._element) == etree.tostring(b._element)
        for i, (a, b) in enumerate(zip(before.slides, reopened.slides)) if i != 13
    )
    new_chart = next(s.chart for s in reopened.slides[18].shapes if s.has_chart)
    assert unchanged_slides and len(reopened.slides) == len(before.slides) == 55
    assert response_data(old_chart) == response_data(new_chart)
    assert len(new_chart.series[0].values) == 6 and new_chart.series[0].marker.size == 11
    assert SOURCE.read_bytes() == backup.read_bytes()
    with ZipFile(SOURCE) as old, ZipFile(OUTPUT) as new:
        changed_parts = [name for name in new.namelist() if name not in old.namelist() or new.read(name) != old.read(name)]
    assert set(changed_parts) == set(replacements)
    result = {
        'slides': 55, 'page14_added_dashed_components': 3,
        'page19_red_samples': {'left': 5, 'isolated': 1, 'marker_size_pt': 11},
        'other_slide_objects_unchanged': unchanged_slides,
        'two_response_curves_unchanged': True, 'source_preserved': True,
        'page19_speaker_note_color_synced': '灰点' not in reopened.slides[18].notes_slide.notes_text_frame.text,
        'changed_package_parts': changed_parts,
    }
    (HERE / 'two_figure_verification.json').write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
