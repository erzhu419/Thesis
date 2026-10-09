from pathlib import Path
from zipfile import ZipFile
import json
import shutil
from PIL import Image

root = Path(__file__).resolve().parents[2]
work = Path(__file__).resolve().parent
assets = root / 'output/imagegen/论文五图_20261005'
source = root / '排版完成_最终核对_20261005.docx'
target = root / '排版完成_最终核对_五图重绘_20261005.docx'
mapping = {
    '2-1': ('word/media/image2.png', 'figure_2-1_corrected.png'),
    '2-3': ('word/media/image4.png', 'figure_2-3_final.png'),
    '4-1': ('word/media/image21.png', 'figure_4-1_corrected.png'),
    '5-2': ('word/media/image32.png', 'figure_5-2_sources.png'),
    '6-1': ('word/media/image44.png', 'figure_6-1_weight.png'),
}
assert source.read_bytes() == (work / '修改前.docx').read_bytes(), 'Author source changed; review latest source before replacing images.'
replacements = {part: (assets / name).read_bytes() for part, name in mapping.values()}
with ZipFile(source) as original, ZipFile(target, 'w') as revised:
    for entry in original.infolist():
        revised.writestr(entry, replacements.get(entry.filename, original.read(entry.filename)))

# Detect any unintended change to text, equations, styles, sections or other assets.
with ZipFile(source) as original, ZipFile(target) as revised:
    assert original.namelist() == revised.namelist()
    changed = [name for name in original.namelist() if original.read(name) != revised.read(name)]
    assert sorted(changed) == sorted(replacements), changed
    assert revised.testzip() is None

final_assets = work / '交付图片'
final_assets.mkdir(exist_ok=True)
records = []
for figure, (part, name) in mapping.items():
    dest = final_assets / f'图{figure}.png'
    shutil.copy2(assets / name, dest)
    with Image.open(dest) as im:
        records.append({'figure': figure, 'image': str(dest), 'pixels': list(im.size), 'word_part': part})
record = {
    'source': str(source), 'word': str(target),
    'generation': {'endpoint': 'https://ruoli.dev/v1', 'model': 'gpt-image-2', 'mode': 'imagegen bundled CLI API generation and editing'},
    'changed_parts': changed,
    'all_other_word_parts_unchanged': True,
    'figures': records,
}
(work / '交付记录.json').write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding='utf-8')
print(json.dumps(record, ensure_ascii=False, indent=2))
