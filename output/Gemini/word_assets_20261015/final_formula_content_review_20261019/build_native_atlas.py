"""Pack original-position Word UI crops at native scale; never touch captures."""
from pathlib import Path
from functools import lru_cache
import json
from PIL import Image, ImageDraw, ImageFont

A = Path(__file__).resolve().parent
OUT = A / 'native_atlas'
OUT.mkdir(exist_ok=True)
EXPECTED = 4825
WIDTH, MAX_HEIGHT = 1600, 2000
MARGIN, GAP, LABEL_H = 12, 12, 22
PAD_X, PAD_Y = 10, 48
FONT = ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf', 15)


@lru_cache(maxsize=6)
def viewport(number):
    path = A / 'native_viewports' / f'viewport_{number:04d}.png'
    with Image.open(path) as image:
        return image.convert('RGB')


def read_finished_rows():
    rows = {}
    sources = [('manifest_1.jsonl', 1, 2400), ('manifest_2.jsonl', 2401, 4825), ('manifest_3.jsonl', 3741, 3741)]
    for filename, first, last in sources:
        path = A / filename
        if not path.exists():
            continue
        data = path.read_bytes()
        # Each capture appends concurrently: ignore a not-yet-finished line.
        lines = data.splitlines()
        if data and not data.endswith((b'\n', b'\r')):
            lines = lines[:-1]
        for line in lines:
            if line.strip():
                row = json.loads(line.decode('utf-8-sig'))
                index = int(row['math_index'])
                if first <= index <= last:
                    row['capture_manifest'] = filename
                    rows[index] = row
    return rows


def native_tiles(row):
    image = viewport(int(row['viewport']))
    x, y, width, height = row['getpoint_local_rect']
    box = (max(0, x-PAD_X), max(0, y-PAD_Y),
           min(image.width, x+width+PAD_X), min(image.height, y+height+PAD_Y))
    crop = image.crop(box)
    capacity = WIDTH-2*MARGIN
    # A wide original formula is split with overlap, never resized.
    offsets = [0]
    while offsets[-1]+capacity < crop.width:
        offsets.append(offsets[-1]+capacity-32)
    for part, offset in enumerate(offsets, 1):
        tile = crop.crop((offset, 0, min(crop.width, offset+capacity), crop.height))
        yield tile, box, part, len(offsets), offset


def build_group(group, indices, rows):
    placements = []
    sheets = []
    part = 0
    canvas = None
    x = y = shelf_h = 0
    current = []

    def new_sheet():
        nonlocal canvas, part, x, y, shelf_h, current
        part += 1
        canvas = Image.new('RGB', (WIDTH, MAX_HEIGHT), 'white')
        draw = ImageDraw.Draw(canvas)
        draw.text((MARGIN, 8), f'M{indices[0]:06d}-M{indices[-1]:06d} | part {part} | Word UI, native scale', fill='black', font=FONT)
        x, y, shelf_h, current = MARGIN, 38, 0, []

    def save_sheet():
        filename = f'group_{group:02d}_part_{part:02d}.png'
        used_height = min(MAX_HEIGHT, y+shelf_h+MARGIN)
        canvas.crop((0, 0, WIDTH, used_height)).save(OUT / filename)
        sheets.append({'file': filename, 'width': WIDTH, 'height': used_height,
                       'math_indices': sorted(set(current))})

    new_sheet()
    for index in indices:
        row = rows[index]
        for tile, box, tile_part, tile_count, offset in native_tiles(row):
            item_w = max(tile.width, 100)
            item_h = tile.height+LABEL_H
            if x+item_w > WIDTH-MARGIN:
                x, y, shelf_h = MARGIN, y+shelf_h+GAP, 0
            if y+item_h > MAX_HEIGHT-MARGIN:
                save_sheet()
                new_sheet()
            label = f'M{index:06d}'
            if tile_count > 1:
                label += f' [{tile_part}/{tile_count}]'
            ImageDraw.Draw(canvas).text((x, y+2), label, fill='black', font=FONT)
            canvas.paste(tile, (x, y+LABEL_H))
            placements.append({'math_index': index, 'name': f'M{index:06d}',
                'sheet': f'group_{group:02d}_part_{part:02d}.png',
                'image_rect_xywh': [x, y+LABEL_H, tile.width, tile.height],
                'native_scale': 1, 'viewport': row['viewport'],
                'capture_manifest': row.get('capture_manifest'),
                'source_viewport_crop_ltrb': list(box),
                'horizontal_tile': tile_part, 'horizontal_tile_count': tile_count,
                'horizontal_offset': offset, 'start_page': row['start_page'],
                'end_page': row['start_page']})
            current.append(index)
            x += item_w+GAP
            shelf_h = max(shelf_h, item_h)
    save_sheet()
    result = {'group': group, 'indices': indices, 'sheets': sheets,
              'placements': placements, 'source': 'native_viewports + original GetPoint rectangles',
              'padding_xy': [PAD_X, PAD_Y], 'native_scale': 1}
    (OUT / f'group_{group:02d}.json').write_text(json.dumps(result, ensure_ascii=False, indent=2)+'\n')
    return result


rows = read_finished_rows()
groups = []
for first in range(1, EXPECTED+1, 100):
    indices = list(range(first, min(first+100, EXPECTED+1)))
    if not all(index in rows for index in indices):
        continue
    group = (first-1)//100+1
    cached = OUT / f'group_{group:02d}.json'
    if cached.exists():
        result = json.loads(cached.read_text())
    else:
        result = build_group(group, indices, rows)
        print(f'Built group {group:02d}: {len(indices)} formulas, {len(result["sheets"])} sheets', flush=True)
    groups.append(result)

covered = sorted({index for group in groups for index in group['indices']})
index = {'expected_math_objects': EXPECTED, 'capture_rows_available': len(rows),
         'atlased_math_objects': len(covered), 'all_4825_covered': covered == list(range(1, EXPECTED+1)),
         'covered_math_indices': covered,
         'not_yet_atlased_math_indices': sorted(set(range(1, EXPECTED+1))-set(covered)),
         'sheet_width': WIDTH, 'sheet_max_height': MAX_HEIGHT, 'native_scale': 1,
         'padding_xy': [PAD_X, PAD_Y], 'group_math_count': 100,
         'sheets': [sheet for group in groups for sheet in group['sheets']],
         'placements': [placement for group in groups for placement in group['placements']],
         'original_captures_modified': False}
index_temp = OUT / 'atlas_index.tmp'
index_temp.write_text(json.dumps(index, ensure_ascii=False, indent=2)+'\n')
index_temp.replace(OUT / 'atlas_index.json')
print(json.dumps({'capture_rows_available': len(rows), 'atlased_math_objects': len(covered),
                  'sheets': len(index['sheets']), 'all_4825_covered': index['all_4825_covered']}), flush=True)
