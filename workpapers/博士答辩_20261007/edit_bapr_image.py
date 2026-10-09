from pathlib import Path
import base64
import getpass

import requests

here = Path(__file__).resolve().parent
source = here / 'comment_assets' / 'bapr_filter_generated.png'
key = getpass.getpass('API key: ')
prompt = '''Edit the provided Chinese scientific diagram with minimal changes.
Preserve the exact three-panel composition, every Chinese label, all probability bars,
colors, funnel, arrows and bottom flow. ONLY change the four miniature black state-change
curves in the middle panel under 连续观测到状态变化.
They currently look almost identical. Make the four mini-curves clearly visually different:
1. A sharp downward step early in its box, then a stable low plateau.
2. A smooth gradual downward ramp, not a step.
3. A delayed downward change with a small temporary upward rebound and mild noise.
4. Two successive downward steps of different magnitudes.
All four start higher on the left and end lower on the right, but must differ in timing,
amplitude and shape. They are schematic observations favoring the same new mode B,
not four unrelated functions. Use thin crisp dark-grey lines and retain the pale-grey boxes.
No new labels, numbers or symbols. Do not change any other portion of the image.
Maintain image resolution, aspect ratio, Chinese typography and white background.
'''
with source.open('rb') as image:
    response = requests.post(
        'https://ruoli.dev/v1/images/edits',
        headers={'Authorization': 'Bearer '+key},
        data={'model': 'gpt-image-2', 'prompt': prompt, 'size': '1536x1024', 'n': '1'},
        files={'image': (source.name, image, 'image/png')}, timeout=600,
    )
if not response.ok:
    safe = response.text.replace(key, '[redacted]')
    raise SystemExit(f'HTTP {response.status_code}: {safe[:1200]}')
item = response.json()['data'][0]
if item.get('b64_json'):
    blob = base64.b64decode(item['b64_json'])
else:
    result = requests.get(item['url'], timeout=180)
    result.raise_for_status()
    blob = result.content
target = here / 'comment_assets' / 'bapr_filter_generated_v2.png'
target.write_bytes(blob)
print('Saved revised BAPR diagram:', target)
