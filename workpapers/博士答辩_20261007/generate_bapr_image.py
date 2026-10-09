from pathlib import Path
import base64
import getpass
import json
import urllib.error
import urllib.request

here = Path(__file__).resolve().parent
key = getpass.getpass('API key: ')
prompt = '''Create a clean Chinese scientific presentation diagram, landscape, white background.
Topic: belief filtering in BAPR. This is an explanatory diagram, not experimental results.
Use three large stages connected by left-to-right arrows, with large readable Chinese labels.
Stage 1 heading: 保持当前模式. Show two candidate operating modes A and B. A has a strong
teal probability bar; B has a smaller coral bar. A small grey noisy observation does not change
the leading mode. Small label: 单次噪声不急于切换.
Stage 2 heading: 积累转移证据. Show several sequential observations of state changes, with
new evidence favoring B. Use a funnel or evidence integration symbol, NOT neural-network art.
Small label: 旧信念＋新证据.
Stage 3 heading: 更新模式判断. Show B now has the larger coral probability bar, A the smaller
teal bar. Small label: 充分证据才改变判断.
Include one clear bottom causal flow: 已完成的转移 → 模式信念 → 下一次动作.
Use teal #087F8C, coral #C9654B, muted purple #78618C, dark graphite text.
Restrained academic vector-like infographic, thin straight lines, generous spacing, no 3D,
no gradients, no photo, no watermark, no decorative icons, no long prose, no numerical claims,
no mathematical formulas. Chinese characters must be exactly readable. Content must fit fully.
'''
body = json.dumps({'model': 'gpt-image-2', 'prompt': prompt,
                   'size': '1536x1024', 'n': 1}).encode()
request = urllib.request.Request('https://ruoli.dev/v1/images/generations', data=body,
                                 headers={'Authorization': 'Bearer '+key,
                                          'Content-Type': 'application/json'})
try:
    with urllib.request.urlopen(request, timeout=600) as response:
        result = json.load(response)
except urllib.error.HTTPError as error:
    safe = error.read().decode(errors='replace').replace(key, '[redacted]')
    raise SystemExit(f'HTTP {error.code}: {safe[:1200]}')
item = result['data'][0]
if item.get('b64_json'):
    blob = base64.b64decode(item['b64_json'])
else:
    # Image download carries no API credential to the returned media host.
    with urllib.request.urlopen(item['url'], timeout=180) as response:
        blob = response.read()
path = here / 'comment_assets' / 'bapr_filter_generated.png'
path.write_bytes(blob)
print('Saved BAPR diagram:', path)
