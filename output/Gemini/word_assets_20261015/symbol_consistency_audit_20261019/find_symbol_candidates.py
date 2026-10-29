"""Find candidate notation-table gaps; findings still require contextual review."""
from pathlib import Path
import json
import re
from collections import Counter

A = Path(__file__).resolve().parent
formulas = json.loads((A / 'formulas.json').read_text())
rows = json.loads((A / 'symbol_table.json').read_text())
by_id = {x['math_id']: x for x in formulas}
body = [f for f in formulas if f['paragraph'] >= 416 and f['chapter'] not in
        ['参考文献', '攻读学位期间主要的研究成果', '致  谢']]

def normalize(tex):
    tex = re.sub(r'\\(?:left|right|middle|displaystyle|limits|nolimits)\b', '', tex)
    tex = re.sub(r'\\(?:mathrm|mathit|mathbf|mathsf|mathbb|mathcal|mathscr|text|operatorname)\b', '', tex)
    tex = re.sub(r'\\(?:quad|qquad|thinspace|,|;|!| )', '', tex)
    tex = tex.replace('\\lbrack', '[').replace('\\rbrack', ']')
    tex = tex.replace('\\parallel', '∥').replace('\\mid', '∣')
    tex = re.sub(r'\s+', '', tex)
    return tex.replace('{', '').replace('}', '')

def split(tex):
    tex = re.sub(r'\\(mathcal|mathsf|mathbb)\{([^{}]*,[^{}]*)\}',
                 lambda m: ','.join('\\' + m[1] + '{' + x + '}' for x in m[2].split(',') if x), tex)
    result, start, depth = [], 0, 0
    for i, char in enumerate(tex):
        if char == '{':
            depth += 1
        elif char == '}':
            depth -= 1
        elif char in ',，；;' and depth == 0:
            if tex[start:i].strip():
                result.append(tex[start:i].strip())
            start = i + 1
    if tex[start:].strip():
        result.append(tex[start:].strip())
    return result

canonical = [(f, normalize(f['tex'])) for f in body]
results = []
for row in rows:
    entries = []
    for math_id in row['math_ids']:
        for symbol in split(by_id[math_id]['tex']):
            key = normalize(symbol)
            regex = re.compile(r'(?<![A-Za-z])' + re.escape(key) + r'(?![A-Za-z])')
            matches = [f for f, text in canonical if regex.search(text)]
            entries.append({'symbol': symbol, 'candidate_matches': len(matches),
                            'examples': [{k: f[k] for k in ['math_id', 'paragraph', 'chapter', 'formula_number', 'tex']}
                                         for f in matches[:5]],
                            'chapters': dict(Counter(f['chapter'] for f in matches))})
    results.append({**row, 'entries': entries})
(A / 'table_match_candidates.json').write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding='utf-8')
for row in results:
    missing = [e['symbol'] for e in row['entries'] if not e['candidate_matches']]
    if missing:
        print(row['row'], ' | '.join(missing))
print('Candidate matching ignores fonts and cannot establish semantic equivalence.')
