"""Record the contact sheets actually inspected in this editing turn."""
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
records = []
for stem, expected in [('visual_acceptance', 64), ('visual_final_delta', 10),
                       ('visual_endpoint_delta', 4)]:
    path = HERE / (stem + '.json')
    data = json.loads(path.read_text())
    assert len(data['changed_pages']) == expected, (stem, data['changed_pages'])
    data['changed_pages_visually_reviewed'] = True
    data['review_method'] = 'Every listed four-page contact sheet was opened and visually inspected.'
    data['findings_after_repair'] = []
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding='utf-8')
    records.append({'record': path.name, 'pages': data['changed_pages'],
                    'sheets': len(data['sheets'])})
summary = {
    'final_pages': 317,
    'baseline': 'Previous Word-native 317-page preview, all pages inspected in the preceding pass.',
    'passes': records,
    'all_changed_pages_inspected': True,
    'unchanged_pages': 'Reused only where direct rendered pixels were identical to the inspected preceding preview.',
    'scope': 'A cumulative whole-document visual acceptance, not a claim that all 317 pages were freshly reopened this turn.'}
(HERE / '逐页视觉验收汇总.json').write_text(
    json.dumps(summary, ensure_ascii=False, indent=2), encoding='utf-8')
print(json.dumps(summary, ensure_ascii=False))
