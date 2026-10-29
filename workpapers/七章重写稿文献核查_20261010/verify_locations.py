"""Detect broken reference targets, misnumbered links, and unlinked citations."""
import json
from pathlib import Path
import re
from zipfile import ZipFile

from lxml import etree as ET

from audit import NS, SOURCE, read, write, HERE

refs = read(HERE / '当前文献与引用.json')
blocks = read(HERE / '当前正文块.json')
by_key = {r['key']: r['number'] for r in refs}
with ZipFile(SOURCE) as package:
    body = ET.fromstring(package.read('word/document.xml')).find('w:body', NS)

missing, wrong_numbers, unlinked = [], [], []
count = 0
for i, node in enumerate(body):
    linked_numbers = set()
    for link in node.xpath('.//w:hyperlink[starts-with(@w:anchor,"ref-")]', namespaces=NS):
        key = link.get('{' + NS['w'] + '}anchor')
        if key not in by_key:
            missing.append(dict(block=i, anchor=key))
            continue
        text = ''.join(link.xpath('.//w:t/text()', namespaces=NS))
        digits = re.findall(r'\d+', text)
        if digits != [str(by_key[key])]:
            wrong_numbers.append(dict(block=i, anchor=key, display=text))
        linked_numbers.add(by_key[key])
        count += 1
    if blocks[i]['style'] == 'bibliography':
        continue
    matches = re.findall(r'\[(\d+(?:\s*[,，-]\s*\d+)*)\]', blocks[i]['text'])
    for match in matches:
        numbers = [int(s) for s in re.findall(r'\d+', match)]
        if 0 in numbers or any(n > len(refs) for n in numbers):
            continue
        if '-' in match and len(numbers) == 2:
            numbers = list(range(numbers[0], numbers[-1] + 1))
        if set(numbers) - linked_numbers:
            unlinked.append(dict(block=i, bracket=match, style=blocks[i]['style'], text=blocks[i]['text']))

identifiers = {}
duplicates = []
for ref in refs:
    identifier = ref['arxiv'].split('v')[0] or ref['doi'].lower() or ref['prior_source']
    if identifier in identifiers:
        duplicates.append([identifiers[identifier], ref['number']])
    identifiers[identifier] = ref['number']

result = dict(reference_entries=len(refs), reference_hyperlinks=count,
              missing_targets=missing, wrong_link_numbers=wrong_numbers,
              possible_unlinked_citations=unlinked, duplicate_identifiers=duplicates,
              uncited_entries=[r['number'] for r in refs if not r['uses']])
write('引用结构检查.json', result)
print(json.dumps(result, ensure_ascii=False, indent=2))
