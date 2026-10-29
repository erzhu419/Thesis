from zipfile import ZipFile
import re
import xml.etree.ElementTree as ET

from edit_pass2_section_1_2 import BACKUP, DOCUMENT, REVISIONS, W
from edit_section_3_2 import prose_groups
from edit_section_4_4 import replace_paragraph


with ZipFile(DOCUMENT) as source:
    original = source.read("word/document.xml").decode("utf-8")
    body = ET.fromstring(original).find(W + "body")
    nodes, groups, protected = prose_groups(body[296])
    prose = ["".join(nodes[i].text or "" for i in group) for group in groups]
    prose[2] = (
        "作者提出的鲁棒集成软演员评论家方法（Robust Ensemble Soft Actor-Critic，RE-SAC）"
        "由此引入目标价值偏移与集成保守评价。第3章分析这两类修正对驻站动作的作用，"
        "相关鲁棒解释及成立条件见"
    )
    updated = replace_paragraph(original, body[296], prose)
    paragraph = re.search(r'<w:p\b[^>]*w14:paraId="1BD58705"[^>]*>.*?</w:p>', updated, re.S)
    assert paragraph is not None
    old = paragraph.group(0)
    boundary = '<w:fldChar w:fldCharType="end"/></w:r>'
    assert old.count(boundary) == 1 and ' REF appA ' in old
    # The cached appendix name is supplied by Word; punctuation follows the field.
    new = old.replace(boundary, '<w:fldChar w:fldCharType="end"/><w:t>。</w:t></w:r>')
    updated = updated[:paragraph.start()] + new + updated[paragraph.end():]
    strip_text = lambda xml: re.sub(r'<w:t(?:\s[^>]*)?>.*?</w:t>', '', xml, flags=re.S)
    assert strip_text(original) == strip_text(updated)
    revised = ET.fromstring(updated).find(W + "body")
    assert [i for i, (a, b) in enumerate(zip(body, revised)) if ET.tostring(a) != ET.tostring(b)] == [296]
    assert [ET.tostring(n) for n in body[296].iter(W + 'fldChar')] == [ET.tostring(n) for n in revised[296].iter(W + 'fldChar')]
    assert [ET.tostring(n) for n in body[296].iter(W + 'instrText')] == [ET.tostring(n) for n in revised[296].iter(W + 'instrText')]
    with ZipFile(DOCUMENT.with_suffix('.tmp'), 'w') as target:
        for item in source.infolist():
            target.writestr(item, updated.encode('utf-8') if item.filename == 'word/document.xml' else source.read(item.filename))
DOCUMENT.with_suffix('.tmp').replace(DOCUMENT)
with ZipFile(BACKUP) as before, ZipFile(DOCUMENT) as after:
    assert [n for n in before.namelist() if before.read(n) != after.read(n)] == ['word/document.xml']
    original_backup = before.read('word/document.xml').decode('utf-8')
    assert strip_text(original_backup) == strip_text(updated)
    backup_body = ET.fromstring(original_backup).find(W+'body')
    changed = [i for i, (a, b) in enumerate(zip(backup_body, revised)) if ET.tostring(a) != ET.tostring(b)]
    assert changed == sorted([*REVISIONS, 296]), changed
print({'changed_blocks': changed, 'field_target_preserved': 'appA'})
