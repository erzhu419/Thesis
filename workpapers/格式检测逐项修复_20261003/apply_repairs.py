"""Patch the current Word package without rebuilding its content or objects."""
from pathlib import Path
from zipfile import ZipFile, ZIP_DEFLATED
import json
from lxml import etree
from apply_semantics import apply_semantics
from fix_report_fonts import apply_report_fonts

ROOT = Path(__file__).resolve().parents[2]
WORK = Path(__file__).resolve().parent
source = WORK / '排版完成_修复前.docx'
target = WORK / '排版完成_修复候选.docx'
with ZipFile(source) as original:
    document = etree.fromstring(original.read('word/document.xml'))
    styles = etree.fromstring(original.read('word/styles.xml'))
    changes = {
        'source': str(source),
        'semantics': apply_semantics(document, styles),
        'fonts': apply_report_fonts(document),
    }
    replacements = {
        'word/document.xml': etree.tostring(document, encoding='UTF-8', xml_declaration=True, standalone=True),
        'word/styles.xml': etree.tostring(styles, encoding='UTF-8', xml_declaration=True, standalone=True),
    }
    with ZipFile(target, 'w', ZIP_DEFLATED) as output:
        for item in original.infolist():
            output.writestr(item, replacements.get(item.filename, original.read(item.filename)))
(WORK / 'applied_changes.json').write_text(json.dumps(changes, ensure_ascii=False, indent=2)+'\n')
print(json.dumps({'candidate': str(target), 'western_runs': len(changes['fonts']['western']),
                  'native_h_runs': len(changes['fonts']['native_math_h'])}, ensure_ascii=False))
