from pathlib import Path
from zipfile import ZipFile,ZIP_DEFLATED
import json
from lxml import etree
from fix_native_letterlike_fonts import apply_native_letterlike_fonts

HERE=Path(__file__).resolve().parent
source=HERE/'排版完成_修复候选.docx'
before=HERE/'排版完成_花体补充修复前.docx'
before.write_bytes(source.read_bytes())
with ZipFile(before) as original:
    document=etree.fromstring(original.read('word/document.xml'))
    changes=apply_native_letterlike_fonts(document)
    with ZipFile(source,'w',ZIP_DEFLATED) as output:
        for item in original.infolist():
            data=etree.tostring(document,encoding='UTF-8',xml_declaration=True,standalone=True) if item.filename=='word/document.xml' else original.read(item.filename)
            output.writestr(item,data)
(HERE/'花体字体补充修复.json').write_text(json.dumps(changes,ensure_ascii=False,indent=2)+'\n')
(HERE/'排版完成_最终分页前.docx').write_bytes(source.read_bytes())
print('Supplement applied without changing text, mathematical alphabet selectors or size.')
