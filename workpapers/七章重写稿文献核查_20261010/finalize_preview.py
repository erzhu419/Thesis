"""Keep Word's TOC page results and inspect only the changed rendered pages."""
import importlib.util
import json
from pathlib import Path
import shutil
import sys
from zipfile import ZipFile

import fitz

from audit import HERE, ROOT, SOURCE, read, write
from finish_audit import OUT, REPORT

STAGE = Path('/mnt/c/Users/erzhu/AppData/Local/Temp/thesis_md_merge_20261010')
PDF = ROOT / 'output/排版完成_七章整合_文献核验_20261011.pdf'
sys.path.insert(0, str(ROOT / 'workpapers/七章重写稿整合_20261010'))
spec = importlib.util.spec_from_file_location('toc_cache', ROOT / 'workpapers/七章重写稿整合_20261010/refresh_toc_cache.py')
helper = importlib.util.module_from_spec(spec)
spec.loader.exec_module(helper)
helper.OUT, helper.WORK, helper.STAGE = OUT, HERE, str(STAGE / 'citation_audit.docx')
helper.main()
shutil.copy2(STAGE / 'citation_audit.pdf', PDF)

base = fitz.open(ROOT / 'output/排版完成_七章重写稿整合_20261010.pdf')
final = fitz.open(PDF)
assert len(base) == len(final) == 290
changed = [i + 1 for i, (a,b) in enumerate(zip(base,final)) if a.get_text() != b.get_text()]
first_bib = next(i for i,p in enumerate(base) if '[1]' in p.get_text() and 'DAGANZO C F' in p.get_text())
last_bib = next(i for i,p in enumerate(base) if '[214]' in p.get_text() and 'source code' in p.get_text())
assert all(first_bib <= p-1 <= last_bib for p in changed), changed
assert all(doi in ''.join(p.get_text() for p in final) for doi in (
    '10.16097/j.cnki.1009-6744.2023.06.017',
    '10.16097/j.cnki.1009-6744.2024.05.016'))
folder = HERE / '新增DOI页面预览'
folder.mkdir(exist_ok=True)
for n in changed:
    final[n-1].get_pixmap(matrix=fitz.Matrix(1.5,1.5)).save(folder / ('page_' + str(n) + '.png'))

result = dict(WordVersion='16.0', WordBuild='16.0.14334', pdf_pages=290,
              changed_pdf_pages=changed, bibliography_pdf_pages=[first_bib+1,last_bib+1],
              non_bibliography_page_text_identical=True,
              chapter_start_pages=[19,39,69,106,134,168,213],
              visual_review='待人工查看已导出的变化页，不提前标记通过')
write('Word分页及预览核对.json', result)
print(json.dumps(result, ensure_ascii=False, indent=2))
