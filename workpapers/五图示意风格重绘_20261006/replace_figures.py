from pathlib import Path
from zipfile import ZipFile
import json
import shutil
from lxml import etree as E
from PIL import Image

root = Path(__file__).resolve().parents[2]
work = Path(__file__).resolve().parent
source = root / '排版完成_分页图表修复_20261006.docx'
target = root / '排版完成_五图公交示意版_20261006.docx'
assets = root / 'output/imagegen/论文五图_示意风格_20261006'
mapping = {'2-1': 'image2.png', '2-3': 'image4.png', '4-1': 'image21.png',
           '5-2': 'image32.png', '6-1': 'image44.png'}
assert source.read_bytes() == (work / '修改前.docx').read_bytes(), 'Baseline was edited during drawing; inspect before replacement.'
chosen = {fig: f'图{fig}_'+({'2-3':'修正','4-1':'精修'}.get(fig,'定稿'))+'.png' for fig in mapping}
chosen['2-1']='图2-1_到站定义.png'
replacements = {'word/media/'+part: (assets / chosen[fig]).read_bytes()
                for fig, part in mapping.items()}
ns={'w':'http://schemas.openxmlformats.org/wordprocessingml/2006/main',
    'a':'http://schemas.openxmlformats.org/drawingml/2006/main',
    'pic':'http://schemas.openxmlformats.org/drawingml/2006/picture',
    'wp':'http://schemas.openxmlformats.org/drawingml/2006/wordprocessingDrawing',
    'r':'http://schemas.openxmlformats.org/officeDocument/2006/relationships'}
with ZipFile(source) as src:
    original_xml=src.read('word/document.xml');doc=E.fromstring(original_xml)
    rels=E.fromstring(src.read('word/_rels/document.xml.rels'))
    rid=next(r.get('Id') for r in rels if r.get('Target')=='media/image2.png')
pic=doc.xpath('.//pic:pic[pic:blipFill/a:blip/@r:embed=$rid]',namespaces=ns,rid=rid)
assert len(pic)==1
extent=pic[0].xpath('ancestor::wp:inline/wp:extent',namespaces=ns)[0]
with Image.open(assets/chosen['2-1']) as im: image_ratio=im.width/im.height
container_ratio=int(extent.get('cx'))/int(extent.get('cy'))
inset=round((1-container_ratio/image_ratio)*50000)
fill=pic[0].find('pic:blipFill/a:stretch/a:fillRect',ns)
assert fill is not None and inset>0
old_attributes=dict(fill.attrib)
fill.set('t',str(inset));fill.set('b',str(inset))
replacements['word/document.xml']=E.tostring(doc,xml_declaration=True,encoding='UTF-8',standalone=True)
# Only the picture's inner fit changes; its Word frame and all text stay fixed.
fill.attrib.clear();fill.attrib.update(old_attributes)
assert E.tostring(doc)==E.tostring(E.fromstring(original_xml))
with ZipFile(source) as src, ZipFile(target, 'w') as dst:
    for entry in src.infolist():
        dst.writestr(entry, replacements.get(entry.filename, src.read(entry.filename)))
with ZipFile(source) as src, ZipFile(target) as dst:
    assert src.namelist() == dst.namelist()
    changed = [n for n in src.namelist() if src.read(n) != dst.read(n)]
    assert sorted(changed) == sorted(replacements), changed
final = work / '交付图片'; final.mkdir(exist_ok=True)
for fig in mapping:
    shutil.copy2(assets / chosen[fig], final / f'图{fig}.png')
record = {'baseline': str(source), 'word': str(target), 'changed_parts': changed,
          'all_other_parts_identical': True, 'original_image_extents_preserved': True,
          'figure_2_1_vertical_inset_percent': inset/1000,
          'document_xml_only_change': 'figure 2-1 native picture fit, no text/paragraph/equation changes',
          'model': 'gpt-image-2', 'endpoint': 'https://ruoli.dev/v1', 'selected_images': chosen,
          'method': 'imagegen bundled CLI edit with two style references',
          'references': ['/home/erzhu419/mine_code/MC-WM/paper/figures/architecture.pdf',
                         '/mnt/d/Paper/RL with bus control/bus_holding_demonstration.pdf']}
(work / '交付记录.json').write_text(json.dumps(record, ensure_ascii=False, indent=2))
print(json.dumps(record, ensure_ascii=False, indent=2))
