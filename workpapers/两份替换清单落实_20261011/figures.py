"""Relabel copies of existing figures; preserve quantitative vector geometry."""
from io import BytesIO
from pathlib import Path
from zipfile import ZipFile
import json
import re
import sys
import os

from lxml import etree as E
from PIL import Image

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
sys.path.insert(0,str(ROOT/'workpapers/全文编号符号简称复核_20261011'))
from scan import content,W
SOURCE=ROOT/'output/排版完成_残留问题替换_20261011.docx'
OUT=ROOT/'output/排版完成_双清单替换_20261011.docx'
S='{http://www.w3.org/2000/svg}'
R='{http://schemas.openxmlformats.org/officeDocument/2006/relationships}'
A='{http://schemas.openxmlformats.org/drawingml/2006/main}'
O='{urn:schemas-microsoft-com:office:office}'
V='{urn:schemas-microsoft-com:vml}'
VN='{http://schemas.microsoft.com/office/visio/2012/main}'
NUMBERS=['2-3','2-6','2-7','3-8','3-9','3-10','3-12','4-2','4-4','4-5','4-6','4-8','4-10',
         '5-7','5-8','5-9','5-10','5-12','5-13','6-9','6-10','6-12','6-14','6-16','6-17','F-1','F-2','F-3']
LABELS={
    '2-6':{'广义乘客时间':'人均广义时间成本'},
    '2-7':{'广义时间响应差':'人均广义时间成本响应差'},
    '3-8':{'人均广义时间':'人均广义时间成本'},
    '3-9':{'人均广义时间':'人均广义时间成本'},
    '3-10':{'人均广义时间':'人均广义时间成本'},
    '3-12':{'人均广义时间':'人均广义时间成本'},
    '4-2':{'早期BAPR':'BAPR（部分稳定化）'},
    '4-4':{'早期BAPR':'BAPR（部分稳定化）'},
    '4-5':{'前序BAPR':'BAPR（未采用稳定化设计）'},
    '4-6':{'早期BAPR':'BAPR（部分稳定化）'},
    '4-8':{'前序BAPR':'BAPR'},
    '4-10':{'前序BAPR':'BAPR'},
    '5-7':{'每出发乘客时间差':'人均广义时间成本差'},
    '5-8':{'每出发乘客广义时间减少':'人均广义时间成本减少量'},
    '5-9':{'每出发乘客时间':'人均广义时间成本','广义时间 / 秒':'人均广义时间成本 / 秒'},
    '5-10':{'广义时间变化 / 秒每人':'人均广义时间成本变化 / 秒'},
    '5-12':{'人均广义时间差':'人均广义时间成本差'},
    '5-13':{'候车收益、车内代价与净时间变化':'人均广义时间成本的候车贡献、车内贡献与净变化'},
    '6-9':{'人均广义时间减少量':'人均广义时间成本减少量'},
    '6-10':{'人均广义时间：负值表示MC-WM较低':'人均广义时间成本差：负值表示MC-WM较低'},
    '6-12':{'广义时间减少':'人均广义时间成本减少量'},
    '6-14':{'规则减方法的广义时间 / 秒':'人均广义时间成本差（规则减方法）/ 秒'},
    '6-16':{'方法减规则的广义时间 / 秒':'人均广义时间成本差（方法减规则）/ 秒'},
    '6-17':{'广义时间分解':'人均广义时间成本分解','净成本差':'人均广义时间成本净差'},
    'F-1':{'人均广义时间':'人均广义时间成本'},
    'F-3':{'广义时间下降':'人均广义时间成本降幅'},
}


def mapping():
    folder=HERE/'图内标签'
    folder.mkdir(exist_ok=True)
    files={p.stem:p.read_bytes() for p in (ROOT/'manuscript/figures').glob('*.png')}
    result=[]
    with ZipFile(SOURCE) as z:
        b=E.fromstring(z.read('word/document.xml')).find(W+'body')
        rels={r.get('Id'):r.get('Target') for r in E.fromstring(z.read('word/_rels/document.xml.rels'))}
        for i,p in enumerate(b):
            match=re.match(r'^图('+ '|'.join(NUMBERS)+')\s',content(p))
            if not match:
                continue
            fig=match[1];picture=b[i-1]
            obj=picture.find('.//'+W+'object')
            if obj is not None:
                embed='word/'+rels[obj.find(O+'OLEObject').get(R+'id')]
                preview='word/'+rels[obj.find('.//'+V+'imagedata').get(R+'id')]
                path=folder/f'图{fig}_修改前.vsdx'
                path.write_bytes(z.read(embed))
                with ZipFile(path) as vz:
                    page=E.fromstring(vz.read('visio/pages/page1.xml'))
                texts=[''.join(t.itertext()) for t in page.iter(VN+'Text')]
                result.append(dict(figure=fig,embedding=embed,preview=preview,texts=texts))
            else:
                blip=picture.find('.//'+A+'blip')
                part='word/'+rels[blip.get(R+'embed')]
                raw=z.read(part)
                names=[name for name,data in files.items() if data==raw]
                if not names:
                    current=Image.open(BytesIO(raw))
                    for name,data in files.items():
                        other=Image.open(BytesIO(data))
                        if current.size==other.size and current.convert('RGB').tobytes()==other.convert('RGB').tobytes():
                            names.append(name)
                if fig=='F-2':
                    assert raw==(ROOT/'workpapers/F2最终控制器结果更新_20261006/图F-2.png').read_bytes()
                    result.append(dict(figure=fig,preview=part,source='F2_current',texts=[],
                                       width=Image.open(BytesIO(raw)).width))
                    continue
                assert len(names)==1,(fig,names)
                svg=ROOT/'manuscript/figures'/f'{names[0]}.svg'
                tree=E.parse(str(svg))
                texts=[''.join(t.itertext()) for t in tree.iter(S+'text')]
                result.append(dict(figure=fig,preview=part,source=names[0],texts=texts,
                                   width=Image.open(BytesIO(raw)).width))
    (HERE/'图内标签映射.json').write_text(json.dumps(result,ensure_ascii=False,indent=2))
    for row in result:
        print(row['figure'],row.get('source','Visio'),[t for t in row['texts'] if any(s in t for s in ['广义','乘客','BAPR','净','时间','完成'])])


def relabel():
    os.environ['FONTCONFIG_FILE']=str(HERE/'fonts.conf')
    import cairosvg
    rows=json.loads((HERE/'图内标签映射.json').read_text())
    folder=HERE/'图内标签'
    for row in rows:
        fig=row['figure']
        if fig in {'2-3','F-2'}:
            continue
        path=ROOT/'manuscript/figures'/f"{row['source']}.svg"
        tree=E.parse(str(path));geometry=[E.tostring(p) for p in tree.iter(S+'path')]
        edits=[]
        for t in tree.iter(S+'text'):
            before=t.text or '';after=before
            for old,new in LABELS[fig].items():
                after=after.replace(old,new)
            if before==after:
                continue
            edits.append([before,after]);t.text=after
            if fig=='4-4' and after=='BAPR（部分稳定化）':
                x=t.get('x');y=float(t.get('y'));t.text='BAPR';t.set('y',str(y-4))
                E.SubElement(t,S+'tspan',{'x':x,'y':str(y+6),'style':'font-size: 7.5px'}).text='（部分稳定化）'
            if fig=='4-5' and after.startswith('BAPR（'):
                t.text='BAPR'
                x=t.get('x');y=float(t.get('y'))
                t.set('y',str(y-4.5))
                E.SubElement(t,S+'tspan',{'x':x,'y':str(y+6.0)}).text='（未采用稳定化设计）'
                t.set('style',re.sub(r'font-size: [\d.]+px','font-size: 8px',t.get('style')))
            if fig=='4-5' and after.startswith('(b)'):
                t.set('style',re.sub(r'font-size: [\d.]+px','font-size: 8px',t.get('style')))
            if fig in {'6-14','6-16'}:
                t.set('style',re.sub(r'font-size: [\d.]+px','font-size: 8.5px',t.get('style')))
        assert edits,(fig,'no matched labels')
        assert geometry==[E.tostring(p) for p in tree.iter(S+'path')]
        target=folder/f'图{fig}.svg'
        tree.write(str(target),encoding='utf-8',xml_declaration=True)
        cairosvg.svg2png(url=str(target),write_to=str(target.with_suffix('.png')),output_width=row['width'])
        row['edits']=edits
    # F-2 has no stored SVG; run only its original figure section on copied outputs.
    source=ROOT/'workpapers/F2最终控制器结果更新_20261006/update_f2.py'
    code=source.read_text().split('\nNS = {',1)[0]
    code=code.replace('HERE = Path(__file__).resolve().parent',f'HERE = Path({str(folder)!r})')
    code=code.replace('ROOT = HERE.parents[1]',f'ROOT = Path({str(ROOT)!r})')
    code=code.replace('assert SOURCE.read_bytes() == BACKUP.read_bytes(), "The latest manuscript changed after backup"','')
    code=code.replace('人均广义时间（秒）','人均广义时间成本（秒）')
    code=code.replace('人均广义时间差（秒；负值表示前者更优）','人均广义时间成本差（秒；负值表示前者更优）')
    code=code.replace('"pdf.fonttype": 42','"pdf.fonttype": 42, "svg.fonttype": "none"')
    code=code.replace('fig.savefig(HERE / "图F-2.pdf")','fig.savefig(HERE / "图F-2.svg")')
    exec(compile(code,str(source),'exec'),{'__file__':str(source),'__name__':'relabel_f2'})
    assert (folder/'图F-2_作图数据.csv').read_bytes()==(ROOT/'workpapers/F2最终控制器结果更新_20261006/图F-2_作图数据.csv').read_bytes()
    # Narrow Visio text edit on an extracted copy; no geometry or cells change.
    path=folder/'图2-3_修改前.vsdx'
    with ZipFile(path) as z:
        raw=z.read('visio/pages/page1.xml');page=E.fromstring(raw)
        hits=[]
        for text in page.iter(VN+'Text'):
            for element in text.iter():
                for attr in ['text','tail']:
                    value=getattr(element,attr)
                    if value and '广义时间成本' in value:
                        setattr(element,attr,value.replace('广义时间成本','人均广义时间成本'));hits.append(value)
        assert len(hits)==1,hits
        with ZipFile(folder/'图2-3.vsdx','w') as target:
            for info in z.infolist():
                target.writestr(info,E.tostring(page) if info.filename=='visio/pages/page1.xml' else z.read(info.filename))
    (HERE/'图内标签映射.json').write_text(json.dumps(rows,ensure_ascii=False,indent=2))
    print('27 quantitative figures relabeled; original source files untouched. Visio 2-3 copy prepared.')


def integrate():
    rows=json.loads((HERE/'图内标签映射.json').read_text())
    parts={}
    for row in rows:
        fig=row['figure']
        if fig=='2-3':
            parts[row['embedding']]=(HERE/'图内标签/图2-3.vsdx').read_bytes()
            parts[row['preview']]=(HERE/'图内标签/图2-3.emf').read_bytes()
        else:
            parts[row['preview']]=(HERE/f'图内标签/图{fig}.png').read_bytes()
    with ZipFile(OUT) as z:
        entries=[(info,parts.get(info.filename,z.read(info.filename))) for info in z.infolist()]
    with ZipFile(OUT,'w') as z:
        for info,data in entries:
            z.writestr(info,data)
    print('28 figure labels synchronized in the output copy.')


if __name__=='__main__':
    {'map':mapping,'labels':relabel,'integrate':integrate}[sys.argv[1]]()
