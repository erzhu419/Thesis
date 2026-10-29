"""Stage native diagram copies and relabel the existing quantitative evidence."""
from pathlib import Path
from zipfile import ZipFile
from io import BytesIO
import json
import sys
from lxml import etree as E

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
sys.path.insert(0,str(ROOT/'workpapers/全文编号符号简称复核_20261011'))
from scan import W,content
R='{http://schemas.openxmlformats.org/officeDocument/2006/relationships}'
O='{urn:schemas-microsoft-com:office:office}'
V='{urn:schemas-microsoft-com:vml}'
A='{http://schemas.openxmlformats.org/drawingml/2006/main}'
OUT=ROOT/'output/排版完成_残留问题替换_20261011.docx'
BASE=HERE/'修改前_用户已完成1至13项.docx'


def stage():
    folder=HERE/'流程图'
    folder.mkdir(exist_ok=True)
    rows=[]
    with ZipFile(BASE) as z:
        body=E.fromstring(z.read('word/document.xml')).find(W+'body')
        rels={r.get('Id'):r.get('Target') for r in E.fromstring(z.read('word/_rels/document.xml.rels'))}
        for i,p in enumerate(body):
            number=content(body[i+1]).split(' ')[0] if i+1<len(body) else ''
            if number not in ['图1-1','图4-1','图5-2','图6-1','图6-15']:
                continue
            obj=p.find('.//'+W+'object')
            if obj is not None:
                embedding='word/'+rels[obj.find(O+'OLEObject').get(R+'id')]
                preview='word/'+rels[obj.find('.//'+V+'imagedata').get(R+'id')]
                (folder/(number+'_修改前.vsdx')).write_bytes(z.read(embedding))
                shape=obj.find(V+'shape')
                style=shape.get('style')
                rows.append({'figure':number,'embedding':embedding,'preview':preview,'style':style})
            else:
                blip=p.find('.//'+A+'blip')
                part='word/'+rels[blip.get(R+'embed')]
                (folder/'图6-15_修改前.png').write_bytes(z.read(part))
                rows.append({'figure':number,'preview':part})
    (HERE/'图替换关系.json').write_text(json.dumps(rows,ensure_ascii=False,indent=2))
    print(json.dumps(rows,ensure_ascii=False))


def plot():
    sys.path.insert(0,str(ROOT/'scripts'))
    import mcwm_confirmation_figures as c
    b=c.b
    fig,axes=b.plt.subplots(3,1,figsize=(7.25,5.75),sharex=True)
    # Keep the original numerical axes area; add room only for expanded labels.
    fig.subplots_adjust(left=2.535/7.25,right=(7.25-.08625)/7.25,bottom=.15,top=.94,hspace=.62)
    labels=['MC-WM\n（模型损失系数0.25）','MC-WM\n（模型损失系数0.30）','按源偏差调节\n模型损失系数']
    before=[]
    for panel,(ax,scene,label) in enumerate(zip(axes,c.SCENES,['正常标定','联合50%源参数偏差','联合75%源参数偏差'])):
        for y,(arm,_,color) in enumerate(c.METHODS):
            row=c.comparison(scene,arm,'h2o_matched')
            c.draw_estimate(ax,row,y,color)
            # Place the count in the reserved right margin, without changing any data.
            ax.text(123,y,f"{row['better_seeds']}/20次训练改善",ha='right',va='center',fontsize=7.5,color=color)
            before.append({'scene':scene,'arm':arm,'points':[r['cost_seconds'] for r in row['seed_deltas']],
                           'estimate':row['cost_delta_seconds'],'better':row['better_seeds']})
        ax.axvline(0,color=b.GRAY,lw=.8,ls='--');ax.grid(axis='x')
        ax.set(yticks=range(3),yticklabels=labels,ylim=(2.45,-.45),xlim=(-170,130),xticks=[-150,-100,-50,0,50,100])
        ax.set_title(f'({chr(97+panel)}) {label}',loc='left',pad=9)
    axes[-1].set_xlabel('相对于无世界模型混合训练基线的\n人均广义时间成本差（秒）',fontsize=8.5)
    for ax in axes:
        for line in ax.get_xgridlines()+ax.get_ygridlines():
            line.set_color('#DCE1E5');line.set_linewidth(.45);line.set_alpha(.75)
    for suffix in ['png','pdf']:
        fig.savefig(HERE/'流程图'/('图6-15_校订.'+suffix),dpi=300,bbox_inches='tight')
    b.plt.close(fig)
    (HERE/'图6-15数据保留.json').write_text(json.dumps(before,ensure_ascii=False,indent=2))
    print('Relabeled 180 original seed differences and nine original confidence intervals.')


def integrate():
    rows=json.loads((HERE/'图替换关系.json').read_text())
    parts={}
    for row in rows:
        name=row['figure']
        if name=='图6-15':
            parts[row['preview']]=(HERE/'流程图'/(name+'_校订.png')).read_bytes()
    with ZipFile(OUT) as z:
        from PIL import Image
        sys.path.insert(0,str(ROOT/'workpapers/七章重写稿整合_20261010'))
        from integrate import body_spans
        raw=z.read('word/document.xml')
        body=E.fromstring(raw).find(W+'body')
        spans=body_spans(raw)
        for i,p in enumerate(body):
            if i+1<len(body) and content(body[i+1]).startswith('图6-15 '):
                width,height=Image.open(HERE/'流程图/图6-15_校订.png').size
                extent=p.find('.//{http://schemas.openxmlformats.org/drawingml/2006/wordprocessingDrawing}extent')
                cy=str(round(int(extent.get('cx'))*height/width))
                extent.set('cy',cy)
                for e in p.findall('.//'+A+'xfrm/'+A+'ext'):
                    e.set('cy',cy)
                start,end=spans[i]
                parts['word/document.xml']=raw[:start]+E.tostring(p)+raw[end:]
        entries=[(info,parts.get(info.filename,z.read(info.filename))) for info in z.infolist()]
    with ZipFile(OUT,'w') as z:
        for info,data in entries:
            z.writestr(info,data)
    print('Only numerical figure 6-15 relabeled; all native Visio objects retained unchanged.')


if __name__=='__main__':
    {'stage':stage,'plot':plot,'integrate':integrate}[sys.argv[1]]()
