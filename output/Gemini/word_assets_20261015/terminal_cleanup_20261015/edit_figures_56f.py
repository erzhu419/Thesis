"""Authorized SVG layout edits and original-pixel panel crops; no generative tools."""
from pathlib import Path
from zipfile import ZipFile
from lxml import etree as E
from PIL import Image
from collections import Counter
import re, json, os, base64, io, copy

ROOT = Path('/home/erzhu419/mine_code/Thesis')
A = ROOT/'output/Gemini/word_assets_20261015'
OUT = A/'terminal_cleanup_20261015/figures'
OUT.mkdir(parents=True, exist_ok=True)
FONTS = OUT/'fonts'; FONTS.mkdir(exist_ok=True)
for name in ['simsun.ttc','times.ttf','timesbd.ttf','timesi.ttf','timesbi.ttf','msyh.ttc','msyhbd.ttc']:
    p = FONTS/name
    if not p.exists(): p.symlink_to(Path('/mnt/c/Windows/Fonts')/name)
config = OUT/'fonts.conf'
config.write_text(f'<?xml version="1.0"?><!DOCTYPE fontconfig SYSTEM "fonts.dtd"><fontconfig><include>/etc/fonts/fonts.conf</include><dir>{FONTS}</dir></fontconfig>')
os.environ['FONTCONFIG_FILE'] = str(config)
import cairosvg

S='http://www.w3.org/2000/svg'; X='http://www.w3.org/1999/xlink'; NS={'s':S}
BASE=Path('/mnt/d/Paper/Thesis/Gemini/重构论文_通篇校核_20261015.docx')
MAP={f['figure']:f for f in json.loads((A/'figure_map_final.json').read_text())['figures']}
MEDIA={'图5-3':30,'图5-4':31,'图5-6':33,'图5-7':34,'图5-8':35,'图5-10':37,'图5-12':39,
       '图6-1':40,'图6-2':41,'图6-11':50,'图6-12':51,'图6-13':52,'图6-14':53,'图6-15':54,'图6-16':55,'图6-17':56,
       '图F-1':58,'图F-2':59,'图F-3':60}
MANIFEST=[]
def val(t): return ''.join(t.itertext()).strip()
def texts(r): return list(r.iter('{'+S+'}text'))
def text_find(r,s):
    found=[t for t in texts(r) if val(t)==s]
    assert len(found)==1,(s,len(found))
    return found[0]
def group(t):
    p=t.getparent()
    return p if p.tag=='{'+S+'}g' and (p.get('id')or'').startswith('text_') else t
def number(x):return float(re.search(r'-?\d+(?:\.\d+)?',x)[0])
def axes(r):return r.xpath('.//s:g[starts-with(@id,"axes_")]',namespaces=NS)
def bbox(a):
    p=a.find('s:g/s:path',NS)
    nums=list(map(float,re.findall(r'-?\d+(?:\.\d+)?',p.get('d'))))
    return min(nums[0::2]),min(nums[1::2]),max(nums[0::2]),max(nums[1::2])
def remove(e):e.getparent().remove(e)
def replace_text(t,s):
    assert len(t)==0, val(t)
    t.text=s
def move_caption(t,x,y,newtext=None,center=True):
    if newtext is not None:replace_text(t,newtext)
    t.set('x',str(x));t.set('y',str(y));t.attrib.pop('transform',None)
    if center:
        st=t.get('style','')
        st=re.sub(r'text-anchor:\s*[^;]+','text-anchor: middle',st)
        if 'text-anchor:' not in st:st+='; text-anchor: middle'
        t.set('style',st)
    return y
def below_x(a):
    xaxis=a.xpath('./s:g[starts-with(@id,"matplotlib.axis_")]',namespaces=NS)
    ys=[]
    for t in texts(xaxis[0]) if xaxis else []:
        if t.get('y'):ys.append(float(t.get('y')))
        else:
            m=re.search(r'translate\(\s*([-\d.]+)[ ,]+([-\d.]+)',t.get('transform',''))
            if m:ys.append(float(m.group(2)))
    return max(ys+[bbox(a)[3]])+18
def canvas(r,w=None,h=None):
    v=list(map(float,r.get('viewBox').split()));w=w or v[2];h=h or v[3]
    r.set('viewBox',f'0 0 {w:g} {h:g}');r.set('width',f'{w:g}pt');r.set('height',f'{h:g}pt')
    bg=E.Element('{'+S+'}rect',x='0',y='0',width=str(w),height=str(h),fill='white')
    r.insert(0,bg)
def get_source(fig):
    if fig=='图F-1':return ROOT/'manuscript/figures/fig6_world_model_pilot.svg'
    if fig=='图F-3':return ROOT/'manuscript/figures/fig6_transition_mismatch.svg'
    f=MAP[fig]
    return Path(f.get('editable_svg') or str(Path(f['source']).with_suffix('.svg')))
def native_titles(r):return [t for t in texts(r) if re.match(r'^\([a-z]\)',val(t))]
def record(fig,source,r,actions,kept=None,removed=None,note=None):
    name=fig.replace('图','fig').replace('-','_')
    p=OUT/(name+'.svg')
    p.write_bytes(E.tostring(r,encoding='UTF-8',xml_declaration=True))
    cairosvg.svg2png(url=str(p),write_to=str(p.with_suffix('.png')),dpi=600)
    im=Image.open(p.with_suffix('.png'))
    MANIFEST.append({'figure':fig,'word_media':f'word/media/image{MEDIA[fig]}.png','source':str(source),
                     'svg':str(p),'png':str(p.with_suffix('.png')),'pixel_size':list(im.size),
                     'actions':actions,'kept_panels':kept,'removed_panels':removed,'note':note,
                     'numbering':'旧图号；原6-11..6-17的新编号由根代理统一。'})

# Ordinary plots: move the original full caption below the horizontal-axis title.
for fig in ['图5-6','图5-10','图5-12','图6-11','图6-12','图6-13','图6-14','图6-15','图6-16','图6-17']:
    src=get_source(fig);r=E.parse(str(src)).getroot();last=0;kept=[]
    for a in axes(r):
        ts=native_titles(a)
        if not ts:continue
        assert len(ts)==1,(fig,len(ts))
        t=ts[0];x0,y0,x1,y1=bbox(a);y=below_x(a)
        last=max(last,move_caption(t,(x0+x1)/2,y));kept.append(val(t))
    canvas(r,h=max(float(r.get('viewBox').split()[3]),last+13))
    record(fig,src,r,['原子图字母和原中文题完整移至该横轴标题下方；保留曲线、误差条、数值、字体与配色。'],kept)

# Four previously unlettered calibration panels.
fig='图5-7';src=get_source(fig);r=E.parse(str(src)).getroot();last=0;kept=[]
for k,a in enumerate(axes(r)):
    title=['广义乘客时间','候车时间','车内时间','车辆行程时长'][k]
    t=text_find(a,title);x0,y0,x1,y1=bbox(a);new=f'({chr(97+k)}) {title}'
    last=max(last,move_caption(t,(x0+x1)/2,below_x(a),new));kept.append(new)
canvas(r,h=max(float(r.get('viewBox').split()[3]),last+13))
record(fig,src,r,['确认对应Word image34；补齐(a)–(d)，原中文子图题下移。'],kept)

# Sampling schematic and two lower plots.
fig='图5-3';src=get_source(fig);r=E.parse(str(src)).getroot();aa=axes(r);last=0;kept=[]
for k,a in enumerate(aa):
    t=native_titles(a)[0]
    if k==0:y=85;x=256
    else:
        x0,y0,x1,y1=bbox(a);x=(x0+x1)/2;y=below_x(a)
    last=max(last,move_caption(t,x,y));kept.append(val(t))
canvas(r,h=max(306,last+13))
record(fig,src,r,['(a)题置上方示意流程下；(b)(c)题置各横轴标题下；原数据路径保持。'],kept)

# Intervention-response panel: remove completion rate, preserve the original cost panel.
fig='图5-8';src=get_source(fig);r=E.parse(str(src)).getroot();aa=axes(r);a=aa[0];remove(aa[1]);x0,y0,x1,y1=bbox(a)
t=native_titles(a)[0];last=move_caption(t,(x0+x1)/2,below_x(a))
oldw=float(r.get('viewBox').split()[2]);neww=x1+9
for legend in r.xpath('.//s:g[starts-with(@id,"legend_")]',namespaces=NS):
    if legend.getparent() is not a:legend.set('transform',f'translate({neww/2-oldw/2:g},0)')
canvas(r,w=neww,h=max(float(r.get('viewBox').split()[3]),last+13))
record(fig,src,r,['删除(b)完成率响应差，保留(a)广义时间响应差；三档需求图例保留并居中；题置横轴标题下。'],[val(t)],['(b) 完成率响应差'])

# H2O+ workflow: remove the execution guards explicitly rejected by the user.
fig='图5-4';src=get_source(fig);r=E.parse(str(src)).getroot()
for ident in ['patch_5','patch_6','patch_12','patch_18','patch_19','patch_26']:
    q=r.xpath(f'.//s:g[@id="{ident}"]',namespaces=NS)
    if q:remove(q[0])
remove(group(text_find(r,'低载尾部完成保障')))
replace_text(text_find(r,'动作幅度校准'),'策略输出')
a=axes(r)[0]
arrow=E.SubElement(a,'{'+S+'}g',id='terminal_direct_execution_arrow')
E.SubElement(arrow,'{'+S+'}path',d='M 128.37 335.52 L 286.47 335.52',style='fill: none; stroke: #787f87; stroke-width: 1; stroke-linecap: round')
E.SubElement(arrow,'{'+S+'}path',d='M 282.47 333.52 L 286.47 335.52 L 282.47 337.52 z',style='fill: #787f87; stroke: #787f87; stroke-linejoin: miter')
ys=[178.5,290.5,369.5];kept=[]
for t,y in zip(native_titles(r),ys):move_caption(t,207,y);kept.append(val(t))
canvas(r,h=383)
record(fig,src,r,['原(a)(b)(c)题移至各框图区下方。','删除低载尾部完成保障及动作幅度校准执行链；策略输出直接连接SUMO闭环评价。'],kept, ['低载尾部完成保障','动作幅度校准'])

# World-model architecture: clarify the three-way split, preserving all algorithm boxes.
fig='图6-2';src=get_source(fig);r=E.parse(str(src)).getroot()
replace_text(text_find(r,'拟合、选择、校准'),'拟合、早停、选择')
for q in list(r.xpath('.//s:path[contains(@style,"opacity: 0.1")]',namespaces=NS)):
    if '#286a98' in q.get('style','') or '#27877c' in q.get('style',''):remove(q.getparent())
tt=native_titles(r);move_caption(tt[0],21.7557,179,center=False);move_caption(tt[1],207,380)
canvas(r,h=393)
record(fig,src,r,['(a)题移至上框图下方左侧避开连接箭头；(b)题移至评论家更新框下方。','三份数据划分改为拟合、早停、选择。'],[val(t)for t in tt])

# Chapter roadmap: only delete the explicitly removed controlled-dynamics ending.
fig='图6-1';src=get_source(fig);r=E.parse(str(src)).getroot()
t=text_find(r,'多档源标定偏差下的SUMO多线路场景与受控动力学平台')
replace_text(t,'多档源标定偏差下的SUMO多线路场景')
for t in texts(r):
    # Cairo selects only the first named font. Use the actual Chinese face.
    # A single text run also preserves the source's whole-line centering.
    t.set('style',t.get('style','').replace("'Times New Roman', 'SimSun'","'SimSun'"))
record(fig,src,r,['末端删除“与受控动力学平台”，其余框线、中文字及位置保持。'])

# Appendix F-1: original (a),(c) cost panels become adjacent (a),(b).
fig='图F-1';src=get_source(fig);r=E.parse(str(src)).getroot();aa=axes(r)
for i in [1,3]:remove(aa[i])
for t in texts(r):
    if val(t)=='人均广义时间 / 秒':replace_text(t,'人均广义时间成本 / 秒')
kept=[];last=0
for k,a in enumerate([aa[0],aa[2]]):
    t=native_titles(a)[0];x0,y0,x1,y1=bbox(a);cap=f'({chr(97+k)}) '+re.sub(r'^\([a-z]\)\s*','',val(t))
    y=below_x(a);move_caption(t,(x0+x1)/2,y,cap)
    if k:a.set('transform','translate(208.02332,-147.6)');y-=147.6
    last=max(last,y);kept.append(cap)
canvas(r,h=last+13)
record(fig,src,r,['删除原(b)(d)行程完成率；原(a)需求1倍与(c)需求1.25倍成本面板并排，连续标(a)(b)，标题置横轴下。','保留Word中的“人均广义时间成本”完整纵轴题，不采用较旧SVG省略的“成本”。'],kept,['(b) 需求1倍完成率','(d) 需求1.25倍完成率'])

# Appendix F-3: cost heatmap and its own color scale only.
fig='图F-3';src=get_source(fig);r=E.parse(str(src)).getroot();aa=axes(r)
remove(aa[1]);remove(aa[3]);t=native_titles(aa[0])[0]
last=move_caption(t,147.235067,below_x(aa[2]),'(a) 人均广义时间成本降幅')
canvas(r,w=239,h=max(295.2,last+13))
record(fig,src,r,['删除完成率变化热力图及其色标；原成本热力图数值/颜色保留，标题按Word原中文移至其色标横轴标题下。'],[val(t)],['(b) 完成率变化'])

# Appendix F-2 has no matching native SVG: preserve original pixels for all retained data.
fig='图F-2'
with ZipFile(BASE)as z:im=Image.open(io.BytesIO(z.read('word/media/image59.png'))).convert('RGB')
assert im.size==(4320,3305)
crop_a=im.crop((0,140,2200,1500));crop_c=im.crop((0,1920,4320,3305))
r=E.Element('{'+S+'}svg',nsmap={None:S,'xlink':X},width='518.4pt',height='393.6pt',viewBox='0 0 4320 3280',version='1.1')
E.SubElement(r,'{'+S+'}rect',x='0',y='0',width='4320',height='3280',fill='white')
for ident,crop,x,y in [('retained_cost_panel',crop_a,1060,0),('retained_paired_cost_panel',crop_c,0,1630)]:
    buf=io.BytesIO();crop.save(buf,format='PNG')
    e=E.SubElement(r,'{'+S+'}image',id=ident,x=str(x),y=str(y),width=str(crop.width),height=str(crop.height))
    e.set('{'+X+'}href','data:image/png;base64,'+base64.b64encode(buf.getvalue()).decode())
for text,y in [('(a) 乘客时间',1490),('(b) 同一训练种子下的配对差异',3210)]:
    e=E.SubElement(r,'{'+S+'}text',x='2160',y=str(y),style="font-family: 'SimSun'; font-size: 96px; text-anchor: middle; fill: #000000")
    e.text=text
record(fig,'Word原media/image59.png（原始像素）',r,['删除原(b)服务完成情况；保留(a)成本和(c)配对成本图的原始像素，重排(a)(b)，将原中文题置下方。'],['(a) 乘客时间','(b) 同一训练种子下的配对差异'],['(b) 服务完成情况'],note='SVG内嵌两个无重采样的原PNG裁切面板；新增子图题为可编辑矢量文字。')

assert len(MANIFEST)==19,len(MANIFEST)
MANIFEST.sort(key=lambda f:(0 if f['figure'].startswith('图5')else 1 if f['figure'].startswith('图6')else 2,int(f['figure'].split('-')[1])))
(OUT/'manifest_56F.json').write_text(json.dumps({'baseline':str(BASE),'figures':MANIFEST,'unchanged_verified':[{'figure':'图6-18','word_media':'word/media/image57.png','decision':'保留原图-88.39；原记录-88.38702208764929正确四舍五入。正文88.38由根代理单值同步。','source':str(ROOT/'output/第6章_删除完成率重绘_20261014/作图数据.csv')}],'deleted_figures_skipped':['图6-'+str(i)for i in range(3,10)]},ensure_ascii=False,indent=2)+'\n')
print(json.dumps({'figures':len(MANIFEST),'manifest':str(OUT/'manifest_56F.json')},ensure_ascii=False))
