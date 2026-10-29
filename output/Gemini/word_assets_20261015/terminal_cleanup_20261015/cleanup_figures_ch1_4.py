from pathlib import Path
from copy import deepcopy
import io,json,re,shutil,os
from lxml import etree as E
from PIL import Image,ImageDraw
BASE=Path('/home/erzhu419/mine_code/Thesis')
OUT=BASE/'output/Gemini/word_assets_20261015/terminal_cleanup_20261015/figures'
OUT.mkdir(parents=True,exist_ok=True)
FONTDIR=OUT.parent/'fonts';FONTDIR.mkdir(exist_ok=True)
for name in ['msyh.ttc','msyhbd.ttc','simsun.ttc','times.ttf','timesbd.ttf','timesi.ttf','timesbi.ttf']:
 p=FONTDIR/name
 if not p.exists():p.symlink_to(Path('/mnt/c/Windows/Fonts')/name)
CONFIG=OUT.parent/'fonts.conf'
CONFIG.write_text(f'<?xml version="1.0"?><!DOCTYPE fontconfig SYSTEM "fonts.dtd"><fontconfig><include>/etc/fonts/fonts.conf</include><dir>{FONTDIR}</dir></fontconfig>')
os.environ['FONTCONFIG_FILE']=str(CONFIG)
import cairosvg
import cairocffi as cairo
NS='http://www.w3.org/2000/svg'; XL='http://www.w3.org/1999/xlink'; N={'s':NS}
D=json.loads((BASE/'output/Gemini/word_assets_20261015/figure_map_final.json').read_text())
ROWS=[r for r in D['figures'] if re.match(r'图[1-4]-',r['figure'])]
manifest=[]
def tx(e):return ''.join(e.itertext()).strip()
def setprop(e,k,v):
 s=e.get('style','');s=re.sub(r'(?<![-\w])'+re.escape(k)+r'\s*:[^;]*;?','',s)
 e.set('style',s.rstrip(';')+'; '+k+': '+str(v))
def move(g,dy):
 if abs(dy)>1e-6:g.set('transform',f'translate(0 {dy:.6f}) '+g.get('transform',''))
def bound(root,groups):
 r=E.Element('{'+NS+'}svg',nsmap=root.nsmap)
 for k in ('width','height','viewBox'):r.set(k,root.get(k))
 for e in root.xpath('./s:defs|./s:style|.//s:clipPath',namespaces=N):r.append(deepcopy(e))
 for g in groups:r.append(deepcopy(g))
 im=Image.open(io.BytesIO(cairosvg.svg2png(bytestring=E.tostring(r),dpi=144)))
 b=im.getbbox();return [v/2 for v in b] if b else [0,0,0,0]
def center(ax,oldx):
 # Matplotlib title centers are already correct; left-aligned titles use the axis rectangle.
 rects=ax.xpath('.//s:clipPath/s:rect',namespaces=N)
 if rects:
  q=max(rects,key=lambda x:float(x.get('width','0'))*float(x.get('height','0')))
  return float(q.get('x','0'))+float(q.get('width','0'))/2
 p=ax.xpath('./s:g[starts-with(@id,"patch_")]/s:path',namespaces=N)
 if p:
  nums=[float(x) for x in re.findall(r'[-+]?(?:\d*\.\d+|\d+)',p[0].get('d',''))]
  if len(nums)>=8:return (min(nums[::2])+max(nums[::2]))/2
 return oldx

def mixedminus(t):
 value=tx(t)
 if '−' not in value:return
 for c in list(t):t.remove(c)
 style=t.get('style','');fm=re.search(r'font-family:\s*([^;]+)',style);family=(fm[1].split(',')[0].strip().strip("'\"")) if fm else 'SimSun'
 size=float(re.search(r'font-size:\s*([\d.]+)',style)[1]);ctx=cairo.Context(cairo.ImageSurface(cairo.FORMAT_ARGB32,1,1))
 chunks=re.split('(−)',value);widths=[]
 for chunk in chunks:
  ctx.select_font_face('Times New Roman' if chunk=='−' else family);ctx.set_font_size(size);widths.append(ctx.text_extents(chunk)[4])
 x=float(t.get('x','0'));y=t.get('y','0')
 if 'text-anchor: middle' in style:x-=sum(widths)/2
 elif 'text-anchor: end' in style:x-=sum(widths)
 t.text=None
 for chunk,width in zip(chunks,widths):
  sp=E.SubElement(t,'{'+NS+'}tspan',x=str(x),y=y);sp.set('style',f"font-family: '{'Times New Roman' if chunk=='−' else family}'; text-anchor: start");sp.text=chunk;x+=width

def replacecolor(root,old,new):
 n=0
 for e in root.iter():
  for k,v in list(e.attrib.items()):
   if old.lower() in v.lower():e.set(k,re.sub(old,new,v,flags=re.I));n+=1
 return n

def extras(root,fig):
 changes=[]
 if fig=='图2-1':
  for old,new in [('#286a98','#174a8b'),('#27877c','#00945e')]:
   n=replacecolor(root,old,new);changes.append({'type':'color','old':old,'new':new,'elements':n})
 if fig=='图3-7':
  # Change only K=40; K=2 and the lower-panel ablation color remain as supplied.
  n=replacecolor(root,'#b2754b','#9b3f16');changes.append({'type':'color','series':'K=40','old':'#b2754b','new':'#9b3f16','elements':n})
  ax=root.xpath('.//s:g[@id="axes_1"]',namespaces=N)[0]
  for p in ax.xpath('./s:g[starts-with(@id,"line2d_")]/s:path',namespaces=N):
   if '#9b3f16' in p.get('style',''):setprop(p,'stroke-dasharray','4,2')
  for p in root.xpath('.//s:g[@id="legend_1"]//s:path',namespaces=N):
   if '#9b3f16' in p.get('style',''):setprop(p,'stroke-dasharray','4,2')
  changes.append({'type':'line_style','series':'K=40','new':'虚线，图例同步'})
 if fig=='图3-12':
  n=replacecolor(root,'#27877c','#3d3d3d');changes.append({'type':'color','series':'指数60秒','old':'#27877c','new':'#3d3d3d','elements':n})
 if fig=='图3-10':
  ax=root.xpath('.//s:g[@id="axes_1"]',namespaces=N)[0]
  defs=E.SubElement(root,'{'+NS+'}defs');p=E.SubElement(defs,'{'+NS+'}path',id='cleanup_training_repeat_diamond',d='M 0 -2.6 L 2.6 0 L 0 2.6 L -2.6 0 z')
  n=0
  for g in ax.xpath('./s:g[starts-with(@id,"PathCollection_")]',namespaces=N):
   us=g.xpath('.//s:use',namespaces=N)
   if len(us)==2:us[1].set('{'+XL+'}href','#cleanup_training_repeat_diamond');n+=1
  changes.append({'type':'marker','panel':'a','old':'每对两个空心圆','new':'每对空心圆与空心菱形','pairs':n,'coordinates':'全部原坐标保留'})
 if fig=='图3-13':
  repl={'行动者分歧项':'策略分歧项','评论家分歧项':'价值分歧项','启用减关闭':'启用 − 关闭（正值改善）','关闭减启用 / 秒':'关闭 − 启用 / 秒（正值改善）'}
  for t in root.xpath('.//s:text',namespaces=N):
   old=tx(t)
   if old in repl:
    for c in list(t):t.remove(c)
    t.text=repl[old];mixedminus(t);changes.append({'type':'text','old':old,'new':repl[old]})
  assert any(tx(t)=='快速切换' for t in root.xpath('.//s:text',namespaces=N))
  changes.append({'type':'verified_label','text':'快速切换','note':'源SVG已含完整四字，新PNG保留并确保留在画布内'})
 if fig=='图4-6':
  for t in root.xpath('.//s:text',namespaces=N):
   if re.match(r'\([bc]\)',tx(t)):
    assert '−' in tx(t);mixedminus(t)
    changes.append({'type':'minus_font','text':tx(t),'glyph':'U+2212','font':'Times New Roman'})
 if fig=='图4-9':
  ax=root.xpath('.//s:g[@id="axes_2"]',namespaces=N)[0]
  styles={'#787f87':('none','circle',0),'#286a98':('1,3','square',3),'#27877c':('7,4','triangle',6)}
  for g in ax.xpath('./s:g[starts-with(@id,"line2d_")]',namespaces=N):
   for p in g.xpath('./s:path',namespaces=N):
    col=next((c for c in styles if c in p.get('style','')),None)
    if not col:continue
    dash,shape,phase=styles[col];setprop(p,'stroke-dasharray',dash)
    nums=[float(x) for x in re.findall(r'[-+]?(?:\d*\.\d+|\d+)(?:[eE][-+]?\d+)?',p.get('d',''))]
    pts=list(zip(nums[::2],nums[1::2]));mk=E.SubElement(g,'{'+NS+'}g',id='cleanup_'+shape)
    if p.get('clip-path'):mk.set('clip-path',p.get('clip-path'))
    for x,y in pts[phase::9]:
     at={'fill':'white','stroke':col,'stroke-width':'0.8'}
     if shape=='circle':E.SubElement(mk,'{'+NS+'}circle',cx=str(x),cy=str(y),r='2',**at)
     elif shape=='square':E.SubElement(mk,'{'+NS+'}rect',x=str(x-1.8),y=str(y-1.8),width='3.6',height='3.6',**at)
     else:E.SubElement(mk,'{'+NS+'}path',d=f'M {x} {y-2.3} L {x+2.2} {y+1.7} L {x-2.2} {y+1.7} z',**at)
    changes.append({'type':'overlap_readability','series':col,'dash':dash,'marker':shape,'sample_phase':phase,'markers':len(pts[phase::9]),'data':'原路径和采样点坐标均未改变'})
  for p in ax.xpath('.//s:g[starts-with(@id,"legend_")]//s:path',namespaces=N):
   col=next((c for c in styles if c in p.get('style','')),None)
   if col:setprop(p,'stroke-dasharray',styles[col][0])
  # Display one corresponding marker on each legend segment.
  for g in ax.xpath('.//s:g[starts-with(@id,"legend_")]/s:g[starts-with(@id,"line2d_")]',namespaces=N):
   ps=g.xpath('./s:path',namespaces=N)
   if not ps:continue
   p=ps[0];col=next((c for c in styles if c in p.get('style','')),None)
   if not col:continue
   nums=[float(x) for x in re.findall(r'[-+]?(?:\d*\.\d+|\d+)',p.get('d',''))];x=(nums[0]+nums[-2])/2;y=nums[1];shape=styles[col][1]
   at={'fill':'white','stroke':col,'stroke-width':'0.8'}
   if shape=='circle':E.SubElement(g,'{'+NS+'}circle',cx=str(x),cy=str(y),r='2',**at)
   elif shape=='square':E.SubElement(g,'{'+NS+'}rect',x=str(x-1.8),y=str(y-1.8),width='3.6',height='3.6',**at)
   else:E.SubElement(g,'{'+NS+'}path',d=f'M {x} {y-2.3} L {x+2.2} {y+1.7} L {x-2.2} {y+1.7} z',**at)
 return changes

for r in ROWS:
 fig=r['figure'];src=Path(r.get('editable_svg',Path(r['source']).with_suffix('.svg')));root=E.parse(str(src)).getroot()
 original_paths={p.get('id'):p.get('d') for p in root.xpath('.//s:path[@id]',namespaces=N)}
 extra=extras(root,fig);titles=[]
 for t in root.xpath('.//s:text',namespaces=N):
  if not re.match(r'\([a-d]\)',tx(t)):continue
  g=t.getparent();ax=next((a for a in g.iterancestors() if a.get('id','').startswith('axes_')),None)
  titles.append({'text':tx(t),'el':t,'group':g,'ax':ax,'x':float(t.get('x')),'y':float(t.get('y'))})
 if not titles and not extra:
  manifest.append({'figure':fig,'status':'unchanged','png':r['source'],'svg':str(src),'labels':[],'changes':[]});continue
 for t in titles:t['group'].getparent().remove(t['group'])
 w,h=[float(x) for x in root.get('viewBox').split()[2:]]
 labels=[]
 if fig in ('图3-2','图4-2','图3-4'):
  bottoms={'图3-2':[124.0272,222.2154,339.0255],'图4-2':[108.5589,229.4217,315.2979],'图3-4':[121.4,267.6,337.3]}[fig]
  for i,t in enumerate(titles):
   t['el'].set('x',str(w/2));t['el'].set('y',str(bottoms[i]+10.5));t['el'].set('transform',f'rotate(0 {w/2} {bottoms[i]+10.5})');setprop(t['el'],'text-anchor','middle');root.append(t['group'])
   labels.append({'text':t['text'],'old':[t['x'],t['y']],'new':[w/2,bottoms[i]+10.5],'placement':'对应示意子图下方'})
  if fig in ('图3-2','图4-2'):
   ids={'图3-2':['patch_5','patch_24','patch_28'],'图4-2':['patch_5','patch_6','patch_7']}[fig]
   for i,pid in enumerate(ids):
    g=root.xpath('.//s:g[@id="'+pid+'"]',namespaces=N)[0];p=g[0];p.set('d',f'M 16.2495 {bottoms[i]+1} L 397.7505 {bottoms[i]+1} L 397.7505 {bottoms[i]+13} L 16.2495 {bottoms[i]+13} z');p.attrib.pop('clip-path',None)
    g.getparent().remove(g);root.insert(len(root)-len(titles),g)
   extra.append({'type':'header_band_position','old':'子图顶部浅色条','new':'随子图标题置于图下方，保留原色'})
  else:
   for t in root.xpath('.//s:text',namespaces=N):
    if tx(t).startswith('实线：'):move(t.getparent(),18)
   h+=18
 else:
  ys=[]
  for t in sorted(titles,key=lambda x:x['y']):
   if not ys or t['y']-ys[-1]>20:ys.append(t['y'])
  axes=root.xpath('.//s:g[starts-with(@id,"axes_")]',namespaces=N)
  # A legend below the axis label is a figure legend; place the panel title before it.
  for ax in axes:
   for lg in list(ax.xpath('./s:g[starts-with(@id,"legend_")]',namespaces=N)):
    lb=bound(root,[lg]);ax.remove(lg);ab=bound(root,[ax])
    if lb[1]>=ab[3]-2:root.append(lg)
    else:ax.append(lg)
  ainfo=[]
  for ax in axes:
   b=bound(root,[ax]);owned=[t for t in titles if t['ax'] is ax]
   row=min(range(len(ys)),key=lambda i:abs(ys[i]-owned[0]['y'])) if owned else max([i for i,y in enumerate(ys) if y<=b[1]+12] or [0])
   ainfo.append((ax,b,row))
  bottoms=[max(b[3] for ax,b,j in ainfo if j==i) for i in range(len(ys))]
  if fig=='图3-3':
   artists=root.xpath('.//s:g[starts-with(@id,"axis3d_") or starts-with(@id,"pane3d_") or starts-with(@id,"grid3d_")]',namespaces=N)
   bottoms[0]=max(bottoms[0],bound(root,artists)[3])
  for ax,b,j in ainfo:move(ax,22*j)
  # Figure-level legends are shifted past inserted title bands. Legends nested in an axis move with it.
  for lg in root.xpath('.//s:g[starts-with(@id,"legend_")]',namespaces=N):
   if any(a.get('id','').startswith('axes_') for a in lg.iterancestors()):continue
   b=bound(root,[lg]);move(lg,22*sum(v<b[1] for v in bottoms))
  for t in titles:
   i=min(range(len(ys)),key=lambda i:abs(ys[i]-t['y']))
   x=t['x'] if 'text-anchor: middle' in t['el'].get('style','') else (center(t['ax'],t['x']) if t['ax'] is not None else w/2)
   y=bottoms[i]+14+22*i;t['el'].set('x',str(x));t['el'].set('y',str(y));t['el'].set('transform',f'rotate(0 {x} {y})');setprop(t['el'],'text-anchor','middle');root.append(t['group'])
   labels.append({'text':t['text'],'old':[t['x'],t['y']],'new':[x,y],'row_content_bottom':bottoms[i]+22*i,'placement':'对应子图横轴标题下方'})
  h+=22*len(ys)
  if fig=='图3-9':extra.append({'type':'legend_title_collision','old':'顶部图例与(a)题同一高度','new':'图例保留顶部；(a)整段题移到第一子图横轴下方，二者分离'})
 root.set('height',f'{h}pt');root.set('viewBox',f'0 0 {w} {h}')
 bg=E.Element('{'+NS+'}rect',x='0',y='0',width=str(w),height=str(h),fill='white');root.insert(0,bg)
 # Annotated minus signs explicitly use a font containing the glyph.
 for t in root.xpath('.//s:text',namespaces=N):
  if '−' in tx(t):mixedminus(t)
 target=OUT/src.name;E.ElementTree(root).write(str(target),encoding='UTF-8',xml_declaration=True)
 cairosvg.svg2png(url=str(target),write_to=str(target.with_suffix('.png')),dpi=600,background_color='white')
 # Mathematical/data path definitions are kept exactly; added marker definitions are annotation-only.
 now={p.get('id'):p.get('d') for p in root.xpath('.//s:path[@id]',namespaces=N)}
 assert all(now.get(k)==v for k,v in original_paths.items()),fig
 manifest.append({'figure':fig,'status':'modified','source_png':r['source'],'source_svg':str(src),'png':str(target.with_suffix('.png')),'svg':str(target),'labels':labels,'changes':extra,'data_paths':'所有原有带ID路径定义保持不变；原曲线/误差线坐标不变','height_pt':h})
 print(fig,len(labels),'labels',len(extra),'extra',flush=True)
(OUT.parent/'manifest_ch1_4.json').write_text(json.dumps({'baseline':'重构论文_通篇校核_20261015.docx + original 重构论文.md / figure_map_final.json','figures':manifest,'modified':sum(r['status']=='modified' for r in manifest),'moved_panel_labels':sum(len(r['labels']) for r in manifest)},ensure_ascii=False,indent=2)+'\n')
for start in range(0,len(manifest),9):
 sheet=Image.new('RGB',(1500,1410),'#cccccc')
 for j,r in enumerate(manifest[start:start+9]):
  im=Image.open(r['png']).convert('RGB');im.thumbnail((480,440));tile=Image.new('RGB',(500,470),'white');tile.paste(im,((500-im.width)//2,25));ImageDraw.Draw(tile).text((10,5),r['figure'],fill='black');sheet.paste(tile,((j%3)*500,(j//3)*470))
 sheet.save(OUT.parent/f'after_sheet_{start//9+1}.png')
print('DONE',sum(r['status']=='modified' for r in manifest),'modified',sum(len(r['labels']) for r in manifest),'labels')
