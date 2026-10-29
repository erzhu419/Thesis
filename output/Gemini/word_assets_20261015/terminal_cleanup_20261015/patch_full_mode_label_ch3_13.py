from pathlib import Path
from copy import deepcopy
from collections import Counter
import json,os
from lxml import etree as E
from PIL import Image
B=Path('/home/erzhu419/mine_code/Thesis/output/Gemini/word_assets_20261015/terminal_cleanup_20261015')
os.environ['FONTCONFIG_FILE']=str(B/'fonts.conf')
import cairosvg
N={'s':'http://www.w3.org/2000/svg'};NS=N['s']
src=B/'figures/fig3_disagreement_factorial.svg';root=E.parse(str(src)).getroot();before=Counter(p.get('d') for p in root.xpath('.//s:path',namespaces=N))
ax1=root.xpath('.//s:g[@id="axes_1"]',namespaces=N)[0];ax2=root.xpath('.//s:g[@id="axes_2"]',namespaces=N)[0]
labels=[t for t in ax1.xpath('.//s:text',namespaces=N) if ''.join(t.itertext()).strip() in ['正常运行','客流激增','快速切换']]
assert len(labels)==3
for i,t in enumerate(labels):
 if ''.join(t.itertext()).strip()=='快速切换':t.text='运行模式快速切换'
 c=deepcopy(t);c.set('x','260.659832');c.set('transform',f'rotate(0 260.659832 {c.get("y")})')
 g=E.SubElement(ax2,'{'+NS+'}g',id='cleanup_panel_b_condition_'+str(i));g.append(c)
def move(g,dx):g.set('transform',f'translate({dx} 0) '+g.get('transform',''))
move(ax1,14);move(ax2,54)
for g in root.xpath('.//s:g[starts-with(@id,"legend_")]',namespaces=N):
 if not any(a.get('id','').startswith('axes_') for a in g.iterancestors()):move(g,34)
for t in root.xpath('./s:g/s:text',namespaces=N):
 value=''.join(t.itertext()).strip()
 if value.startswith('(a)'):move(t.getparent(),14)
 elif value.startswith('(b)'):move(t.getparent(),54)
w=468;h=float(root.get('viewBox').split()[3]);root.set('width',f'{w}pt');root.set('viewBox',f'0 0 {w} {h}')
root.xpath('./s:rect',namespaces=N)[0].set('width',str(w))
assert before==Counter(p.get('d') for p in root.xpath('.//s:path',namespaces=N))
target=src.with_name('fig3_disagreement_factorial_full_mode.svg');E.ElementTree(root).write(str(target),encoding='UTF-8',xml_declaration=True)
cairosvg.svg2png(url=str(target),write_to=str(target.with_suffix('.png')),dpi=600,background_color='white')
p=B/'manifest_ch1_4.json';d=json.loads(p.read_text());r=next(x for x in d['figures'] if x['figure']=='图3-13');r['png']=str(target.with_suffix('.png'));r['svg']=str(target);r['width_pt']=w
for t in r['labels']:t['new'][0]+=14 if t['text'].startswith('(a)') else 54
r['changes']=[x for x in r['changes'] if x.get('type')!='verified_label']
r['changes'].extend([{'type':'text','old':'快速切换','new':'运行模式快速切换','panels':['a','b']},{'type':'layout','left_margin_added_pt':14,'inter_panel_gap_added_pt':40,'panel_b_labels':'补齐正常运行、客流激增、运行模式快速切换，与(a)保持对应','data_paths':'原路径和数据坐标完全未变，仅平移完整面板以增加留白'}])
p.write_text(json.dumps(d,ensure_ascii=False,indent=2)+'\n')
q=B/'qa_ch1_4.json';d=json.loads(q.read_text());d['fig3_13_full_label']={'label':'运行模式快速切换','panels':['a','b'],'all_path_coordinates_unchanged':True,'left_margin_added_pt':14,'inter_panel_gap_added_pt':40};q.write_text(json.dumps(d,ensure_ascii=False,indent=2)+'\n')
im=Image.open(target.with_suffix('.png'));im.thumbnail((1800,1200));im.save(B/'fig3_13_full_mode_preview.png')
print(target)
