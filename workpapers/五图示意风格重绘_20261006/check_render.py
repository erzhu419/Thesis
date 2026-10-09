from pathlib import Path
import json, re
import fitz

root=Path(__file__).resolve().parents[2];work=Path(__file__).resolve().parent
old=fitz.open(root/'排版完成_分页图表修复_20261006.pdf')
new=fitz.open(root/'排版完成_五图公交示意版_20261006.pdf')
norm=lambda text:re.sub(r'\s+','',text)
assert len(old)==len(new)==318, (len(old),len(new))
captions={
    '2-1':'图2-1 事件触发的公交运行过程控制闭环',
    '2-3':'图2-3 跨保真代理验证与真实运营应用的数据角色',
    '4-1':'图4-1 CS-BAPR的策略结构与训练稳定化机制',
    '5-2':'图5-2 公交H2O+的跨保真训练与完成保障确认流程',
    '6-1':'图6-1 校准选择驱动的世界模型适应与策略更新'}
figure21_page=next(i for i,p in enumerate(old) if norm(captions['2-1']) in norm(p.get_text()))
inset=json.loads((work/'交付记录.json').read_text())['figure_2_1_vertical_inset_percent']/100
text_changes=[];image_position_changes=[]
for i,(a,b) in enumerate(zip(old,new)):
    if norm(a.get_text())!=norm(b.get_text()):text_changes.append(i+1)
    aa=[x['bbox'] for x in a.get_image_info()];bb=[x['bbox'] for x in b.get_image_info()]
    if i==figure21_page:
        assert len(aa)==1
        x0,y0,x1,y1=aa[0];delta=(y1-y0)*inset
        aa=[(x0,y0+delta,x1,y1-delta)]
    if len(aa)!=len(bb) or any(abs(x-y)>0.2 for ar,br in zip(aa,bb) for x,y in zip(ar,br)):
        image_position_changes.append(i+1)
assert not text_changes,('PDF text pagination changed',text_changes)
assert not image_position_changes,('PDF image positions changed',image_position_changes)
renders=work/'排版预览';renders.mkdir(exist_ok=True)
pages={}
for number in ['2-1','2-3','4-1','5-2','6-1']:
    hits=[i for i,p in enumerate(new) if norm(captions[number]) in norm(p.get_text())]
    assert len(hits)==1,(number,hits)
    i=hits[0];pages[number]=i+1
    new[i].get_pixmap(matrix=fitz.Matrix(1.8,1.8),alpha=False).save(renders/f'图{number}_PDF第{i+1}页.png')
result={'pages':len(new),'page_text_and_pagination_unchanged':True,
        'all_other_image_positions_unchanged':True,'figure_2_1_aspect_fit_verified':True,'figure_pages':pages}
(work/'页面核对.json').write_text(json.dumps(result,ensure_ascii=False,indent=2))
print(json.dumps(result,ensure_ascii=False,indent=2))
