"""Read-only, complete text and native formatting comparison of two thesis DOCX."""
from pathlib import Path
from zipfile import ZipFile
from lxml import etree as E
from difflib import SequenceMatcher
from functools import lru_cache
from collections import Counter
import json, re, unicodedata

A = Path(__file__).resolve().parent
BASE = Path('/mnt/d/Paper/Thesis/Gemini/重构论文_导言小结同步_20261018.docx')
OTHER = A.parents[1] / '调整完格式的.docx'
W = '{http://schemas.openxmlformats.org/wordprocessingml/2006/main}'
M = '{http://schemas.openxmlformats.org/officeDocument/2006/math}'
NS = {'w': W[1:-1], 'm': M[1:-1]}
def tx(e):
    return ''.join(n.text or '' for n in e.iter() if n.tag in (W+'t', M+'t'))
def norm(s):
    return ''.join(c for c in s if not c.isspace() and c != '\u200b')
def local(tag):
    q = E.QName(tag)
    return ('m:' if q.namespace == M[1:-1] else 'w:' if q.namespace == W[1:-1] else '') + q.localname

class Doc:
    def __init__(self, path):
        self.path = path
        with ZipFile(path) as z:
            self.parts = {n: z.read(n) for n in z.namelist()}
        self.root = E.fromstring(self.parts['word/document.xml'])
        self.styles_root = E.fromstring(self.parts['word/styles.xml'])
        self.styles = {e.get(W+'styleId'): e for e in self.styles_root.findall(W+'style')}
        self.names = {k: (e.find(W+'name').get(W+'val') if e.find(W+'name') is not None else k) for k,e in self.styles.items()}
        self.blocks = list(self.root.find(W+'body'))
        self.headings, scope = {}, '前置页'
        for i,e in enumerate(self.blocks):
            s = tx(e); st = self.style(e)
            if not st.lower().startswith('toc') and e.tag == W+'p' and (re.match(r'^(第\d+章|附录[A-G]|\d+\.\d+(?:\.\d+)?\s)', s) or st in ('chapter','h2','h3','h4','ae','aeb','1','2','3')) and len(s)<160:
                scope = s
            if s.strip() in ('参考文献','攻读学位期间主要的研究成果','致  谢'):
                scope = s
            self.headings[i] = scope
        self.flow = [i for i,e in enumerate(self.blocks) if norm(tx(e)) or list(e.iter(W+'drawing')) or e.tag in (W+'tbl',W+'sectPr')]
    def style(self,e):
        st=e.find(W+'pPr/'+W+'pStyle')
        return st.get(W+'val','') if st is not None else ''
    def label(self,e):
        st=self.style(e);s=tx(e)
        if st.lower().startswith('toc'):
            return ('toc',re.sub(r'\d+$','',norm(s)))
        if e.tag == W+'tbl':
            cells=e.findall('.//'+W+'tr/'+W+'tc')
            last=tx(cells[-1]) if cells else ''
            if re.fullmatch(r'\s*[（(][1-7A-G]-\d+[a-z]*[）)]\s*',last):
                return ('eq',norm(last))
        if list(e.iter(W+'drawing')) and not norm(s):
            # Adjacent figure captions anchor individual image blocks.
            return ('drawing','')
        return (e.tag,norm(s))

def props(el,doc):
    result={}
    if el is None:return result
    def visit(e,path):
        if not isinstance(e.tag,str):return
        name=local(e.tag);path=path+'/'+name
        attrs={local(k):v for k,v in e.attrib.items() if not E.QName(k).localname.startswith('rsid')}
        for k,v in attrs.items():
            if e.tag in (W+'pStyle',W+'rStyle',W+'basedOn',W+'next',W+'link') and k=='w:val':
                v=doc.names.get(v,v).lower()
            result[path+'/@'+k]=v
        if (e.text or '').strip():result[path+'/#text']=e.text
        if not attrs and not len(e):result[path]='present'
        for child in e:visit(child,path)
    visit(el,'')
    return result

def changes(a,b):
    return [{'property':k,'before':a.get(k),'after':b.get(k)} for k in sorted(set(a)|set(b)) if a.get(k)!=b.get(k)]

left,right=Doc(BASE),Doc(OTHER)
def align(xs,ys,da,db):
    pairs=[];groups=[]
    for mode,a,b,c,d in SequenceMatcher(None,[da.label(e) for e in xs],[db.label(e) for e in ys],autojunk=False).get_opcodes():
        if mode=='equal':pairs.extend(zip(range(a,b),range(c,d)))
        else:
            groups.append((mode,a,b,c,d))
            @lru_cache(None)
            def dp(i,j):
                if i==b or j==d:return 0,()
                best=max(dp(i+1,j),dp(i,j+1),key=lambda p:p[0])
                if xs[i].tag==ys[j].tag:
                    ratio=SequenceMatcher(None,norm(tx(xs[i])),norm(tx(ys[j])),autojunk=False).ratio()
                    if ratio>=.35:
                        score,tail=dp(i+1,j+1)
                        if score+ratio>best[0]:best=(score+ratio,((i,j),)+tail)
                return best
            pairs.extend(dp(a,c)[1])
    return sorted(pairs),groups

xs=[left.blocks[i] for i in left.flow];ys=[right.blocks[i] for i in right.flow]
pairs,groups=align(xs,ys,left,right)
content=[]
for number,(mode,a,b,c,d) in enumerate(groups,1):
    old='\n\n'.join(tx(e) for e in xs[a:b]);new='\n\n'.join(tx(e) for e in ys[c:d])
    toc=all(left.style(e).lower().startswith('toc') for e in xs[a:b]) and all(right.style(e).lower().startswith('toc') for e in ys[c:d])
    kind='目录页码' if toc else '空白或段落结构' if norm(old)==norm(new) else '新增内容' if a==b else '删除内容' if c==d else '文字或符号变化'
    content.append({'id':number,'kind':kind,'operation':mode,'location':left.headings[left.flow[a]] if a<len(xs) else right.headings[right.flow[c]],'old_body_blocks':[left.flow[i]+1 for i in range(a,b)],'new_body_blocks':[right.flow[i]+1 for i in range(c,d)],'before':old,'after':new,'fragments':[{'op':op,'before':old[i:j],'after':new[k:l]} for op,i,j,k,l in SequenceMatcher(None,old,new,autojunk=False).get_opcodes() if op!='equal']})
# Exact-key matches can still differ in spaces, or in cached TOC numbers.
for i,j in pairs:
    a,b=xs[i],ys[j]
    if left.label(a)==right.label(b) and tx(a)!=tx(b):
        old,new=tx(a),tx(b)
        content.append({'id':len(content)+1,'kind':'目录页码' if left.style(a).lower().startswith('toc') else '空白或段落结构' if norm(old)==norm(new) else '文字或符号变化','operation':'replace','location':left.headings[left.flow[i]],'old_body_blocks':[left.flow[i]+1],'new_body_blocks':[right.flow[j]+1],'before':old,'after':new,'fragments':[{'op':op,'before':old[x:y],'after':new[k:l]} for op,x,y,k,l in SequenceMatcher(None,old,new,autojunk=False).get_opcodes() if op!='equal']})

paragraphs=[];runs=[];tables=[]
def rsegments(p,doc):
    segments=[];pos=0
    for e in p.iter():
        if e.tag not in (W+'t',M+'t'):continue
        value=e.text or '';run=e.getparent()
        if run.tag not in (W+'r',M+'r'):continue
        pr=props(run.find(W+'rPr'),doc)
        # Mathematical controls are covered independently by the formula audit.
        segments.append((pos,pos+len(value),pr));pos+=len(value)
    return segments
def compare_p(a,b,location,oldid,newid):
    ap,bp=props(a.find(W+'pPr'),left),props(b.find(W+'pPr'),right)
    diff=changes(ap,bp)
    if diff:paragraphs.append({'location':location,'old_address':oldid,'new_address':newid,'text_before':tx(a),'text_after':tx(b),'changes':diff})
    old,new=tx(a),tx(b)
    if not old or not new:return
    aseg,bseg=rsegments(a,left),rsegments(b,right)
    for mode,x,y,k,l in SequenceMatcher(None,old,new,autojunk=False).get_opcodes():
        if mode!='equal':continue
        boundaries={x,y}
        for a0,a1,_ in aseg:
            if x<a0<y:boundaries.add(a0)
            if x<a1<y:boundaries.add(a1)
        for b0,b1,_ in bseg:
            q=b0-k+x;r=b1-k+x
            if x<q<y:boundaries.add(q)
            if x<r<y:boundaries.add(r)
        ordered=sorted(boundaries)
        for start,end in zip(ordered,ordered[1:]):
            bp0=k+start-x
            af=next((v for lo,hi,v in aseg if lo<=start<hi),{})
            bf=next((v for lo,hi,v in bseg if lo<=bp0<hi),{})
            diff=changes(af,bf)
            if diff:
                row={'location':location,'old_address':oldid,'new_address':newid,'old_span':[start,end],'new_span':[bp0,bp0+end-start],'text':old[start:end],'changes':diff}
                if runs and all(runs[-1].get(q)==row.get(q) for q in ('old_address','new_address','changes')) and runs[-1]['old_span'][1]==start and runs[-1]['new_span'][1]==bp0:
                    runs[-1]['old_span'][1]=end;runs[-1]['new_span'][1]=bp0+end-start;runs[-1]['text']+=old[start:end]
                else:runs.append(row)

for i,j in pairs:
    a,b=xs[i],ys[j];ai,bi=left.flow[i],right.flow[j];loc=left.headings[ai]
    if a.tag==W+'p':compare_p(a,b,loc,f'正文块{ai+1}',f'正文块{bi+1}')
    elif a.tag==W+'tbl':
        oldprops={};newprops={}
        for e in a.iter():
            if e.tag in (W+'tblPr',W+'tblGrid',W+'trPr',W+'tcPr'):oldprops[str(len(oldprops))]=props(e,left)
        for e in b.iter():
            if e.tag in (W+'tblPr',W+'tblGrid',W+'trPr',W+'tcPr'):newprops[str(len(newprops))]=props(e,right)
        if oldprops!=newprops:tables.append({'location':loc,'old_address':f'正文块{ai+1}','new_address':f'正文块{bi+1}','before':oldprops,'after':newprops})
        pa=list(a.iter(W+'p'));pb=list(b.iter(W+'p'));pp,_=align(pa,pb,left,right)
        for x,y in pp:compare_p(pa[x],pb[y],loc,f'正文块{ai+1}/表内段{x+1}',f'正文块{bi+1}/表内段{y+1}')

styles=[]
oldstyles={left.names[k].lower():(k,e) for k,e in left.styles.items()}
newstyles={right.names[k].lower():(k,e) for k,e in right.styles.items()}
for name in sorted(set(oldstyles)|set(newstyles)):
    oldid,old=oldstyles.get(name,('',None));newid,new=newstyles.get(name,('',None))
    ap,bp=props(old,left),props(new,right)
    # A style's own identifier is a serialization label, not a font/layout value.
    for p in (ap,bp):
        for k in list(p):
            if k.endswith('/@w:styleId'):p.pop(k)
    diff=changes(ap,bp)
    if diff:styles.append({'style_name':name,'old_style_id':oldid,'new_style_id':newid,'old_usage':sum(1 for p in left.root.iter(W+'p') if left.style(p)==oldid) if oldid else 0,'new_usage':sum(1 for p in right.root.iter(W+'p') if right.style(p)==newid) if newid else 0,'changes':diff})
parts=[]
for name in sorted(set(left.parts)|set(right.parts)):
    av,bv=left.parts.get(name),right.parts.get(name)
    if av!=bv:parts.append({'part':name,'status':'新增' if av is None else '删除' if bv is None else '有变化','old_bytes':len(av) if av is not None else None,'new_bytes':len(bv) if bv is not None else None})

result={'baseline':str(BASE),'compared':str(OTHER),'baseline_authority':'用户最终明确D盘','method':'比较OOXML实际文字与属性；字符级原文完整保留；忽略命名空间声明和rsid编辑标记；样式按名称对应','content':content,'paragraph_format':paragraphs,'character_format':runs,'table_format':tables,'style_definitions':styles,'package_parts':parts,'alignment':{'base_blocks':len(left.blocks),'other_blocks':len(right.blocks),'base_flow':len(xs),'other_flow':len(ys),'paired_blocks':len(pairs)},'body_mapping':[{'old_body_index':left.flow[i],'new_body_index':right.flow[j]} for i,j in pairs]}
(A/'native_comparison.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
print(json.dumps({k:len(result[k]) for k in ('content','paragraph_format','character_format','table_format','style_definitions','package_parts')},ensure_ascii=False))
