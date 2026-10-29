from pathlib import Path
from zipfile import ZipFile
from collections import Counter,defaultdict
from lxml import etree as E
import json,re
from compare_layout_package import N,W,tx,props,diff,node,effective_p,get_styles
T=Path(__file__).resolve().parent;P=T/'layout_package_independent.json';j=json.loads(P.read_text())
def cnode(x):
 if isinstance(x,dict):
  if set(x)=={'tag','attrs','text','children'}:
   value=dict(x['attrs'])
   if x['text']:value['text']=x['text']
   if x['children']:
    children=defaultdict(list)
    for ch in x['children']:children[ch['tag']].append(cnode(ch))
    value.update({k:v[0]if len(v)==1 else v for k,v in children.items()})
   return value or True
  return{k:cnode(v)for k,v in x.items()}
 if isinstance(x,list):return[cnode(v)for v in x]
 return x
def groups(items,location):
 out={}
 for item in items:
  dd=item['differences'];key=json.dumps(dd,ensure_ascii=False,sort_keys=True)
  if key not in out:out[key]={'differences':cnode(dd),'locations':[]}
  out[key]['locations'].append(location(item))
 return list(out.values())
def fieldrecords(root):
 stack=[];out=[]
 for e in root.iter():
  if e.tag==W+'fldChar':
   typ=e.get(W+'fldCharType')
   if typ=='begin':stack.append({'instruction':'','result':'','mode':'instruction'})
   elif typ=='separate'and stack:stack[-1]['mode']='result'
   elif typ=='end'and stack:out.append(stack.pop())
  elif stack and e.tag==W+'instrText':stack[-1]['instruction']+=e.text or''
  elif e.tag==W+'t':
   for record in stack:
    if record['mode']=='result':record['result']+=e.text or''
 return out
with ZipFile(j['baseline'])as a,ZipFile(j['candidate'])as b:
 da,db=[E.fromstring(z.read('word/document.xml'))for z in(a,b)];sta,sa=get_styles(a);stb,sb=get_styles(b)
 tables=da.xpath('.//w:tbl',namespaces=N)
 for t in j['tables']['changed_exact_text_tables']:
  elem=tables[t['before_table_index_0based']];previous=elem.getprevious();caption=''
  while previous is not None:
   if previous.tag==W+'p':
    ps=previous.find('w:pPr/w:pStyle',N)
    if ps is not None and ps.get(W+'val')=='tablecaption':caption=tx(previous);break
   previous=previous.getprevious()
  t['nearest_preceding_table_caption']=caption
 # Complete field reconstruction avoids instrText split false positives.
 fs=[]
 for which,d in [('before',da),('after',db)]:
  bookmarks=d.xpath('.//w:bookmarkStart/@w:name',namespaces=N);names=set(bookmarks);f=fieldrecords(d);bad=[]
  for r in f:
   match=re.search(r'\b(?:REF|PAGEREF)\s+(?:"([^"]+)"|(\S+))',r['instruction'])
   if match and(match[1]or match[2])not in names:bad.append(r)
  fs.append({'file':which,'complex_fields':len(f),'instruction_kind_counts':dict(Counter(r['instruction'].strip().split()[0]if r['instruction'].strip()else'EMPTY'for r in f)), 'toc_instructions':[r['instruction']for r in f if r['instruction'].strip().startswith('TOC ')],'dangling_REF_PAGEREF':bad,'dangling_hyperlink_anchors':sorted(set(d.xpath('.//w:hyperlink/@w:anchor',namespaces=N))-names),'duplicate_bookmark_names':[n for n,c in Counter(bookmarks).items()if c>1]})
 j['internal_fields_and_bookmarks']=fs
 # Headers compared in their actual section slot, with real paragraph boundaries.
 for x in j['header_footer_changes']:
  for key,z in [('before',a),('after',b)]:
   part=x[key+'_part'];r=E.fromstring(z.read('word/'+part))if part else None
   x[key+'_paragraphs']=[]if r is None else [{'text':tx(p),'pPr':props(p.find('w:pPr',N)),'run_rPr':[props(q.find('w:rPr',N))for q in p.xpath('.//w:r',namespaces=N)]}for p in r.xpath('.//w:p',namespaces=N)]
   x[key+'_field_instructions']=[]if r is None else[r['instruction']for r in fieldrecords(r)]
   x.pop(key+'_xml',None)
 # Numbering IDs may change but the resolved list level should retain its meaning.
 def resolved_num(z,d):
  num=E.fromstring(z.read('word/numbering.xml'));ns={n.get(W+'numId'):n for n in num.findall('w:num',N)};abstract={n.get(W+'abstractNumId'):n for n in num.findall('w:abstractNum',N)};out=[]
  for i,p in enumerate(d.xpath('.//w:p',namespaces=N)):
   n=p.find('w:pPr/w:numPr',N)
   if n is None:continue
   ni=n.find('w:numId',N);li=n.find('w:ilvl',N);nid=ni.get(W+'val')if ni is not None else'0';lvl=li.get(W+'val')if li is not None else'0'
   x=ns.get(nid);abi=x.find('w:abstractNumId',N)if x is not None else None;ab=abstract.get(abi.get(W+'val'))if abi is not None else None;le=ab.find(f'w:lvl[@w:ilvl="{lvl}"]',N)if ab is not None else None
   out.append({'paragraph_index_0based':i,'text':tx(p),'num_id':nid,'level':lvl,'resolved_level':cnode(node(le,True))})
  return out
 na,nb=resolved_num(a,da),resolved_num(b,db)
 j['numbering_changes']={'before_abstract_lists':len(E.fromstring(a.read('word/numbering.xml')).findall('w:abstractNum',N)),'after_abstract_lists':len(E.fromstring(b.read('word/numbering.xml')).findall('w:abstractNum',N)),'before_resolved_used_lists':na,'after_resolved_used_lists':nb,'resolved_used_level_sequences_identical':[(x['text'],x['resolved_level'])for x in na]==[(x['text'],x['resolved_level'])for x in nb]}
 fonts=[]
 for z in(a,b):fonts.append({e.get(W+'name'):e for e in E.fromstring(z.read('word/fontTable.xml'))})
 j['font_table_changes']={'added_names':sorted(set(fonts[1])-set(fonts[0])),'removed_names':sorted(set(fonts[0])-set(fonts[1])),'changed_entries':[{'font':name,'before':cnode(node(fonts[0][name],True)),'after':cnode(node(fonts[1][name],True))}for name in sorted(set(fonts[0])&set(fonts[1]))if node(fonts[0][name],True)!=node(fonts[1][name],True)]}
 toc=[]
 for d in(da,db):toc.append([{'style':p.find('w:pPr/w:pStyle',N).get(W+'val'),'text':tx(p)}for p in d.xpath('.//w:p[w:pPr/w:pStyle[starts-with(@w:val,"TOC")]]',namespaces=N)])
 j['toc']={'counts':[len(toc[0]),len(toc[1])],'before_TOC4':[x['text']for x in toc[0]if x['style']=='TOC4'],'after_TOC4':[x['text']for x in toc[1]if x['style']=='TOC4'],'missing_still_present_h4':['（一） BAPR近期统计反馈与动态保守权重模型','（二）CS-BAPR可学习基准动作与均衡态归零差分修正结构','（三） SINDy短期符号代价辨识与软符号雅可比导数约束']}
 j['package']['content_type_semantic_changes']=diff({x.get('PartName')or x.get('Extension'):x.get('ContentType')for x in E.fromstring(a.read('[Content_Types].xml'))},{x.get('PartName')or x.get('Extension'):x.get('ContentType')for x in E.fromstring(b.read('[Content_Types].xml'))})
 j['notes_parts']={p:{'before_text':tx(E.fromstring(a.read(p))),'after_text':tx(E.fromstring(b.read(p))),'before_count':len(E.fromstring(a.read(p))),'after_count':len(E.fromstring(b.read(p)))}for p in ['word/footnotes.xml','word/endnotes.xml']}

j['findings']=[
 {'type':'actual_layout_change','location':'第3/4章页眉 header5.xml/header6.xml','finding':'原1段变2段，学校名从段首移入章名中间，顺序分别变为“第3章…单智能体 / 中南大学博士学位论文鲁棒强化学习…”与“第4章…条件识别 / 中南大学博士学位论文与因果符号约束…”。','impact':'页眉文字组织已变化，不能称仅调整外观且原文字顺序未变。'},
 {'type':'actual_toc_omission','location':'目录第4章','finding':'目录117项→114项；3个第4章h4目录条目丢失，而正文对应h4仍存在。','impact':'当前目录未完整覆盖原有层级。'},
 {'type':'toc_style_configuration_change','location':'目录域及13章/附录标题、5个h2、1个h3、参考文献标题','finding':'TOC的\\t自定义样式清单未改，20个对应标题已改用heading1/2/3，名称不在该清单中。','impact':'标题样式与目录域配置不再同步；应以实际原生目录更新结果判断所需调整。'},
 {'type':'actual_paragraph_spacing_change','location':'368个精确文本匹配段','finding':'直接lineRule=atLeast/line=400被删除，改继承body样式lineRule=exact/line=400，即20磅最小值变20磅固定值；包括大量原生行内公式正文。','impact':'取消了这些段落随高行内公式伸展的排版属性，原有行高保护未保留。未原生渲染，不声称已经截断。'},
 {'type':'actual_table_pagination_change','location':'表5-6，原table索引162','finding':'4行cantSplit均被删除，首行tblHeader被删除。','impact':'该表行现允许跨页，首行不再按原设置重复。'},
 {'type':'actual_table_spacing_change','location':'83个精确文本匹配表内段','finding':'15磅最小行距、段前后各2磅，改为单倍行距且未保留原段前后值。','impact':'表内密度与分页可能改变。'},
 {'type':'actual_geometry_change','location':'12个既有表（含6个公式表）','finding':'tblGrid列宽改变；另5个表tblW改变，2表tblLayout改变。具体旧新数值见tables。','impact':'列宽及公式/表格可用宽度真实改变。'},
 {'type':'actual_citation_format_change','location':'精确文本匹配段中的239个引用字符','finding':'vertAlign superscript被移除，改普通基线。','impact':'部分引文上标格式未保留。'},
 {'type':'actual_font_and_size_change','location':'正文/表格/图表题的直接文字格式','finding':'新增Times New Roman Regular、Cambria Math等西文字体设置；834个已匹配普通字符字号10.5→12磅，51字符12→10.5磅；777字符斜体取消。详见effective_format。','impact':'不能把全部字体/字号变化视为styleId重排；字体别名的实际显示由本机字体环境决定。'},
 {'type':'actual_caption_spacing_change','location':'图表题','finding':'49段增加段后1行；44表题改段前1行、段后0，并指定20磅固定行距（其中42段原无直接行距）。','impact':'图表题与正文/图表的距离改变。'},
 {'type':'actual_header_footer_mode_change','location':'settings.xml及第3–16节','finding':'删除evenAndOddHeaders；第3节默认页脚改footer4，旧footer3改空even；后续默认页脚顺移到footer5…footer17；PAGE域保留。','impact':'关闭奇偶页不同页脚/页眉后，default槽对两种页有效；空even槽不生效。footer17是普通PAGE页脚部件，无嵌入对象。'},
 {'type':'package_editor_metadata','location':'docProps/custom.xml等','finding':'新增KSOProductBuildVer、ICV，修订次数/创建修改时间和缓存统计变化；应用缓存页数304→289。','impact':'有WPS保存元数据痕迹；痕迹本身不代表恶意。缓存页数不当作原生分页验证。'},
 {'type':'package_relationship_change','location':'document.xml.rels','finding':'6条新增外部关系均为arxiv.org论文URL；内部新增footer17。无原外链目标删除，无悬空内部部件关系或REF/PAGEREF/超链接锚点。','impact':'这些外链是普通论文链接；包未增加宏、OLE、ActiveX或altChunk。'},
 {'type':'font_change_in_math_only','location':'MS Mincho 224处','finding':'224处全部在m:oMath内，普通正文文字为0处。','impact':'不能据此说正文被改为MS Mincho。'},
 {'type':'correct_unchanged','location':'页面与核心样式','finding':'16分节数量、所有页边距和纸张宽高、正文body/h2/h3/h4及图表题的样式字体/字号/核心行距定义保持；Normal中新增szCs22与移除docDefaults szCs22相互承接。','impact':'这些设置正确；实际变化主要在直接格式、样式应用、表结构及页眉目录。'},
 {'type':'equivalent_or_metadata_change','location':'内建样式、customXml、numbering','finding':'大量内建styleId重分配；qFormat等样式列表显示属性改变；空bibliography customXml属性顺序改变并删空schemaRef；编号定义重排。','impact':'不得把styleId或序列化字节变化直接认定格式损坏。已使用列表的解析结果单列核对。'}
]
# Deduplicate repetitive property payloads while retaining every matched location.
pf=j['paragraph_format'];pf['change_groups']=groups(pf.pop('changes'),lambda x:{'before_p':x['before']['paragraph_index_0based'],'after_p':x['after']['paragraph_index_0based'],'before_style':x['before']['style_name'],'after_style':x['after']['style_name'],'text':x['before']['text']})
ef=j['effective_format'];ef['paragraph_change_groups']=groups(ef.pop('paragraph_changes'),lambda x:{'before_p':x['before']['paragraph_index_0based'],'after_p':x['after']['paragraph_index_0based'],'text':x['before']['text']})
rf=j['run_direct_format'];rg=[]
for p in rf.pop('changes'):
 for s in p['segments']:rg.append({'differences':s['differences'],'location':{'before_p':p['before']['paragraph_index_0based'],'after_p':p['after']['paragraph_index_0based'],'text':s['text']}})
rf['change_groups']=groups(rg,lambda x:x['location'])
sc=j['styles'];sc['definition_change_groups']=groups(sc.pop('definition_changes'),lambda x:{k:x[k]for k in ['name','type','before_id','after_id','before_usage','after_usage']})
# Format dictionaries into compact property form; no observations are discarded.
j=cnode(j)
P.write_text(json.dumps(j,ensure_ascii=False,indent=2)+'\n')
print('saved',P,'bytes',P.stat().st_size,'findings',len(j['findings']))
