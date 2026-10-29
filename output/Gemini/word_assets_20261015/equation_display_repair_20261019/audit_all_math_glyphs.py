from pathlib import Path
from zipfile import ZipFile
from collections import Counter,defaultdict
from lxml import etree as E
from fontTools.ttLib import TTFont,TTCollection
import json,unicodedata as U,re

T=Path(__file__).resolve().parent;SOURCE=T/'source_snapshot_current.docx'
N={'w':'http://schemas.openxmlformats.org/wordprocessingml/2006/main','m':'http://schemas.openxmlformats.org/officeDocument/2006/math','a':'http://schemas.openxmlformats.org/drawingml/2006/main'}
W='{'+N['w']+'}';M='{'+N['m']+'}'
FONTS=Path('/mnt/c/Windows/Fonts');loaded={};font_evidence={}

def ln(x):return E.QName(x).localname
def text(x):return ''.join(x.xpath('.//w:t/text()|.//m:t/text()',namespaces=N))
def flat(e):
 if e is None:return {}
 return{ln(c):{ln(k):v for k,v in c.attrib.items()}or{'val':'1'}for c in e}
def on(e):return e is not None and e.get(W+'val',e.get(M+'val','1'))not in('0','false','off')
def chain(styles,sid):
 out=[];seen=set()
 while sid in styles and sid not in seen:
  seen.add(sid);s=styles[sid];out.append(s);b=s.find('w:basedOn',N);sid=b.get(W+'val')if b is not None else None
 return list(reversed(out))
def apply_rpr(dst,e,style=False):
 if e is None:return
 f=flat(e)
 for prop in ['b','i','bCs','iCs']:
  if prop in f:
   yes=f[prop].get('val','1')not in('0','false','off');dst[prop]=not dst.get(prop,False)if style and yes else(dst.get(prop,False)if style else yes)
 for prop,val in f.items():
  if prop in('b','i','bCs','iCs'):continue
  if prop=='rFonts':
   old=dst.setdefault('rFonts',{})
   for key,theme in [('ascii','asciiTheme'),('hAnsi','hAnsiTheme'),('eastAsia','eastAsiaTheme'),('cs','cstheme')]:
    if key in val and theme not in val:old.pop(theme,None)
   old.update(val)
  elif prop=='lang':dst.setdefault(prop,{}).update(val)
  else:dst[prop]=val
def effective(r,p,styles,defaults):
 out={'b':False,'i':False,'bCs':False,'iCs':False};apply_rpr(out,defaults)
 sid=p.find('w:pPr/w:pStyle',N)if p is not None else None;sid=sid.get(W+'val')if sid is not None else'a0'
 for s in chain(styles,sid):apply_rpr(out,s.find('w:rPr',N),True)
 rstyle=r.find('w:rPr/w:rStyle',N)
 if rstyle is not None:
  for s in chain(styles,rstyle.get(W+'val')):apply_rpr(out,s.find('w:rPr',N),True)
 apply_rpr(out,r.find('w:rPr',N));return out,sid
def is_cjk(cp):return(0x2E80<=cp<=0x9FFF or 0xAC00<=cp<=0xD7AF or 0xF900<=cp<=0xFAFF or 0x20000<=cp<=0x323AF or 0xFF00<=cp<=0xFFEF)
def script_slot(c,hint):
 cp=ord(c)
 if cp<=0x7F:return'ascii'
 if is_cjk(cp):return'eastAsia'
 if 0x590<=cp<=0x8FF:return'cs'
 if hint=='eastAsia'and(0x100<=cp<=0x2FF or 0x370<=cp<=0x52F or 0x1E00<=cp<=0x1FFF or 0x2000<=cp<=0x27BF):return'eastAsia'
 return'hAnsi'
def resolve_face(fonts,slot,themes):
 theme_key={'ascii':'asciiTheme','hAnsi':'hAnsiTheme','eastAsia':'eastAsiaTheme','cs':'cstheme'}[slot]
 if fonts.get(theme_key):
  th=fonts[theme_key];major=th.startswith('major');key='major'if major else'minor';return themes[key]['eastAsia'if slot=='eastAsia'else'latin'],f'theme:{th}'
 return fonts.get(slot)or fonts.get('hAnsi')or fonts.get('ascii')or'Times New Roman',f'explicit_or_inherited:{slot}'
def font_key(face,bold=False,italic=False):
 low=face.lower().replace(' ','')
 if low in('timesnewroman','timesnewromanregular'):return('Times New Roman',bold,italic)
 if low in('宋体','simsun','nsimsun','新宋体'):return('NSimSun'if low in('nsimsun','新宋体')else'SimSun',False,False)
 if low in('cambria math','cambriamath'):return('Cambria Math',False,False)
 if low=='cambria':return('Cambria',bold,italic)
 if low in('msmincho','ｍｓ明朝'):return('MS Mincho',False,False)
 if low in('msgothic','ｍｓゴシック'):return('MS Gothic',False,False)
 return(face,bold,italic)
def cmap_for(key):
 if key in loaded:return loaded[key]
 family,bold,italic=key
 if family=='Times New Roman':file='times'+('bi'if bold and italic else'bd'if bold else'i'if italic else'')+'.ttf';idx=None
 elif family in('SimSun','NSimSun'):file='simsun.ttc';idx=0 if family=='SimSun'else 1
 elif family=='Cambria Math':file='cambria.ttc';idx=1
 elif family=='Cambria':file=('cambriab'if bold else'cambriai'if italic else'cambria')+('.ttf'if bold or italic else'.ttc');idx=None if bold or italic else 0
 else:
  candidates={p.name.lower():p for p in FONTS.iterdir()}
  file={'MS Mincho':'msmincho.ttc','MS Gothic':'msgothic.ttc'}.get(family,'');idx=0
  if file not in candidates:loaded[key]=None;font_evidence[str(key)]={'family':family,'available':False};return None
  file=candidates[file].name
 path=FONTS/file
 if not path.exists():loaded[key]=None;font_evidence[str(key)]={'file':str(path),'available':False};return None
 if idx is not None:
  coll=TTCollection(path,lazy=True);f=coll.fonts[idx]
 else:coll=None;f=TTFont(path,lazy=True)
 cmap={k:v for t in f['cmap'].tables if t.isUnicode()for k,v in t.cmap.items()}
 font_evidence[str(key)]={'file':str(path),'ttc_index':idx,'family':f['name'].getDebugName(1),'style':f['name'].getDebugName(2),'available':True,'unicode_glyphs':len(cmap)}
 if coll:coll.close()
 else:f.close()
 loaded[key]=cmap;return cmap

# Unicode names also cover the mathematical-alphanumeric legacy exceptions.
math_names={U.name(chr(cp),''):chr(cp)for cp in list(range(0x1D400,0x1D800))+list(range(0x2100,0x2150))}
exceptions={'script':dict(zip('BEFHILMRego',[0x212C,0x2130,0x2131,0x210B,0x2110,0x2112,0x2133,0x211B,0x212F,0x210A,0x2134])),'double-struck':dict(zip('CHNPQRZ',[0x2102,0x210D,0x2115,0x2119,0x211A,0x211D,0x2124])),'fraktur':dict(zip('CHIRZ',[0x212D,0x210C,0x2111,0x211C,0x2128]))}
def styled_letter(c,scr,sty):
 if not c.isascii()or not c.isalpha():return c
 if scr in exceptions and sty not in('b','bi')and c in exceptions[scr]:return chr(exceptions[scr][c])
 prefix={'script':'BOLD SCRIPT'if sty in('b','bi')else'SCRIPT','double-struck':'DOUBLE-STRUCK','fraktur':'BOLD FRAKTUR'if sty in('b','bi')else'FRAKTUR','sans-serif':{'p':'SANS-SERIF','b':'SANS-SERIF BOLD','i':'SANS-SERIF ITALIC','bi':'SANS-SERIF BOLD ITALIC'}.get(sty,'SANS-SERIF'),'monospace':'MONOSPACE','roman':{'p':'','b':'BOLD','i':'ITALIC','bi':'BOLD ITALIC'}.get(sty,'ITALIC')}.get(scr,'')
 if not prefix:return c
 name='MATHEMATICAL '+prefix+' '+('CAPITAL 'if c.isupper()else'SMALL ')+c.upper()
 if scr=='roman'and sty=='i'and c=='h':return'ℎ'
 return math_names.get(name,c)

def main():
 before=SOURCE.stat();missing=[];unresolved=[];styled_missing=[];styled_counts=Counter();summary=Counter();char_counts=Counter();struct_missing=[];partstats=[]
 with ZipFile(SOURCE)as z:
  s=E.fromstring(z.read('word/styles.xml'));styles={e.get(W+'styleId'):e for e in s.findall('w:style',N)};defaults=s.find('w:docDefaults/w:rPrDefault/w:rPr',N)
  sett=E.fromstring(z.read('word/settings.xml'));mf=sett.find('m:mathPr/m:mathFont',N);mathfont=mf.get(M+'val')if mf is not None else'Cambria Math'
  theme=E.fromstring(z.read('word/theme/theme1.xml'));themes={}
  for k in ['major','minor']:
   e=theme.find('.//a:'+k+'Font',N);themes[k]={'latin':e.find('a:latin',N).get('typeface'),'eastAsia':next((x.get('typeface')for x in e.findall('a:font',N)if x.get('script')=='Hans'),'宋体')}
  for part in z.namelist():
   if not(part.startswith('word/')and part.endswith('.xml')):continue
   d=E.fromstring(z.read(part));maths=d.xpath('.//m:oMath',namespaces=N)
   if not maths:continue
   ps=d.xpath('.//w:p',namespaces=N);pidx={p:i for i,p in enumerate(ps)};nearest={};chapter=''
   for p in ps:
    st=p.find('w:pPr/w:pStyle',N);sid=st.get(W+'val')if st is not None else'a0'
    if sid in('chapter','1','h2','2','h3','3','h4'):chapter=text(p)
    nearest[p]=chapter
   partstats.append({'part':part,'oMath_count':len(maths)})
   for mi,math in enumerate(maths):
    p=next((x for x in math.iterancestors()if x.tag==W+'p'),None);row=next((x for x in math.iterancestors()if x.tag==W+'tr'),None);label=''
    if row is not None:
     cells=row.findall(W+'tc');last=text(cells[-1]).strip()if cells else''
     if re.fullmatch(r'[（(](?:\d+|[A-Z])-\d+[a-z]?[)）]',last):label=last
    display=bool(math.xpath('ancestor::m:oMathPara',namespaces=N));kind='numbered'if label else'unlabeled_display'if display else'inline'
    summary['oMath_total']+=1;summary['oMath_'+kind]+=1;summary['native_display_oMath'if display else'native_inline_oMath']+=1
    loc={'part':part,'math_index_0based':mi,'paragraph_index_0based':pidx.get(p),'kind':kind,'label':label,'near_heading':nearest.get(p,''),'paragraph_text':text(p)[:220]if p is not None else'','math_text':text(math)[:180],'math_xpath':d.getroottree().getpath(math)}
    for ri,run in enumerate(math.xpath('.//m:r',namespaces=N)):
     summary['math_runs_total']+=1;normal=on(run.find('m:rPr/m:nor',N));summary['normal_text_runs'if normal else'math_mode_runs']+=1
     ts=run.findall('m:t',N);string=''.join(t.text or''for t in ts);eff,sid=effective(run,p,styles,defaults);fonts=eff.get('rFonts',{});miss=[];unknown=[];expected=[]
     mrpr=flat(run.find('m:rPr',N));scr=mrpr.get('scr',{}).get('val','roman');sty=mrpr.get('sty',{}).get('val','i')
     for ci,c in enumerate(string):
      if c.isspace()or U.category(c)in('Cf','Cc','Cs'):
       if c=='\u200b':summary['U200B_characters']+=1
       continue
      summary['visible_characters_checked']+=1
      if c=='&'and run.xpath('ancestor::m:eqArr',namespaces=N):summary['eqArr_ampersand_characters']+=1;continue
      if normal:
       slot=script_slot(c,fonts.get('hint','default'));face,path=resolve_face(fonts,slot,themes);key=font_key(face,eff.get('b',False),eff.get('i',False));render=c
      else:
       render=styled_letter(c,scr,sty)if'scr'in mrpr or'sty'in mrpr else c
       # A styled letter is still drawn with its run's effective font binding;
       # document mathFont does not erase inherited/direct rFonts. In particular
       # base Z + script renders U+1D4B5, which an inherited TNR run can lack.
       slot=script_slot(render,fonts.get('hint','default'));face,path=resolve_face(fonts,slot,themes);key=font_key(face,eff.get('b',False),eff.get('i',False))
       path+=';math_style_mapped'if'scr'in mrpr or'sty'in mrpr else';math_run_font_binding'
      cmap=cmap_for(key);char_counts[(face,'normal'if normal else'math')]+=1
      record={'character':c,'codepoint':f'U+{ord(c):04X}','character_index_in_run':ci,'unicode_name':U.name(c,'UNNAMED'),'font':face,'font_style':{'bold':key[1],'italic':key[2]},'font_slot':slot,'font_path':path,'expected_rendered_character':render,'expected_rendered_codepoint':f'U+{ord(render):04X}'}
      if cmap is None:unknown.append(record)
      elif ord(render)not in cmap:miss.append(record)
      if not normal and('scr'in mrpr or'sty'in mrpr):
       styled_counts[(scr,sty)]+=1
       if cmap is not None and ord(render)not in cmap:expected.append(record)
     if miss or unknown:
      item=dict(loc,run_index_0based=ri,run_text=string,run_xpath=d.getroottree().getpath(run),normal_text=normal,paragraph_style_id=sid,direct_m_rPr=mrpr,direct_w_rPr=flat(run.find('w:rPr',N)),effective_rFonts=fonts,missing_characters=miss,unresolved_characters=unknown)
      (missing if miss else unresolved).append(item)
      for c in miss:summary['missing_primary_glyph_characters']+=1
     if expected:styled_missing.append(dict(loc,run_index_0based=ri,run_text=string,script=scr,style=sty,missing_characters=expected))
    # Structural signs use the math font; they are not normal-text inherited runs.
    for e in math.xpath('.//m:chr|.//m:begChr|.//m:endChr|.//m:sepChr',namespaces=N):
     value=e.get(M+'val','');cmap=cmap_for(font_key(mathfont))
     for c in value:
      if c.isspace():continue
      summary['structural_sign_characters_checked']+=1
      if cmap is not None and ord(c)not in cmap:struct_missing.append(dict(loc,property=ln(e.tag),character=c,codepoint=f'U+{ord(c):04X}',font=mathfont))
 after=SOURCE.stat()
 grouped=Counter((c['character'],c['font'],r['kind'])for r in missing for c in r['missing_characters']);normal_candidates=[r for r in missing if r['normal_text']]
 out={'source':str(SOURCE),'source_size':before.st_size,'source_mtime_epoch':before.st_mtime,'source_stat_unchanged_during_read':before.st_size==after.st_size and before.st_mtime==after.st_mtime,'scope':'所有word/*.xml中全部oMath（行内、编号、无编号显示、符号表/表格）；只读。','summary':dict(summary),'parts_with_math':partstats,'font_resolution_method':'全部run按docDefaults→段落basedOn链→字符样式链→直接w:rPr解析；样式b/i按OOXML切换语义，直接字体覆盖下层主题。显式m:scr/m:sty先按Unicode数学字母映射，然后在该run有效字体cmap核对，不能无条件用文档mathFont替代。结构符号仍使用控制数学字体。忽略空白/格式控制字符；eqArr &另列，不当缺glyph。','audit_correction':'前次将非nor样式字母统一按Cambria Math检查漏掉3个script Z。现全部818styled字符按run有效字体重新核对；缺主要字体glyph只是候选，PDF实际gid0另标。','font_selection_limit':'缺失列表表示所解析主要字体cmap无该glyph；Office实际fallback和PDF是否gid0由redraw独立审计确认。默认hint下数学符号走hAnsi；hint=eastAsia的相应符号区走东亚字体。','font_evidence':font_evidence,'font_character_counts':[{'font':k[0],'mode':k[1],'characters':v}for k,v in char_counts.most_common()],'missing_primary_glyph_groups':[{'character':k[0],'codepoint':f'U+{ord(k[0]):04X}','font':k[1],'kind':k[2],'characters':v}for k,v in grouped.most_common()],'missing_primary_glyph_runs':missing,'unresolved_font_runs':unresolved,'styled_math_character_checks':{'counts':[{'script':k[0],'style':k[1],'characters_checked':v}for k,v in styled_counts.items()],'missing':styled_missing},'structural_sign_missing_glyphs':struct_missing,'minimal_repair_boundary':{'preferred_scope':'仅missing_primary_glyph_runs中被PDF字形0或原生显示确认的非中文符号字符；不整体替换oMath，不全篇改数学字体。','action':'将确诊符号单独拆run（如同run混有其他字母），只为该符号run显式指定Cambria Math的ascii/hAnsi/eastAsia/cs并去同run冲突主题字体属性；保留m:nor/字号/其他运算，先由单点原生预览证实是否还需恢复数学模式。','preserve':['全部文字和数学字符','原生分式、矩阵、上下标、m:scr/m:sty','已正确的单位/中文Normal Text','段落pPr、公式表几何、式号、引用、书签'],'candidate_normal_text_runs':len(normal_candidates),'not_authorized_here':'本审计不修改原Word，也不实施全篇字体恢复。'}}
 (T/'math_glyph_coverage_audit.json').write_text(json.dumps(out,ensure_ascii=False,indent=2)+'\n')
 print(json.dumps({'summary':out['summary'],'missing_runs':len(missing),'unresolved_runs':len(unresolved),'groups':out['missing_primary_glyph_groups'],'styled_missing':len(styled_missing),'structural_missing':len(struct_missing)},ensure_ascii=False))
if __name__=='__main__':main()
