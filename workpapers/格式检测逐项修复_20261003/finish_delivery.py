"""Deliver the validated Word, preview and a complete annotation accounting."""
from pathlib import Path
import json,shutil

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
OUT=ROOT/'output'
def read(name):return json.loads((HERE/name).read_text())
visual=read('逐页图像比对.json')
reviewed=set()
for filename in ['视觉复查_结构组.json','视觉复查_字体组.json','视觉复查_文献组.json']:
    record=read(filename)
    assert not record['findings'],filename
    reviewed.update(record.get('reviewed_pages',record.get('changed_pages',[])))
assert reviewed==set(visual['changed_pages'])
assert len(reviewed)+visual['identical_pages']==318
assert read('最终内容验收.json')['passed']
assert read('分节页面保留验收.json')['passed']
assert read('目录页码与结构验收.json')['passed']
assert read('字体实际显示验收.json')['passed']
assert read('字体实际显示验收.json')['visual_review_completed']
visual['all_changed_pages_visually_reviewed']=True
visual['review_evidence']=['视觉复查_结构组.json','视觉复查_字体组.json','视觉复查_文献组.json']
visual['reviewed_changed_page_count']=len(reviewed)
visual['conclusion']='All 178 changed pages were actually viewed; 140 pages are pixel-identical to the prior accepted preview. Existing whitespace remains deferred.'
(HERE/'逐页图像比对.json').write_text(json.dumps(visual,ensure_ascii=False,indent=2)+'\n')

word=read('Word分页与公式核对.json')
word['Status']='completed_with_separate_final_export'
word['FinalExportEvidence']='独立Word最终导出.json'
word['InitialUpdaterExport']='The updater saved the stable 318-page document. Its slow export was replaced by a successful fresh Word export; only the task-owned updater was stopped afterward.'
(HERE/'Word分页与公式核对.json').write_text(json.dumps(word,ensure_ascii=False,indent=2)+'\n')

report='''# 格式检测逐项修复记录

2026-10-03。已在最新《排版完成.docx》的副本上修复并保存成品，未使用旧Markdown或LaTeX重建。修复前Word保存在 [备份](../workpapers/格式检测逐项修复_20261003/排版完成_修复前.docx)。原检测PDF和高天宇模板未覆盖。

135条批注归为131个问题组，成对定位批注共同处理：73条已落实修复，42条核实正确并保留，19条空白提醒按要求暂缓，1条为检测汇总。详见 [逐条核对表](../workpapers/格式检测逐项修复_20261003/逐条批注核对表.md) 和 [机器记录](../workpapers/格式检测逐项修复_20261003/逐条批注处置.json)。

| 项目 | 检测要求及实际判断 | 输出实际值与处理 | 验收 |
|---|---|---|---|
| 西文字体5组 | 重音字母和符号实际受宋体选择影响 | 19个普通run中的相关字形采用Times New Roman，保留重音和符号 | 通过 |
| 数学字号5组 | 报告写13pt；原XML/PDF未复现13pt，但h被显示为MS Mincho | 标记位置母字号12pt，数学h为Cambria Math，下标自动缩放8.52pt | 通过；误报差异已记录 |
| 同类数学字体 | 完整预览另发现43个花体/黑板粗体字母选错字体 | 206个h及43个相关字母均为Cambria Math；花体/黑板粗体、上下标和字号保留 | 通过 |
| 四级目录 | 50条文字提醒及3个空定位标记对应53标题 | 自动TOC收录1–4级，382条目录；53个新增条目的页码与实际PDF逐项一致 | 通过 |
| 两条严重表号提醒 | 表3-1、表5-1至5-6末尾句号导致前表题漏识别；编号原本正确 | 删除7个题注末尾句号，保留正确连续表号 | 通过 |
| 段尾逗号12组 | 4处需要修改，其余为合法句内/多行数学分隔 | 2个引导段改冒号，式2-15、3-13末尾改句点；其余保留 | 通过 |
| 括号5组 | 均为合法半开区间或分段函数 | 保留正确数学结构，逐条记录依据 | 核实正确 |
| 文献28组 | 26条DOI及2条中文责任者要求核实 | 按注册元数据、期刊官网及原作者官方资料核实，当前信息正确 | 核实正确 |
| 文献对齐 | 最新稿与高天宇模板均为两端对齐；刘鹏杰PDF视觉也不是居中 | 保留原有两端对齐，12pt、固定20pt、左缩进30pt、悬挂2.5字符 | 通过 |

内容验收按顺序比较实质文字、表格结构、公式运算树和上下标、域码与显示值、书签及图片内容。除上述11处标点调整外，内容与对象保留；2038个原生公式、66张图片、684个正文SEQ/REF域均通过逐项比较，16个分节的页面、网格、编号及页眉页脚引用属性完全相同。不是仅凭对象数量判断。

最终分页和预览由Windows桌面Microsoft Word 16.0 / Build16.0.14334 / 程序版本16.0.14334.20918实际执行。目录更新和分页两轮均318页并稳定；为避免更新目录后的进程导出异常慢，最终预览由重新打开已保存副本的独立Word进程导出。源副本实际只读打开，无额外内容修改或保存。所需宋体、黑体、楷体、Times New Roman、Cambria Math均在Word字体列表中可用。

预览检查使用PyMuPDF1.27.2.2与Pillow11.3.0。全部318页均参与逐页图像比较：178个变化页面的45张联系图全部实际打开复查；140页与旧验收预览整页像素完全一致，复用已有视觉验收。10组字体/字号位置及43个新增数学字母另行查看局部；参考文献物理252–266共15页全部重新目测，154条显示文字及末句点完整，28个唯一DOI链接正确。未发现新增遮挡、溢出或错误页码。[逐页证据](../workpapers/格式检测逐项修复_20261003/逐页图像比对.json)。

## 保留事项

- 19条空白提醒按要求暂缓，前置页一个仅空格的MS Mincho字形也未按文字字体错误改动。
- 模板没有实际生效的第四级目录样例；本轮按收录四级标题的新要求，采用宋体/TNR12pt、固定20pt、31.5pt左缩进，承接前三层递进规则。这是明示推断。
- 未重新提交学校格式检测平台。本记录证明本地逐项修复与验收，不宣称平台已经重新判定通过。
'''
(OUT/'格式检测修复记录.md').write_text(report)

previous=OUT/'格式验收报告.md'
shutil.copy2(previous,HERE/'格式验收报告_修复前.md')
text=previous.read_text()
text=text.replace('317页','318页').replace('全317页','全318页').replace('I–XIX','I–XX')
text=text.replace('329条动态条目','382条动态条目')
text=text.replace('Word自动TOC，样式与有效直接格式一致，382条动态条目','Word自动TOC扩至四层；第四层宋体/TNR12pt、20pt、左缩进31.5pt，382条动态条目')
text=text.replace('自动目录收录原稿的摘要、ABSTRACT、主要符号及前三层标题','自动目录收录原稿的摘要、ABSTRACT、主要符号及前四层标题（本轮依检测报告新增第四层）')
notice='本轮格式检测修复已完成，当前318页；135条批注逐条处置，空白19条暂缓。最新修复、误报依据、字体与目录验收及逐页复查方法见 [格式检测修复记录](格式检测修复记录.md)。第四级目录按本轮要求补入，旧模板未有实际样例。以下历史内容验收与模板规则继续保留；新增检查以后附记录为准。\n\n'
text=text.replace('# 格式验收报告\n\n','# 格式验收报告\n\n'+notice,1)
previous.write_text(text)

# If the user saved a new Word revision while we worked, do not overwrite it.
assert (OUT/'排版完成.docx').read_bytes()==(HERE/'排版完成_修复前.docx').read_bytes(),'The authoritative Word changed during this task; merge the repairs first.'
shutil.copy2(HERE/'排版完成_修复候选.docx',OUT/'排版完成.docx')
shutil.copy2(HERE/'排版预览_修复候选.pdf',OUT/'排版预览.pdf')
print(json.dumps({'delivered_docx':str(OUT/'排版完成.docx'),'pages':318,
                  'annotations':read('逐条批注处置.json')['summary'],
                  'visually_reviewed_changed_pages':len(reviewed),'pixel_identical_pages':visual['identical_pages']},ensure_ascii=False,indent=2))
