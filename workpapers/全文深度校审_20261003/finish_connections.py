"""Connect remaining appendix arguments to their specific main-text uses."""
from pathlib import Path
import shutil
from zipfile import ZipFile, ZIP_DEFLATED
from lxml import etree as E
from extract_word import NS, content
from revise_word import q, run, sub

HERE = Path(__file__).resolve().parent
BASE = HERE.parents[1] / 'output/排版完成.docx'
with ZipFile(BASE) as z:
    parts = {n: z.read(n) for n in z.namelist()}
doc = E.fromstring(parts['word/document.xml'])
blocks = list(doc.find('w:body', NS))
assert len(blocks) == 3164

additions = {
    1155: ('附录C.5进一步说明非负惩罚如何改变保守系数的代数方向，'
           'C.6将模式条件价值函数组织为共享输入函数；这两项分别对应信念调节和共享网络的表示方式。'),
    1157: ('附录C.8在逐步证据一致有利于新模式的假设下推导后验比率达到阈值所需的步数，'
           '用于说明证据强度与响应延迟之间的关系。'),
    1173: ('附录B.3说明正确补偿在共同噪声下保持逐路径等价，B.4分析无充分激励时模式不可辨识的原因，'
           '并以相对熵链式分解表达连续观测所累积的信息。B.5把误判转化为动作加权误差，'
           'B.6和B.7再将局部动作价值差分别沿固定噪声路径和随机轨迹累计；B.8据此得到极性误判的条件回报界。'
           '因此，识别准确率只有结合当时的动作幅度及局部价值敏感性，才足以解释控制损失。'),
    1176: ('理论上，附录D.6分析固定符号惩罚下的模式贝尔曼算子，D.7则用输入依赖反馈说明固定惩罚假设的重要性；'
           '二者区分给定评价式的收缩性质与训练中不断变化的反馈。'),
    1858: ('附录E.1和E.2分别给出随机核合法性与固定策略价值评价的证明；'
           '模型拟合过程若要具有固定点保证，还须满足附录E.5对完整残差更新映射另行提出的收缩条件。'),
}
for i, text in additions.items():
    assert text not in content(blocks[i]), i
    blocks[i].append(run(text))

old = '包含12条城市线路'
nodes = blocks[1228].xpath('.//w:t', namespaces=NS)
text = ''.join(node.text or '' for node in nodes)
start = text.index(old)
end = start + len(old)
offset = 0
for node in nodes:
    value = node.text or ''
    lo, hi = offset, offset + len(value)
    if lo < end and hi > start:
        left, right = max(start-lo, 0), min(end-lo, len(value))
        node.text = value[:left] + ('包含6条物理线路及其12个方向性服务' if lo<=start<hi else '') + value[right:]
    offset = hi

for node in list(blocks[1231].xpath('.//m:t', namespaces=NS)):
    if node.text == 'T':
        r = node.getparent()
        r.getparent().replace(r, sub('T', 'hor'))
blocks[1232].append(run('运营时域上限仍记为'))
math = E.SubElement(blocks[1232], q('m:oMath'))
math.append(sub('T', 'hor'))
blocks[1232].append(run('，与第2章一致。'))

parts['word/document.xml'] = E.tostring(doc, xml_declaration=True, encoding='UTF-8', standalone=True)
stage = HERE / '排版完成_附录衔接收尾.docx'
with ZipFile(stage, 'w', ZIP_DEFLATED) as z:
    for name, value in parts.items():
        z.writestr(name, value)
shutil.copy2(stage, BASE)
print('Updated appendix connections, physical-route terminology and evaluation horizon.')
