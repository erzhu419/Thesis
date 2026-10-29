"""Replace only section 7.3 with the author's supplied text."""
from copy import deepcopy
from pathlib import Path
from shutil import copy2
from zipfile import ZipFile
import sys

from lxml import etree as E

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
sys.path.insert(0,str(ROOT/'workpapers/七章重写稿整合_20261010'))
from integrate import body_spans

SOURCE=ROOT/'output/排版完成_双清单替换_20261011.docx'
OUTPUT=ROOT/'output/排版完成_研究展望替换_20261012.docx'
W='{http://schemas.openxmlformats.org/wordprocessingml/2006/main}'
NS={'w':W[1:-1]}
TEXTS=[
    '本文围绕公交驻站控制开展研究，但尚未将大语言模型（Large Language Model，LLM）引入运行条件识别、策略训练与环境模型修正，也未开展真实道路闭环验证。未来研究可在现有事件驱动决策与乘客服务评价的基础上，重点探索大语言模型与强化学习的结合方式，并进一步检验其在实际公交运营中的适用性。',
    '（1）融合大语言模型的运行条件识别',
    '本文主要利用车辆、站点、车头时距和乘客人数等结构化观测，以及近期奖励和价值估计统计量处理运行变化，尚未将交通管制通告、活动安排和调度记录等文本信息纳入策略输入。未来可结合决策时已经获得的文本资料与车辆运行记录，研究利用大语言模型提取事件类型、影响线路、涉及站点及持续时段等信息的方法。例如，将活动散场信息与相关站点的客流变化联系起来，将道路施工通告与受影响路段的运行时间联系起来。经核验的事件信息可编码为附加状态，与车辆运行观测共同用于驻站策略训练。研究重点在于建立文本事件与公交运行状态之间的对应关系，检验这些信息能否缩短运行变化的识别延迟，并处理信息过期、描述不完整及文本判断与实际观测不一致等问题。',
    '（2）大语言模型辅助的奖励函数设计',
    '本文采用预先设定的车头时距奖励与乘客时间奖励训练策略，奖励形式、衰减尺度及惩罚权重仍需人工确定，尚未研究利用大语言模型辅助构造和调整训练目标的方法。未来可将公交运行规则、状态与动作定义、乘客服务要求作为输入，由大语言模型提出具有明确变量含义的候选奖励函数，经人工审核和程序检查后用于仿真训练，再依据评价结果调整奖励结构。例如，针对减少候车时间却增加车内延误的情形，研究如何在奖励中同时表达乘客时间代价与车辆间隔要求。不同候选奖励应采用相同的人均广义时间成本、车头时距及行程完成率进行评价，不随奖励变化而改变效果判断标准。研究重点在于减少奖励设计对反复人工调试的依赖，并检验所生成的奖励能否在未参与设计的运行场景中保持有效。',
    '（3）大语言模型与世界模型的协同建模',
    '本文通过目标域运行记录学习残差修正与直接目标模型，尚未研究如何利用大语言模型将交通运行知识转化为可供环境模型使用的预测条件或候选关系。未来可探索由大语言模型提出建模假设、由数值模型完成参数估计与运行预测的协同方式。例如，根据活动安排、道路管制和服务异常的描述，判断需要重点修正的是客流到达、路段运行时间还是上下客服务过程，并据此提出相应的输入变量或残差函数形式。候选关系的参数仍由目标域记录估计，其适用性由未参与拟合的数据检验，后继状态、事件间隔和乘客成本则由经检验的数值模型预测。研究重点在于判断语言知识能否在目标数据有限时改善模型修正与跨环境迁移，并识别、剔除与实际运行不符的建模假设，而不是将语言模型生成的描述直接作为真实转移数据。',
    '（4）大语言模型与强化学习的分层控制及实际验证',
    '本文尚未建立大语言模型与驻站策略协同运行的控制系统，现有跨保真验证也未覆盖真实道路中的观测误差、通信延迟和驾驶员执行差异。未来可研究分层控制方式：上层利用大语言模型处理事件信息和调度要求，在运行条件发生明显变化时更新条件描述，或选择经过验证的控制方案；下层由强化学习策略在车辆完成必要上下客后确定附加驻站时长，避免每次到站都依赖大语言模型重新推理。对上层输出应设置格式、有效时段和运行约束检查，在信息不足或响应超时时保留不依赖大语言模型的驻站方案。实际验证可先接入运营记录，仅生成控制建议而不干预车辆，再在限定线路和时段内开展受控试验。研究重点在于明确上下层的职责与更新频率，并在相同信息条件下与原驻站策略及不使用大语言模型的事件识别方法比较，综合评价乘客服务改善、响应时间和计算开销。',
]


def text(p):
    return ''.join(p.xpath('.//w:t/text()',namespaces=NS))


def main():
    HERE.mkdir(exist_ok=True)
    copy2(SOURCE,HERE/'修改前.docx')
    with ZipFile(SOURCE) as z:
        raw=z.read('word/document.xml')
        root=E.fromstring(raw)
        blocks=list(root.find(W+'body'))
        heading=next(i for i,p in enumerate(blocks) if text(p).strip()=='7.3 研究展望')
        start=heading+1
        stop=next(i for i in range(start,len(blocks)) if blocks[i].find('.//'+W+'sectPr') is not None)
        assert text(blocks[stop+1]).strip()=='参考文献'
        assert all(p.tag==W+'p' for p in blocks[start:stop])
        assert not any(p.xpath('.//w:drawing | .//w:object | .//m:oMath',namespaces={**NS,'m':'http://schemas.openxmlformats.org/officeDocument/2006/math'}) for p in blocks[start:stop])
        properties=blocks[start].find(W+'pPr')
        run_properties=blocks[start].find(W+'r/'+W+'rPr')
        bookmarks=[deepcopy(e) for p in blocks[start:stop] for e in p
                   if e.tag in {W+'bookmarkStart',W+'bookmarkEnd'}]
        paragraphs=[]
        for i,body in enumerate(TEXTS):
            p=E.Element(W+'p')
            if properties is not None:
                p.append(deepcopy(properties))
            if i==0:
                p.extend(bookmarks)
            r=E.SubElement(p,W+'r')
            rp=deepcopy(run_properties) if run_properties is not None else E.Element(W+'rPr')
            if i in {1,3,5,7}:
                E.SubElement(rp,W+'b')
                E.SubElement(rp,W+'bCs')
            if len(rp):
                r.append(rp)
            E.SubElement(r,W+'t').text=body
            paragraphs.append(p)
        spans=body_spans(raw)
        a,b=spans[start][0],spans[stop-1][1]
        modified=raw[:a]+b''.join(E.tostring(p) for p in paragraphs)+raw[b:]
        entries=[(info,modified if info.filename=='word/document.xml' else z.read(info.filename)) for info in z.infolist()]
    with ZipFile(OUTPUT,'w') as z:
        for info,data in entries:
            z.writestr(info,data)
    with ZipFile(SOURCE) as old,ZipFile(OUTPUT) as new:
        assert SOURCE.read_bytes()==(HERE/'修改前.docx').read_bytes()
        assert all(old.read(n)==new.read(n) for n in old.namelist() if n!='word/document.xml')
        after=list(E.fromstring(new.read('word/document.xml')).find(W+'body'))
        assert [text(p) for p in after[start:start+len(TEXTS)]]==TEXTS
        assert [E.tostring(p) for p in blocks[:start]]==[E.tostring(p) for p in after[:start]]
        assert [E.tostring(p) for p in blocks[stop:]]==[E.tostring(p) for p in after[start+len(TEXTS):]]
    print(f'Replaced only section 7.3: {stop-start} paragraphs -> {len(TEXTS)} paragraphs. Source and other content preserved. {OUTPUT}')


if __name__=='__main__':
    main()
