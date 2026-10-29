from pathlib import Path
from shutil import copy2
from zipfile import ZipFile
import re
import xml.etree.ElementTree as ET

from edit_section_1_2_1 import DOCUMENT, M, W
from edit_section_3_2 import prose_groups
from edit_section_4_4 import replace_paragraph


BACKUP = Path(__file__).with_name("校订前_7.2.docx")
REVISIONS = {
    1937: (
        "（1）针对公交车辆异步运行、类别差异和经验分散的问题，在前期类别特征研究基础上，将学习嵌入与连续运行量共同用于状态表示，并按同车相邻事件构造转移，使车辆共享驻站经验的同时保留各自的运行身份。在此基础上提出RE-SAC，以集成动作价值估计为基础，通过价值目标的权重范数修正、评论家分歧约束和演员保守评价调节学习，将类别表征与平稳随机环境下的鲁棒驻站控制结合。",
    ),
    1938: (
        "（2）针对客流和路况等运行规律变化，提出BAPR及CS-BAPR，分别研究条件识别与受约束的策略响应。BAPR利用历史转移更新隐含模式判断，在受控执行器任务中按估计模式补偿动作；公交训练则利用近期奖励和价值统计调整保守程度，评价时执行固定策略。CS-BAPR进一步结合训练稳定化与有界驻站输出。后续软符号配置以可学习基准动作和参考状态下为零的神经修正组成完整动作，再用SINDy参考关系对动作及其导数施加软约束，将条件反馈、学习稳定性和驻站响应联系起来。",
    ),
    1939: (
        "（3）在已有H2O+框架基础上开展公交跨保真适配，利用固定SUMO离线数据与简易仿真新经验训练驻站策略。分别编码同时到站车辆，保留同批已选动作；用离散概率表示0秒动作、用连续分布表示其余驻站时长，统一离线初始化与源仿真续训的动作概率定义。另设计行程完成保障，在指定时段内，对有前车、无在途后随车辆、载客率不高于25%且仍有可服务客流的车辆设置最低附加驻站量。贡献集中于公交事件表示、驻站动作建模与执行规则，并通过不同对照分别评价混合训练和保障的服务作用。",
    ),
    1940: (
        "（4）针对源仿真标定不足造成的预测偏差，提出MC-WM，分别决定是否保留源预测，以及模型样本应有多大训练权重。在固定源模型并学习残差修正、直接学习目标模型两类方案之间，依据未参与拟合的目标记录上的预测误差选择；所选模型生成一步转移，再按预测误差构造的置信权重参与价值学习。公交适配联合预测下一事件的类别、人数、连续运行量、持续时间和乘客成本，并以归一化置信更新缓解累计对数置信导致的权重衰减，在固定目标数据条件下研究预测修正和模型信息的有效使用。",
    ),
}


def main():
    assert not BACKUP.exists(), "Do not overwrite the pre-edit backup."
    with ZipFile(DOCUMENT) as source:
        original = source.read("word/document.xml").decode("utf-8")
        body = ET.fromstring(original).find(W + "body")
        assert "".join(body[1936].itertext()) == "7.2 主要创新"
        assert "".join(body[1941].itertext()) == "7.3 研究展望"
        updated = original
        for index, prose in REVISIONS.items():
            updated = replace_paragraph(updated, body[index], prose)

        mask = lambda xml: re.sub(r"(<w:t(?:\s[^>]*)?>).*?(</w:t>)", r"\1\2", xml, flags=re.S)
        assert mask(original) == mask(updated), "Non-text XML changed."
        final_body = ET.fromstring(updated).find(W + "body")
        assert len(body) == len(final_body)
        changed = [i for i, (a, b) in enumerate(zip(body, final_body)) if ET.tostring(a) != ET.tostring(b)]
        assert changed == list(REVISIONS), changed
        protected_count = 0
        for i in changed:
            for tag in (M + "oMath", W + "hyperlink", W + "instrText", W + "fldChar", W + "pPr"):
                assert [ET.tostring(n) for n in body[i].iter(tag)] == [ET.tostring(n) for n in final_body[i].iter(tag)], (i, tag)
            nodes, _, protected = prose_groups(body[i])
            new_nodes = list(final_body[i].iter(W + "t"))
            assert all(ET.tostring(nodes[j]) == ET.tostring(new_nodes[j]) for j in protected), i
            protected_count += len(protected)
        copy2(DOCUMENT, BACKUP)
        temporary = DOCUMENT.with_suffix(".section-edit.docx")
        with ZipFile(temporary, "w") as target:
            for item in source.infolist():
                target.writestr(item, updated.encode("utf-8") if item.filename == "word/document.xml" else source.read(item.filename))
    temporary.replace(DOCUMENT)
    with ZipFile(BACKUP) as before, ZipFile(DOCUMENT) as after:
        assert before.namelist() == after.namelist()
        assert [n for n in before.namelist() if before.read(n) != after.read(n)] == ["word/document.xml"]
    print({"changed_blocks": changed, "protected_text_nodes": protected_count,
           "backup": str(BACKUP), "output": str(DOCUMENT)})


if __name__ == "__main__":
    main()
