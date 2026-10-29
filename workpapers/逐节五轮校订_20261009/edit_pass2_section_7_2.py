from pathlib import Path
from shutil import copy2
from zipfile import ZipFile
import re
import xml.etree.ElementTree as ET

from edit_section_3_2 import W
from edit_section_4_4 import replace_paragraph


ROOT = Path(__file__).resolve().parents[2]
DOCUMENT = ROOT / "output/排版完成_逐节校订_第2轮_20261010.docx"
BACKUP = Path(__file__).with_name("校订前_第2轮_7.2.docx")
REVISIONS = {
    1937: (
        "（1）针对公交车辆异步运行、类别差异和经验分散，在前期类别特征研究基础上，将学习嵌入与连续运行量共同用于状态表示，按同一车辆相邻决策事件记录转移，使全车队共享策略和训练经验，同时保留运行身份差异。进一步提出RE-SAC，以集成动作价值为依据，结合价值目标的权重范数修正、评论家分歧惩罚和演员保守评价，减弱不可靠估值对驻站决策的影响，研究平稳随机环境下的鲁棒控制。",
    ),
    1938: (
        "（2）针对客流和路况等运行规律变化，提出BAPR及CS-BAPR，分别改进运行条件识别与策略响应。BAPR根据已发生转移更新运行模式判断，在受控执行器任务中补偿动作；公交实现则根据近期奖励和价值统计调节训练保守程度，评价时执行固定策略。CS-BAPR进一步结合训练稳定化与有界驻站输出。后续软符号配置以可学习基准动作和参考状态下为零的神经修正组成完整动作，再用SINDy参考关系对动作及其导数施加软约束，将参考动作及其变化规律纳入策略学习。",
    ),
    1939: (
        "（3）针对难以在目标环境持续试错的条件，在已有H2O+框架下开展公交跨保真适配，以固定SUMO离线数据初始化策略，再结合简易仿真新经验混合续训。分别编码同时到站车辆并保留同批已选动作，以离散概率表示不额外驻站（0秒）、连续分布表示其余时长，使离线与续训采用一致的动作概率定义。另设计行程完成保障，在指定时段对有前车、无在途后随车辆、载客率不高于25%且仍有可服务客流的车辆设置最低附加驻站时长。通过分别比较训练方案与执行规则，评价两者的服务作用。",
    ),
    1940: (
        "（4）针对源仿真标定不足造成的预测偏差，提出MC-WM，将预测模型选择与模型样本使用分别处理。依据未参与拟合的目标记录上的预测误差，在固定源模型并学习残差修正、直接学习目标模型两类方案间选择，不预设始终保留源预测。所选模型生成一步转移，以预测误差构造的置信权重参与价值学习。公交适配分别预测事件类别、人数、连续运行量、持续时间和乘客成本；归一化置信更新缓解累计对数置信造成的权重衰减，使预测样本有效参与训练，在固定目标数据下改进驻站动作评价。",
    ),
}


def main():
    if not BACKUP.exists():
        copy2(DOCUMENT, BACKUP)
    with ZipFile(BACKUP) as source:
        original = source.read("word/document.xml").decode("utf-8")
        body = ET.fromstring(original).find(W + "body")
        assert "".join(body[1936].itertext()) == "7.2 主要创新"
        assert "".join(body[1941].itertext()) == "7.3 研究展望"
        with ZipFile(DOCUMENT) as current:
            current_body = ET.fromstring(current.read("word/document.xml")).find(W + "body")
            assert len(current_body) == len(body)
            assert all(
                ET.tostring(a) == ET.tostring(b)
                for i, (a, b) in enumerate(zip(body, current_body))
                if i not in REVISIONS
            )
        updated = original
        for index, prose in REVISIONS.items():
            updated = replace_paragraph(updated, body[index], prose)

        # Retain every object, field and formatting property outside text nodes.
        mask = lambda xml: re.sub(r"(<w:t(?:\s[^>]*)?>).*?(</w:t>)", r"\1\2", xml, flags=re.S)
        assert mask(original) == mask(updated), "Non-text XML changed."
        final = ET.fromstring(updated).find(W + "body")
        assert len(body) == len(final) == 2924
        changed = [i for i, (a, b) in enumerate(zip(body, final)) if ET.tostring(a) != ET.tostring(b)]
        assert changed == list(REVISIONS), changed
        for index in changed:
            numbers = lambda p: re.findall(r"\d+(?:\.\d+)?", "".join(p.itertext()))
            assert numbers(body[index]) == numbers(final[index]), index
        temporary = DOCUMENT.with_suffix(".tmp")
        with ZipFile(temporary, "w") as target:
            for item in source.infolist():
                target.writestr(item, updated.encode("utf-8") if item.filename == "word/document.xml" else source.read(item.filename))
    temporary.replace(DOCUMENT)
    with ZipFile(BACKUP) as before, ZipFile(DOCUMENT) as after:
        assert before.namelist() == after.namelist()
        assert [n for n in before.namelist() if before.read(n) != after.read(n)] == ["word/document.xml"]
    print({"changed_paragraphs": len(changed), "backup": str(BACKUP), "output": str(DOCUMENT)})


if __name__ == "__main__":
    main()
