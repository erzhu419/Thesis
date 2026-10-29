from pathlib import Path
from shutil import copy2
from zipfile import ZipFile
import re
import xml.etree.ElementTree as ET

from edit_section_1_2_1 import W
from edit_section_4_4 import replace_paragraph


ROOT = Path(__file__).resolve().parents[2]
DOCUMENT = ROOT / "output/排版完成_逐节校订_第2轮_20261010.docx"
BACKUP = Path(__file__).with_name("校订前_第2轮_2.4.docx")
REVISIONS = {
    548: (
        "标定沿用式",
        "的时间划分。由一组需求生成种子和SUMO种子确定的基础随机条件称为场景组；同组可设置不同需求倍率，比较不同控制器。路段标定使用前五组、三档需求的15个SUMO场景，其余五组用于留出检验。速度记录按位置与时段整理，供源仿真估计路段速度；乘客到达、上下客和换乘仍按事件推进。以下先描述运行时间分布，再比较留出场景的结果。",
    ),
    549: (
        "道路运行从实际离站起算，必要上下客时间由服务过程决定，附加驻站时间由控制器给定，三者分别计时，避免站内时间被重复计入道路运行。除无附加驻站时的运行结果外，还需检验两环境对驻站动作的响应。",
    ),
    550: (
        "汇总30个无附加驻站SUMO回合的站间道路运行时间。各线路方向的分布不同，同一方向内也有离散性。记录包含站间距离、道路条件和运行时段的差异，不能将全部离散性归因于同一路段的随机波动；用全网统一常数将忽略这些位置差异。",
    ),
    551: (
        "样本来自三档需求、十个场景组的30个无附加驻站回合。箱体表示25%至75%分位区间，中线为中位数，下、上须线分别对应5%和95%分位数。这里只描述站间运行时间的分布，不将相邻路段记录作为独立样本作显著性推断。",
    ),
    577: (
        "七种策略与三档需求共形成21个组合。对每项指标，取各组合95%区间端点的绝对值，再取其中的最大值。广义时间为3.90%，候车时间为4.98%，车内时间为2.00%，车辆行程时长为1.17%，完成率为0.574个百分点。平均附加驻站时长相对差的对应值为20.78%，低于原设35%容差。平均驻站时长较短时，较小的绝对差也可能表现为较大的相对差，需结合乘客时间响应判断其影响。",
    ),
    601: (
        "训练种子控制网络初始化和训练采样顺序，反映同一方法重复训练的差异；环境种子控制客流和交通的随机实现，反映固定策略在不同随机运行条件下的表现。评价时保持已训练参数不变。两类重复分别汇总，不把同一策略的多次运行算作多次独立训练。",
    ),
}


def main():
    assert not BACKUP.exists(), "Keep the existing pre-edit backup."
    with ZipFile(DOCUMENT) as source:
        original = source.read("word/document.xml").decode("utf-8")
        body = ET.fromstring(original).find(W + "body")
        assert "".join(body[515].itertext()) == "2.4 仿真平台与实验设置"
        assert "".join(body[632].itertext()) == "2.5 本章小结"
        updated = original
        for index, prose in REVISIONS.items():
            assert 515 < index < 632
            updated = replace_paragraph(updated, body[index], prose)

        mask = lambda xml: re.sub(r"(<w:t(?:\s[^>]*)?>).*?(</w:t>)", r"\1\2", xml, flags=re.S)
        assert mask(original) == mask(updated), "Non-text XML changed."
        final_body = ET.fromstring(updated).find(W + "body")
        assert len(body) == len(final_body)
        changed = [i for i, (a, b) in enumerate(zip(body, final_body)) if ET.tostring(a) != ET.tostring(b)]
        assert changed == list(REVISIONS), changed
        for index in changed:
            before_text = "".join(body[index].itertext())
            after_text = "".join(final_body[index].itertext())
            assert re.findall(r"\d+(?:\.\d+)?", before_text) == re.findall(r"\d+(?:\.\d+)?", after_text), index

        copy2(DOCUMENT, BACKUP)
        temporary = DOCUMENT.with_suffix(".tmp")
        with ZipFile(temporary, "w") as target:
            for item in source.infolist():
                target.writestr(item, updated.encode("utf-8") if item.filename == "word/document.xml" else source.read(item.filename))
    temporary.replace(DOCUMENT)
    with ZipFile(BACKUP) as before, ZipFile(DOCUMENT) as after:
        assert before.namelist() == after.namelist()
        assert [n for n in before.namelist() if before.read(n) != after.read(n)] == ["word/document.xml"]
    print({"changed_blocks": changed, "paragraph_count": len(REVISIONS),
           "backup": str(BACKUP), "output": str(DOCUMENT)})


if __name__ == "__main__":
    main()
