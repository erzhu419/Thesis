from pathlib import Path

import tighten_section_2_2 as editor


editor.BACKUP = Path(__file__).with_name("2.2行内公式调整前.docx")
editor.REVISIONS = {
    466: (
        "折扣任务采用",
        "；按实际时间折扣时，需指定时间单位，以",
        "计算延续因子。公交H2O+无折扣累计乘客成本，事件间隔直接进入奖励；第3章固定算子分析采用严格折扣。奖励、折扣与终止规则共同定义任务。",
    ),
    475: (
        "偶然不确定性表示给定条件下仍存在的随机结果，如同一客流强度下各分钟到站人数不同；认知不确定性来自数据或模型不足，如尚不清楚路段运行时间分布。增加数据可减少后者，但不能消除运行的随机波动。",
    ),
}


if __name__ == "__main__":
    editor.main()
