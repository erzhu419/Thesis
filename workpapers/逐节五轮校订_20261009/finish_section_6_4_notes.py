from amend_section_6_4 import main


if __name__ == "__main__":
    main(
        paragraphs={
            1796: (
                "时间降幅以H2O+混合续训均值为分母，正值表示MC-WM成本降低；完成率差为MC-WM减H2O+。每格先平均五个配对测试日，再平均三次训练。",
            ),
        },
        terms_by_block={1808: {"完成率差/百分点": "完成率差 百分点"}},
    )
