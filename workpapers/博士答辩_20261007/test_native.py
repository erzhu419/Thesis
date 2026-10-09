from drawing import Deck

d = Deck()
d.page('驻站策略的两类信息来源', '中文与公式显示验证', notes='测试页，不进入正式答辩文件。')
d.node(.8, 2, 3.2, 1.4, '目标离线记录', '已采集的SUMO运行经验')
d.node(5.1, 2, 3.2, 1.4, '简易仿真交互', '可试错、持续产生新经验')
d.node(9.4, 2, 3.1, 1.4, '共享驻站策略', '在新SUMO场景中检验')
d.line(4, 2.7, 5, 2.7, arrow=True)
d.line(8.3, 2.7, 9.3, 2.7, arrow=True)
d.equation(.8, 4, 11.8, 1.5, r'\widetilde{y}_{\Delta}=(1-\gamma_{\Delta})\log c(s,a)+\gamma_{\Delta}\widetilde{Q}^{\Delta,-}(s^{\prime},\pi(s^{\prime}))', 26)
d.verdict('世界模型只参与训练；部署时仍由演员网络输出驻站动作。')
d.save('博士答辩_原生对象测试.pptx')
