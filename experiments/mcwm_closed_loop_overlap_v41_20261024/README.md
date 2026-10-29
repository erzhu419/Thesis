# v41：F后条件重叠与log c来源

当前只读：不训练、不启动原生前缀/分支、不回灌诊断标签。原初始化50000、24000秒目标时长、F和trust不变；完整保留v40最终三线9000–9300同窗、9630完整入库270响应槽、24+432=456更新及285总槽的未来方案，见复制的 `release_plan.json`。

对旧固定45事件链，根强制动作不参与RN，分析后44个执行动作。行为μ为原初始化完整随机Hurdle经F，目标π为三路原456更新后冻结分布。v39/v40参数证据只读复用，只加载服务器原初始化一次；两个场景fork并行，预留2 CPU、high。原子质量合并zero/clamp/floor/reserve，连续区经加权Gaussian交点和CDF解析积分，无Monte Carlo或动作RNG消耗。

各state计算overlap=∫min(dμF,dπF)=Eμc、Eμlogc。每链分别报告L_old=Σlogc(旧mode动作)、L_geom=ΣEμlogc、L_arith=Σlog overlap；旧动作选择偏移为L_old−L_geom，比率离散性由L_arith−L_geom描述。二者均条件于同一旧state序列，不是新随机行为的轨迹期望。

在旧执行动作固定时，用gate/mean/std八个参数组合的Shapley分配clipped logc差。gate同时替换原positive/zero概率，保留native rounding。两处atomic batch输入差异由同记录state下的初始化actor单列input baseline，后续三个分量只比较该可见输入下的初始化与旧目标。输入项+三参数项闭合到原v40logc；k log0.9独立，dt=0也照计。

数值检查针对实际故障：重放v40logc发现参数/事件错接；质量归一与交换对称发现F原子或Gaussian交点丢失；Jensen检查发现log矩积分错误；Shapley闭合发现clipping/输入混入错误。事件级角点、分布和积分证据留服务器，仅收≤24KB汇总。首奖励序号只在44个link计算完成后关联，不使用奖励数值拟合或作模型选择。

比较条件几何信用与旧mode信用后，再判断是否值得新共同前缀原生预检。旧状态选择偏差、真实奖励时序和新行为45事件可达仍未验证；本版本没有采集或训练提交路径。
