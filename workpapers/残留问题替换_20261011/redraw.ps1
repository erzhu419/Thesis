param([Parameter(Mandatory=$true)][string]$Folder)
$ErrorActionPreference='Stop'
$app=[Runtime.InteropServices.Marshal]::GetActiveObject('Visio.Application')
$initial=$app.Documents.Count
$app.Visible=$true
$app.AlertResponse=1
$app.EventsEnabled=$false
$doc=$null
$teal='RGB(0,126,145)'
$violet='RGB(112,86,141)'
$orange='RGB(206,104,71)'
$green='RGB(37,139,99)'
$ink='RGB(35,46,51)'
$pale='RGB(239,247,249)'
$CanvasH=11.0

function Cell($s,$name,$value) { $s.CellsU($name).FormulaU=$value }
function Rect($x,$y,$w,$h,$fill,$stroke=$teal) {
    $s=$page.DrawRectangle($x,$CanvasH-$y-$h,$x+$w,$CanvasH-$y)
    Cell $s 'FillForegnd' $fill
    Cell $s 'FillPattern' '1'
    Cell $s 'LineColor' $stroke
    Cell $s 'LineWeight' '1.1 pt'
    Cell $s 'Rounding' '0.05 in'
    return $s
}
function Text($x,$y,$w,$h,$text,$size=28,$color=$ink,$bold=$false,$align=0) {
    $s=Rect $x $y $w $h 'RGB(255,255,255)' 'RGB(255,255,255)'
    Cell $s 'FillPattern' '0'
    Cell $s 'LinePattern' '0'
    $s.Text=$text
    Cell $s 'Char.Font' 'FONT("宋体")'
    Cell $s 'Char.Size' "$size pt"
    Cell $s 'Char.Color' $color
    Cell $s 'Char.Style' $(if($bold){'1'}else{'0'})
    Cell $s 'Para.HorzAlign' "$align"
    Cell $s 'VerticalAlign' '1'
    foreach ($v in @('LeftMargin','RightMargin')) {Cell $s $v '0.06 in'}
    foreach ($v in @('TopMargin','BottomMargin')) {Cell $s $v '0.025 in'}
    return $s
}
function Box($x,$y,$w,$h,$title,$body,$color=$teal,$size=28) {
    [void](Rect $x $y $w $h 'RGB(255,255,255)' $color)
    [void](Rect $x $y $w .52 $pale $color)
    [void](Text ($x+.05) ($y+.025) ($w-.1) .45 $title 30 $color $true)
    [void](Text ($x+.15) ($y+.66) ($w-.3) ($h-.8) $body $size)
}
function Arrow($x1,$y1,$x2,$y2,$color=$ink,$dash=$false) {
    $s=$page.DrawLine($x1,$CanvasH-$y1,$x2,$CanvasH-$y2)
    Cell $s 'LineColor' $color
    Cell $s 'LineWeight' '1.2 pt'
    Cell $s 'EndArrow' '4'
    if($dash){Cell $s 'LinePattern' '2'}
}
function Bus($x,$y,$color=$teal) {
    [void](Rect $x $y .75 .35 $color $color)
    foreach($dx in @(.08,.27,.46)){[void](Rect ($x+$dx) ($y+.05) .13 .13 'RGB(255,255,255)' 'RGB(255,255,255)')}
    foreach($dx in @(.12,.55)){
        $s=$page.DrawOval($x+$dx,$CanvasH-$y-.44,$x+$dx+.15,$CanvasH-$y-.29)
        Cell $s 'FillForegnd' $ink
        Cell $s 'LinePattern' '0'
    }
}
function Road($x,$y,$w) {
    $s=$page.DrawLine($x,$CanvasH-$y,$x+$w,$CanvasH-$y)
    Cell $s 'LineColor' 'RGB(140,151,157)'
    Cell $s 'LineWeight' '1.1 pt'
}
function Fig11 {
    Box .25 .25 15.5 1.25 '共同建模基础' '道路、线路、需求及时刻表 → 到站事件与附加驻站模型 → 自建事件仿真与SUMO' $teal 27
    Arrow 4 1.5 4 1.86
    Box .25 1.9 7.48 2.45 '第3章  平稳随机运行' "未预先获得目标域训练数据`n类别与连续状态 → 车队共享策略`nRE-SAC：目标价值修正与保守动作评价" $teal 28
    Box 8.27 1.9 7.48 2.45 '第4章  非平稳运行' "近期训练反馈 → 保守程度调节`n当前与参考输入 → 基准动作＋神经修正`n符号代价 → 动作参考与导数约束" $violet 27
    Arrow 7.73 3.1 8.27 3.1
    Arrow 12 4.35 12 4.6
    Arrow 12 4.6 4 4.6
    Arrow 4 4.6 4 4.86
    Box .25 4.9 7.48 2.65 '第5章  固定目标域离线数据可用' "SUMO记录 → 离线初始化`n目标记录＋源仿真交互 → H2O+混合训练`n驻站幅度调整与行程完成保障" $orange 28
    Box 8.27 4.9 7.48 2.65 '第6章  源仿真标定不足' "残差模型／直接目标模型 → 验证误差选择`n一步预测 → 归一化置信度加权价值学习`nMC-WM：模型辅助与完整控制器分别比较" $green 27
    Arrow 7.73 6.2 8.27 6.2
    Arrow 12 7.55 12 8.0
    Box .25 8.05 15.5 1.75 '共同评价与研究归纳' "计划时距偏差、前后时距不均衡度、候车与车内时间、行程完成率`n分别分析完整方法、组成因素与执行规则；SUMO是高保真代理，实车验证是后续应用。" $teal 27
    Bus .5 10.1;Road .4 10.6 15.2
    [void](Text 1.5 9.98 13.9 .6 '运行条件与目标数据条件逐步扩展' 29 $teal $true 1)
}
function Fig41 {
    Box .25 .25 15.5 3.0 '（a）最终软符号驻站策略' '' $teal
    Box .55 1.12 3.1 1.72 '当前与参考输入' "s 与 Π(s)`n仅替换前后时距" $teal 26
    Box 4.18 1.12 5.35 1.72 '同一网络的输出作差' "qθ(s) = bθ + mθ(s) − mθ(Π(s))`n参考输入处神经修正为0" $teal 26
    Box 10.08 1.12 5.35 1.72 '完整动作' "uθ(s) = 30[1 + tanh qθ(s)]`n0至60秒；基准驻站不必为0" $teal 26
    Arrow 3.65 1.95 4.18 1.95;Arrow 9.53 1.95 10.08 1.95
    Box .25 3.55 7.48 3.35 '（b）符号参考与策略导数约束' "随机驻站记录 → 线路—方向一步代价`n13个候选动作按代价加权 → fref(s)`n完整归一化策略与参考的雅可比比较`n归一化导数误差 × 0.01 → 策略损失`n不另加动作值拟合；行为克隆系数为0" $violet 26
    Box 8.27 3.55 7.48 3.35 '（c）价值目标' "奖励标准差缩放`n三个目标价值网络取较小估计＋熵项`n减去停止梯度后的批平均导数误差`n整批共享标量；系数0.01`n最终配置关闭目标权重范数项" $orange 26
    Arrow 4 3.55 4 3.25 $violet $true
    Arrow 12 3.55 12 3.25 $orange $true
    Box .25 7.25 15.5 2.05 '（d）训练反馈调节' "最近256条交互记录 → 奖励偏差与价值分歧变化 → 历史长度权重 → 保守评价系数`n公交评价固定训练完成后的策略；符号参考参与训练，不直接替代执行动作。" $teal 27
    Box .25 9.65 15.5 1.5 '早期结构对照' 'ReLU、tanh、NAU/NMU与KAN按各组设置比较，不与最终无NAU软符号配置混同。' $violet 27
    [void](Text .3 11.36 15.4 .45 '完整定义：附录D.1；主要结果：第4.4.5节' 26 $ink $false 1)
}
function Fig52 {
    Box .25 .25 15.5 3.3 '（a）跨保真训练' '' $teal
    Box .6 1.12 4.35 1.84 '固定SUMO离线数据' "IQL离线初始化`n目标离线经验持续回放" $violet 28
    Box 5.82 1.12 4.35 1.84 '新增源仿真交互' "在快速事件仿真中采样`n源经验与目标经验1∶1" $teal 28
    Box 11.05 1.12 4.35 1.84 'H2O+混合更新' "更新策略与价值网络`n固定训练完成后的策略" $orange 28
    Arrow 4.95 2.05 5.82 2.05;Arrow 10.17 2.05 11.05 2.05
    [void](Text .6 3.05 14.8 .4 '经验回放比例不同于后继价值组合系数。' 26 $ink $false 1)
    Box .25 3.9 7.48 3.3 '（b）同时到站的决策组织' "当前状态＋同批待决策车辆信息`n＋本批已选但尚未执行的动作`n→ 依次选择各车动作`n→ 收齐动作后推进仿真`n决策批次不等于车辆运营班次" $teal 27
    Box 8.27 3.9 7.48 3.3 '（c）混合驻站动作' "策略分别给出：`n不额外驻站的概率`n选择额外驻站时的连续时长分布`n0秒：完成必要上下客后不再额外停留`n不表示跳站或取消上下客" $orange 27
    Box .25 7.55 15.5 3.0 '（d）完整控制器与行程完成保障' "已训练策略 → 驻站幅度调整 → 按运行状态触发保障 → 最终附加驻站时长`n保障条件：限定启用时段、有前车、无在途后随车辆、载客率不高于25%，且仍有可服务客流。`n不将本车直接认定为当日末班车；保障收益与训练方法收益分别比较。" $green 28
    Arrow 8 10.55 8 10.95
    Box .25 11.03 15.5 2.1 '（e）独立服务评价' "10个新场景组 × 3档需求 × 5个训练策略 × 9种控制器 = 1350回合`n同一策略下比较执行保障，不同方法间比较训练方案；1350回合不等于1350次独立训练。" $teal 28
    Bus .5 13.4;Road .4 13.88 15.2
}
function Fig61 {
    Box .25 .25 15.5 1.65 '（a）目标记录划分' "拟合集；特征筛选与提前停止验证集；模型选择验证集`n公交按完整运行日或场景组划分，分别用于拟合决策与候选比较。" $teal 28
    Arrow 8 1.9 8 2.28
    Box .25 2.35 7.48 2.05 '残差候选' "固定源模型＋目标数据学习的残差修正`n保留源模型预测结构" $teal 28
    Box 8.27 2.35 7.48 2.05 '直接目标候选' "用目标数据直接拟合`n不保留源模型预测" $green 28
    Arrow 4 4.4 4 4.83;Arrow 12 4.4 12 4.83
    Box .25 4.9 15.5 1.9 '（b）模型类别选择' "在同一模型选择验证集上比较预测误差，选择误差较小的候选。`n受控连续任务：初始选择；公交：一次拟合固定类别，分阶段方案重新拟合并选择。" $violet 28
    Arrow 8 6.8 8 7.18
    Box .25 7.25 7.48 2.35 '（c）一步模型样本' "已观察到的状态与动作 → 所选世界模型`n→ 预测后继状态、奖励／成本和事件信息`n模型样本不等于新增目标环境观测" $teal 27
    Box 8.27 7.25 7.48 2.35 '（d）样本使用' "实际记录 → 混合经验损失`n模型样本 → 置信加权价值损失`n模型损失系数与逐样本权重分别调节" $orange 27
    Arrow 7.73 8.42 8.27 8.42
    Box .25 9.95 15.5 2.1 '（e）公交置信尺度调整' "当前局部评分的对数 × (1 − γΔ) ＋ 后继对数置信递推`n→ 缓解模型权重过度减小；执行动作仍由策略网络选择。" $green 29
    Box .25 12.4 15.5 1.05 '输出及更新安排' '模型输出、划分单位与更新安排分别按第6.2—6.3节说明。' $teal 26
}

try {
    foreach($name in @('图1-1','图4-1','图5-2','图6-1')) {
        Write-Output ('OPEN '+$name)
        $doc=$app.Documents.OpenEx((Join-Path $Folder ($name+'_修改前.vsdx')),2)
        $page=$doc.Pages.Item(1)
        $page.Export((Join-Path $Folder ($name+'_修改前.png')))
        $CanvasH=switch($name){'图1-1'{10.94};'图4-1'{12.08};'图5-2'{14.03};'图6-1'{13.91}}
        $page.PageSheet.CellsU('PageWidth').FormulaU='16 in'
        $page.PageSheet.CellsU('PageHeight').FormulaU="$CanvasH in"
        while($page.Shapes.Count -gt 0){$page.Shapes.Item(1).Delete()}
        switch($name){'图1-1'{Fig11};'图4-1'{Fig41};'图5-2'{Fig52};'图6-1'{Fig61}}
        $doc.SaveAs((Join-Path $Folder ($name+'_校订.vsdx')))
        $page.Export((Join-Path $Folder ($name+'_校订.emf')))
        $page.Export((Join-Path $Folder ($name+'_校订.png')))
        Write-Output ('EXPORTED '+$name+' '+$page.Shapes.Count+' editable shapes')
        $doc.Close();$doc=$null
    }
    Write-Output ('Visio '+$app.Version)
}
finally {
    if($null-ne$doc){$doc.Close()}
    if($initial-eq0){$app.Quit()}
}
