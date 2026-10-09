param([string]$DocumentPath,[string]$WorkFolder)
$ErrorActionPreference='Stop'
[Console]::OutputEncoding=[Text.UTF8Encoding]::new($false)
$existing=@(Get-Process WINWORD -ErrorAction SilentlyContinue | ForEach-Object {$_.Id})
$folder=Join-Path $env:TEMP ('thesis-reference-review-'+[Guid]::NewGuid().ToString('N'))
[void](New-Item -ItemType Directory -Path $folder)
$local=Join-Path $folder 'reviewed.docx'
Copy-Item -LiteralPath $DocumentPath -Destination $local
$word=$null;$doc=$null;$owned=$false;$options=@{}
$info=[ordered]@{Status='starting';PreviewPages=@();Targets=@()}
function Render-Page($Page,$Path) {
 $stream=[IO.MemoryStream]::new([byte[]]$Page.EnhMetaFileBits,$false)
 $metafile=[Drawing.Imaging.Metafile]::new($stream)
 $bitmap=[Drawing.Bitmap]::new(1191,1684)
 $graphics=[Drawing.Graphics]::FromImage($bitmap)
 $graphics.Clear([Drawing.Color]::White)
 $graphics.DrawImage($metafile,[Drawing.Rectangle]::new(0,0,1191,1684))
 $bitmap.Save($Path,[Drawing.Imaging.ImageFormat]::Png)
 $graphics.Dispose();$bitmap.Dispose();$metafile.Dispose();$stream.Dispose()
}
try {
 $word=New-Object -ComObject Word.Application
 $process=@(Get-Process WINWORD | Where-Object {$existing -notcontains $_.Id})
 if($process.Count -ne 1){throw 'Independent Word process not identified.'}
 $owned=$true;$info.ProcessId=$process[0].Id;$info.Version=$word.Version;$info.Build=$word.Build
 foreach($name in @('UpdateLinksAtOpen','UpdateLinksAtPrint','UpdateFieldsAtPrint','Pagination')){
  $options[$name]=$word.Options.$name;$word.Options.$name=$false
 }
 $word.AutomationSecurity=3;$word.DisplayAlerts=0;$word.Visible=$false
 $doc=$word.Documents.Open($local,$false,$false,$false)
 $doc.PrintRevisions=$false;$doc.ActiveWindow.View.Type=3
 $counts=@()
 for($pass=0;$pass -lt 2;$pass++){
  $doc.Repaginate()
  foreach($toc in $doc.TablesOfContents){$toc.UpdatePageNumbers()}
  $doc.Repaginate()
  $counts+= $doc.ComputeStatistics(2)
 }
 if($counts[0] -ne $counts[1]){throw 'Pagination did not stabilize after two updates.'}
 $info.Pages=$counts[1];$info.Equations=$doc.OMaths.Count;$info.Sections=$doc.Sections.Count
 $selected=[Collections.Generic.HashSet[int]]::new()
 $targets=@(
  '公交控制嵌入在由线网设计、频率配置、时刻表编制和车辆运用构成的决策体系中。',
  '滚动优化进一步将未来车辆和乘客变化纳入当前决策。赖心荷',
  '模型中的乘客行为和车辆约束会改变控制效果。随机行程',
  '国内公交研究也从乘客服务和具体控制权限出发处理运行失衡。接驳',
  '不同控制方式改变公交运行过程的不同环节。驻站、跳站',
  '在价值估计与策略优化方面，策略表征研究讨论状态到动作映射',
  '多智能体研究分别从控制与决策方法',
  '风险偏好决定控制器除了平均表现之外，是否还要重点关注较差的运行结果。',
  '非平稳强化学习将持续变化作为决策过程的一部分。开放',
  '因果建模研究关注因果关系对强化学习控制的作用',
  '首先考虑只能使用既有运行记录的情况。纯离线',
  '离线数据的形成方式决定了可以检验哪些能力。深度',
  '机器人虚实迁移综述从目标模型优化、源知识迁移',
  '世界模型通过学习状态转移与奖励，为策略评价和优化生成额外信息。',
  '潜在空间强化学习综述分别讨论状态表示、动作表示和动力学模型',
  '算术输出头（表中记为NAU/NMU）',
  '驻站策略通过状态特征提取与动作输出两部分构建。',
  '在相同任务下，研究比较了采用ReLU激活的MLP、采用平滑',
  '归一化输出再映射至公交停站动作范围。NAU提供线性组合',
  '已有可学习样本加权研究利用模型样本更新前后在环境样本上的损失变化',
  '固定置信算子的收缩性描述精确价值更新。置信、惩罚和价值同时变化时'
 )
 foreach($text in $targets){
  $range=$doc.Content.Duplicate
  $range.Find.ClearFormatting();$range.Find.Format=$false
  $range.Find.Text=$text;$range.Find.Wrap=0;$range.Find.MatchWildcards=$false
  if(-not $range.Find.Execute()){throw ('Preview target not found: '+$text)}
  $paragraph=$range.Paragraphs.Item(1).Range.Duplicate
  $end=$paragraph.Duplicate;$end.End--;$end.Collapse(0)
  $from=$range.Information(3);$to=$end.Information(3)
  $info.Targets+=[ordered]@{Text=$text;From=$from;To=$to}
  for($n=$from;$n -le $to;$n++){[void]$selected.Add($n)}
 }
 foreach($bookmark in $doc.Bookmarks){
  if($bookmark.Name -like 'ref-*'){
   $range=$bookmark.Range.Duplicate
   $start=$range.Duplicate;$start.Collapse(1)
   $end=$range.Duplicate;$end.Collapse(0)
   for($n=$start.Information(3);$n -le $end.Information(3);$n++){[void]$selected.Add($n)}
  }
 }
 foreach($toc in $doc.TablesOfContents){
  $start=$toc.Range.Duplicate;$start.Collapse(1)
  $end=$toc.Range.Duplicate;$end.Collapse(0)
  for($n=$start.Information(3);$n -le $end.Information(3);$n++){[void]$selected.Add($n)}
 }
 $doc.Save();Copy-Item -LiteralPath $local -Destination $DocumentPath -Force
 Write-Output ('SAVED pages='+$info.Pages+' equations='+$info.Equations)
 Add-Type -AssemblyName System.Drawing
 $preview=Join-Path $WorkFolder 'Word页面复查'
 [void](New-Item -ItemType Directory -Force -Path $preview)
 $pages=$doc.ActiveWindow.Panes.Item(1).Pages
 foreach($n in @($selected | Sort-Object)){
  Render-Page $pages.Item($n) (Join-Path $preview ('page_'+$n.ToString('000')+'.png'))
  $info.PreviewPages+=$n
 }
 $info.Status='completed';$info.TocPageNumbersUpdated=$true
 Write-Output ('PREVIEWS '+$info.PreviewPages.Count)
} finally {
 if($doc){$doc.Close([ref]0)}
 if($owned){foreach($name in $options.Keys){$word.Options.$name=$options[$name]};$word.Quit([ref]0)}
 [IO.File]::WriteAllText((Join-Path $WorkFolder 'Word复查记录.json'),($info | ConvertTo-Json -Depth 6),[Text.UTF8Encoding]::new($false))
}
