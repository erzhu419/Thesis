param([string]$DocumentPath,[string]$WorkFolder)
$ErrorActionPreference='Stop'
[Console]::OutputEncoding=[Text.UTF8Encoding]::new($false)
$existing=@(Get-Process WINWORD -ErrorAction SilentlyContinue | ForEach-Object {$_.Id})
$localFolder=Join-Path $env:TEMP ('thesis-final-reading-'+[Guid]::NewGuid().ToString('N'))
[void](New-Item -ItemType Directory -Path $localFolder)
$local=Join-Path $localFolder 'reviewed.docx'
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
 foreach($field in $doc.Fields){
  if($field.Type -eq 3 -and $field.Code.Text -match 'REF appF[123]num\b'){[void]$field.Update()}
 }
 $doc.Repaginate()
 foreach($toc in $doc.TablesOfContents){$toc.UpdatePageNumbers()}
 $doc.Repaginate()
 foreach($toc in $doc.TablesOfContents){$toc.UpdatePageNumbers()}
 $info.Pages=$doc.ComputeStatistics(2);$info.Equations=$doc.OMaths.Count;$info.Sections=$doc.Sections.Count
 $selected=[Collections.Generic.HashSet[int]]::new()
 $targets=@(
  '（1）针对不同车辆与站点的差异',
  '（2）针对公交客流与路况的统计规律',
  '（3）针对现实目标环境',
  '（4）针对源域和目标域',
  'To address differences among vehicles',
  'To address changing operating laws',
  'To reduce dependence on repeated trial-and-error',
  'To address source-target transition mismatch',
  'These findings provide a basis',
  '本文公交跨保真学习',
  '驻站策略通过状态特征提取与动作输出两部分构建。',
  '为考察评论家规模，研究另比较',
  '首先采用短预算因素比较协议',
  '校准选择驱动的世界模型适应与策略更新',
  '为平滑更新的目标网络。生成样本保留模型预测的物理奖励',
  '归一化常数是各模式证据分数的置信加权和',
  '(A-3)',
  '(C-11)',
  '(C-12)',
  '(C-13)',
  '给定非负证据',
  'F.1 早期接入检验',
  'F.2 最终控制器比较',
  'F.3 早期标定偏差探索'
 )
 foreach($text in $targets){
  $range=$doc.Content.Duplicate
  if($text -match '^F\.[123] '){$range.Start=$doc.Bookmarks.Item('appF').Range.Start}
  if($text -match '^\(([AC])-'){$range.Start=$doc.Bookmarks.Item('app'+$Matches[1]).Range.Start}
  $range.Find.ClearFormatting();$range.Find.Format=$false
  $range.Find.Text=$text;$range.Find.Wrap=0;$range.Find.MatchWildcards=$false
  if(-not $range.Find.Execute()){throw ('Preview target not found: '+$text)}
  $paragraph=$range.Paragraphs.Item(1).Range.Duplicate
  $end=$paragraph.Duplicate;$end.End--;$end.Collapse(0)
  $from=$range.Information(3);$to=$end.Information(3)
  $info.Targets+=[ordered]@{Text=$text;From=$from;To=$to}
  for($n=$from;$n -le $to;$n++){[void]$selected.Add($n)}
  if($text -eq '校准选择驱动的世界模型适应与策略更新'){[void]$selected.Add($from-1)}
 }
 foreach($bookmark in @('ref-niu2022h2o','ref-haarnoja2018soft','appA')){
  if($doc.Bookmarks.Exists($bookmark)){[void]$selected.Add($doc.Bookmarks.Item($bookmark).Range.Information(3))}
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
