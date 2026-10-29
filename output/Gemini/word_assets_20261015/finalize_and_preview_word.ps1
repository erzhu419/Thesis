param([string]$InputDocument,[string]$PreviewDirectory)
$ErrorActionPreference='Stop'
Add-Type -AssemblyName System.Drawing
$word=$null;$doc=$null
try {
 $word=New-Object -ComObject Word.Application
 $word.Visible=$false;$word.DisplayAlerts=0
 $doc=$word.Documents.Open($InputDocument,$false,$false)
 $doc.TrackRevisions=$false
 $doc.ActiveWindow.View.Type=3
 $doc.Fields.Update() | Out-Null
 foreach($toc in $doc.TablesOfContents){$toc.Update() | Out-Null}
 $doc.Repaginate()
 foreach($toc in $doc.TablesOfContents){$toc.UpdatePageNumbers() | Out-Null}
 $doc.Save()
 $pages=$doc.ComputeStatistics(2)
 Write-Output ('PAGES='+$pages)
 Write-Output ('TOC_COUNT='+$doc.TablesOfContents.Count)
 Write-Output ('PANE_PAGES='+$doc.ActiveWindow.Panes.Item(1).Pages.Count)
 $targets=@('第1章 绪论','图1-1 目标信息','图3-8 固定温度','图5-9 在线学习','表6-5','参考文献','附录A RE-SAC','(5-14)','(5-10)','(4-23)','(2-25)','(3-9)')
 $samples=@(2,3,4,8)
 $locations=@()
 foreach($target in $targets){
  $r=$doc.Content.Duplicate;$r.Find.ClearFormatting()
  if($r.Find.Execute($target)){
   $pn=$r.Information(3);$samples+=$pn
   $locations+=@{text=$target;page=$pn;start=$r.Start}
  }
 }
 $samples=$samples | Select-Object -Unique
 foreach($pageNo in $samples){
  if($pageNo -lt 1 -or $pageNo -gt $pages){continue}
  try{$page=$doc.ActiveWindow.Panes.Item(1).Pages.Item([int]$pageNo)}catch{Write-Output ('BLANK_PAGE='+$pageNo);continue}
  [byte[]]$bits=$page.EnhMetaFileBits
  $stream=[System.IO.MemoryStream]::new($bits)
  $metafile=[System.Drawing.Imaging.Metafile]::new($stream)
  $bitmap=[System.Drawing.Bitmap]::new(1000,1414)
  $g=[System.Drawing.Graphics]::FromImage($bitmap)
  $g.Clear([System.Drawing.Color]::White)
  $g.DrawImage($metafile,0,0,1000,1414)
  $png=Join-Path $PreviewDirectory ('page_'+$pageNo.ToString('000')+'.png')
  $bitmap.Save($png,[System.Drawing.Imaging.ImageFormat]::Png)
  $g.Dispose();$bitmap.Dispose();$metafile.Dispose();$stream.Dispose()
  Write-Output ('PREVIEW='+$png)
 }
 $result=@{pages=$pages;toc=$doc.TablesOfContents.Count;locations=$locations;sample_pages=$samples;word_version=$word.Version}
 $result | ConvertTo-Json -Depth 5 | Out-File -Encoding utf8 (Join-Path $PreviewDirectory 'word_layout.json')
 $doc.Close(0);$doc=$null
 $word.Quit();$word=$null
}
finally {
 if($null -ne $doc){$doc.Close(0)}
 if($null -ne $word){$word.Quit()}
}
