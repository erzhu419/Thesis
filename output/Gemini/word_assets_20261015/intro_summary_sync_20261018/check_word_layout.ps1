param([string]$InputDocument,[string]$PreviewDirectory)
$ErrorActionPreference='Stop'
Add-Type -AssemblyName System.Drawing
$word=$null;$doc=$null
try {
 $word=New-Object -ComObject Word.Application
 $word.Visible=$false;$word.DisplayAlerts=0
 $doc=$word.Documents.Open($InputDocument,$false,$false)
 $doc.ActiveWindow.View.Type=3
 $doc.Repaginate()
 foreach($toc in $doc.TablesOfContents){$toc.UpdatePageNumbers() | Out-Null}
 $doc.Repaginate()
 $doc.Save()
 $pages=$doc.ComputeStatistics(2)
 $targets=@('城市常规公交系统在日常运营中','本章面向尚未预先获取目标运营环境','本章针对城市公交仿真模型','统计信息','各自的Lipschitz')
 $locations=@();$samples=@()
 foreach($target in $targets){
  $r=$doc.Content.Duplicate;$r.Find.ClearFormatting()
  if(-not $r.Find.Execute($target)){throw ('Missing supplied text: '+$target)}
  $pn=$r.Information(3);$samples+=$pn
  $locations+=@{text=$target;page=$pn}
 }
 foreach($pageNo in ($samples | Select-Object -Unique)){
  $page=$doc.ActiveWindow.Panes.Item(1).Pages.Item([int]$pageNo)
  [byte[]]$bits=$page.EnhMetaFileBits
  $stream=[System.IO.MemoryStream]::new($bits)
  $metafile=[System.Drawing.Imaging.Metafile]::new($stream)
  $bitmap=[System.Drawing.Bitmap]::new(1000,1414)
  $g=[System.Drawing.Graphics]::FromImage($bitmap)
  $g.Clear([System.Drawing.Color]::White)
  $g.DrawImage($metafile,0,0,1000,1414)
  $bitmap.Save((Join-Path $PreviewDirectory ('page_'+$pageNo.ToString('000')+'.png')),[System.Drawing.Imaging.ImageFormat]::Png)
  $g.Dispose();$bitmap.Dispose();$metafile.Dispose();$stream.Dispose()
 }
 @{pages=$pages;locations=$locations;toc=$doc.TablesOfContents.Count;word_version=$word.Version} | ConvertTo-Json -Depth 5 | Out-File -Encoding utf8 (Join-Path $PreviewDirectory 'word_layout.json')
 Write-Output ('PAGES='+$pages)
 $doc.Close(0);$doc=$null
 $word.Quit();$word=$null
}
finally {
 if($null -ne $doc){$doc.Close(0)}
 if($null -ne $word){$word.Quit()}
}
