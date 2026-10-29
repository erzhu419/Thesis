param([string]$BaseDocument,[string]$OtherDocument,[string]$PreviewDirectory)
$ErrorActionPreference='Stop'
Add-Type -AssemblyName System.Drawing
$word=$null;$doc=$null;$results=@()
try {
 $word=New-Object -ComObject Word.Application
 $word.Visible=$false;$word.DisplayAlerts=0
 $targets=@('完整RE-SAC使全网平均计划间隔偏差','源仿真标定可','图3-6 ','图3-7 ','图4-7 ','图5-2 ')
 foreach($entry in @(@{name='baseline';path=$BaseDocument},@{name='adjusted';path=$OtherDocument})){
  $doc=$word.Documents.Open($entry.path,$false,$true)
  $doc.ActiveWindow.View.Type=3
  $doc.Repaginate()
  $locations=@();$samples=@()
  foreach($target in $targets){
   $r=$doc.Content.Duplicate;$r.Find.ClearFormatting()
   if($r.Find.Execute($target)){
    $pn=$r.Information(3);$samples+=$pn
    $locations+=@{text=$target;physical_page=$pn;printed_page=$r.Information(1)}
   }
  }
  foreach($pageNo in ($samples | Select-Object -Unique)){
   $page=$doc.ActiveWindow.Panes.Item(1).Pages.Item([int]$pageNo)
   [byte[]]$bits=$page.EnhMetaFileBits
   $stream=[System.IO.MemoryStream]::new($bits)
   $metafile=[System.Drawing.Imaging.Metafile]::new($stream)
   $bitmap=[System.Drawing.Bitmap]::new(1200,1697)
   $g=[System.Drawing.Graphics]::FromImage($bitmap)
   $g.Clear([System.Drawing.Color]::White)
   $g.DrawImage($metafile,0,0,1200,1697)
   $bitmap.Save((Join-Path $PreviewDirectory ($entry.name+'_page_'+$pageNo.ToString('000')+'.png')),[System.Drawing.Imaging.ImageFormat]::Png)
   $g.Dispose();$bitmap.Dispose();$metafile.Dispose();$stream.Dispose()
  }
  $results+=@{document=$entry.path;label=$entry.name;read_only=$doc.ReadOnly;pages=$doc.ComputeStatistics(2);locations=$locations}
  $doc.Close(0);$doc=$null
 }
 $results | ConvertTo-Json -Depth 6 | Out-File -Encoding utf8 (Join-Path $PreviewDirectory 'native_preview.json')
 Write-Output ('READ_ONLY_PREVIEWS_COMPLETED='+$results.Count)
 $word.Quit();$word=$null
}
finally {
 if($null -ne $doc){$doc.Close(0)}
 if($null -ne $word){$word.Quit()}
}
