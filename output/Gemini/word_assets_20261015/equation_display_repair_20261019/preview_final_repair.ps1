param([string]$InputDocument,[string]$PreviewDirectory)
$ErrorActionPreference='Stop'
Add-Type -AssemblyName System.Drawing
$word=$null;$doc=$null
try {
 $word=New-Object -ComObject Word.Application
 $word.Visible=$false;$word.DisplayAlerts=0
 $doc=$word.Documents.Open($InputDocument,$false,$true)
 if(-not $doc.ReadOnly){throw 'Expected read-only final repair copy.'}
 $doc.ActiveWindow.View.Type=3;$doc.ActiveWindow.View.ShowAll=$false;$doc.Repaginate()
 $targets=@('(2-35)','(4-6)','推论4.1','(B-57)')
 $locations=@();$pages=@()
 foreach($target in $targets){
  $r=$doc.Content.Duplicate;$r.Find.ClearFormatting()
  if(-not $r.Find.Execute($target)){throw ('Missing target '+$target)}
  $pn=$r.Information(3);$pages+=$pn;$locations+=@{text=$target;page=$pn}
 }
 foreach($pageNo in ($pages | Select-Object -Unique)){
  $page=$doc.ActiveWindow.Panes.Item(1).Pages.Item([int]$pageNo)
  [byte[]]$bits=$page.EnhMetaFileBits
  $stream=[System.IO.MemoryStream]::new($bits)
  $metafile=[System.Drawing.Imaging.Metafile]::new($stream)
  $bitmap=[System.Drawing.Bitmap]::new(1700,2404)
  $g=[System.Drawing.Graphics]::FromImage($bitmap)
  $g.Clear([System.Drawing.Color]::White);$g.DrawImage($metafile,0,0,1700,2404)
  $bitmap.Save((Join-Path $PreviewDirectory ('page_'+$pageNo.ToString('000')+'.png')),[System.Drawing.Imaging.ImageFormat]::Png)
  $g.Dispose();$bitmap.Dispose();$metafile.Dispose();$stream.Dispose()
 }
 @{read_only=$doc.ReadOnly;pages=$doc.ComputeStatistics(2);omath=$doc.OMaths.Count;locations=$locations;pdf_exported=$false} | ConvertTo-Json -Depth 5 | Out-File -Encoding utf8 (Join-Path $PreviewDirectory 'native_verification.json')
 Write-Output 'FINAL_NATIVE_PREVIEW_DONE'
 $doc.Close(0);$doc=$null;$word.Quit();$word=$null
}
finally {
 if($null -ne $doc){$doc.Close(0)}
 if($null -ne $word){$word.Quit()}
}
