param([string]$InputDocument,[string]$WorkDirectory)
$ErrorActionPreference='Stop'
Add-Type -AssemblyName System.Drawing
$targets=Get-Content -Raw (Join-Path $WorkDirectory 'native_targets.json') | ConvertFrom-Json
$word=$null;$doc=$null
try {
 $word=New-Object -ComObject Word.Application
 $word.Visible=$false;$word.DisplayAlerts=0
 $doc=$word.Documents.Open($InputDocument,$false,$true)
 if(-not $doc.ReadOnly){throw 'Expected new repair document read-only.'}
 $doc.ActiveWindow.View.Type=3;$doc.ActiveWindow.View.ShowAll=$false;$doc.Repaginate()
 $pages=@();$samples=@()
 foreach($t in $targets.numbered){
  foreach($i in $t.math_indices){
   $r=$doc.OMaths.Item([int]$i).Range
   $pn=$r.Information(3);$pages+=$pn
   $samples+=@{label=$t.label;math_index=$i;page=$pn;noProofing=$r.NoProofing}
  }
 }
 foreach($i in $targets.inline){
  $r=$doc.OMaths.Item([int]$i).Range
  $samples+=@{label='inline';math_index=$i;noProofing=$r.NoProofing}
 }
 foreach($pageNo in ($pages | Select-Object -Unique)){
  $page=$doc.ActiveWindow.Panes.Item(1).Pages.Item([int]$pageNo)
  [byte[]]$bits=$page.EnhMetaFileBits
  $stream=[System.IO.MemoryStream]::new($bits)
  $metafile=[System.Drawing.Imaging.Metafile]::new($stream)
  $bitmap=[System.Drawing.Bitmap]::new(1700,2404)
  $g=[System.Drawing.Graphics]::FromImage($bitmap)
  $g.Clear([System.Drawing.Color]::White);$g.DrawImage($metafile,0,0,1700,2404)
  $bitmap.Save((Join-Path $WorkDirectory ('native_preview\page_'+$pageNo.ToString('000')+'.png')),[System.Drawing.Imaging.ImageFormat]::Png)
  $g.Dispose();$bitmap.Dispose();$metafile.Dispose();$stream.Dispose()
 }
 @{read_only=$doc.ReadOnly;omath=$doc.OMaths.Count;pages=$doc.ComputeStatistics(2);samples=$samples;pdf_exported=$false} | ConvertTo-Json -Depth 5 | Out-File -Encoding utf8 (Join-Path $WorkDirectory 'native_verification.json')
 Write-Output 'UPRIGHT_BRACKET_PREVIEW_DONE'
 $doc.Close(0);$doc=$null;$word.Quit();$word=$null
}
finally {
 if($null -ne $doc){$doc.Close(0)}
 if($null -ne $word){$word.Quit()}
}
