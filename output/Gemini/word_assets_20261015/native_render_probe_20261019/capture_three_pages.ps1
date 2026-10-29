param([string]$InputDocument,[string]$WorkDirectory)
$ErrorActionPreference='Stop'
$targets=Get-Content -Encoding utf8 -Raw (Join-Path $WorkDirectory 'probe_targets.json') | ConvertFrom-Json
$word=$null;$doc=$null
try {
  $word=New-Object -ComObject Word.Application
  if($word.Documents.Count -ne 0){throw "Expected isolated Word instance."}
  $word.Visible=$false;$word.DisplayAlerts=0
  $doc=$word.Documents.Open($InputDocument,$false,$true,$false)
  if(-not $doc.ReadOnly){throw 'Probe must use read-only Word document.'}
  $doc.ActiveWindow.View.Type=3
  $doc.ActiveWindow.View.ShowAll=$false
  $doc.Repaginate()
  $locations=@();$pages=@()
  foreach($t in $targets.targets){
    $r=$doc.OMaths.Item([int]$t.word_omath_index_1based).Range.Duplicate
    $r.Collapse(1)
    $pn=[int]$r.Information(3)
    $locations+=@{label=$t.label;page=$pn;word_omath_index_1based=$t.word_omath_index_1based}
    $pages+=$pn
  }
  foreach($pn in ($pages | Select-Object -Unique)){
    [byte[]]$bits=$doc.ActiveWindow.Panes.Item(1).Pages.Item([int]$pn).EnhMetaFileBits
    $emf=Join-Path $WorkDirectory ('page_'+$pn.ToString('000')+'.emf')
    [IO.File]::WriteAllBytes($emf,$bits)
  }
  @{source=$InputDocument;read_only=$doc.ReadOnly;math_count=$doc.OMaths.Count;page_count=$doc.ComputeStatistics(2);locations=$locations;emf_pages=($pages|Select-Object -Unique);word_visible=$word.Visible;pdf_generated=$false} |
    ConvertTo-Json -Depth 5 | Out-File -Encoding utf8 (Join-Path $WorkDirectory 'capture_probe.json')
  $doc.Close(0);$doc=$null;$word.Quit();$word=$null
} finally {
  if($null -ne $doc){$doc.Close(0)}
  if($null -ne $word){$word.Quit()}
}
# All image conversion happens after the hidden Word instance has closed.
foreach($pn in ($pages | Select-Object -Unique)){
  $stem=Join-Path $WorkDirectory ('page_'+$pn.ToString('000'))
  & (Join-Path $WorkDirectory 'raster_emf_compare.ps1') -InputEmf ($stem+'.emf') -OutputStem $stem
}
