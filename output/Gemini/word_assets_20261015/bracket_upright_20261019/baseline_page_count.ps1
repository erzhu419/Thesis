param([string]$InputDocument,[string]$WorkDirectory)
$ErrorActionPreference='Stop';$word=$null;$doc=$null
try {
 $word=New-Object -ComObject Word.Application;$word.Visible=$false;$word.DisplayAlerts=0
 $doc=$word.Documents.Open($InputDocument,$false,$true)
 if(-not $doc.ReadOnly){throw 'Expected baseline snapshot read-only.'}
 $doc.Repaginate()
 @{pages=$doc.ComputeStatistics(2);read_only=$doc.ReadOnly;omath=$doc.OMaths.Count} | ConvertTo-Json | Out-File -Encoding utf8 (Join-Path $WorkDirectory 'baseline_native_pagination.json')
 Write-Output ('BASELINE_PAGES='+$doc.ComputeStatistics(2))
 $doc.Close(0);$doc=$null;$word.Quit();$word=$null
}
finally{if($null -ne $doc){$doc.Close(0)};if($null -ne $word){$word.Quit()}}
