param([string]$InputDocument,[string]$PreviewDirectory)
$ErrorActionPreference='Stop'
$word=$null;$doc=$null
try {
 $word=New-Object -ComObject Word.Application
 $word.Visible=$false;$word.DisplayAlerts=0
 $doc=$word.Documents.Open($InputDocument,$false,$true)
 $out=Join-Path $env:TEMP 'thesis_minimal_font_test_20261019.pdf'
 Write-Output 'SAVEAS_STARTED'
 $doc.SaveAs2($out,17)
 Write-Output 'SAVEAS_COMPLETED'
 Copy-Item -LiteralPath $out -Destination (Join-Path $PreviewDirectory 'minimal_font_test.pdf') -Force
 $doc.Close(0);$doc=$null;$word.Quit();$word=$null
}
finally {
 if($null -ne $doc){$doc.Close(0)}
 if($null -ne $word){$word.Quit()}
}
