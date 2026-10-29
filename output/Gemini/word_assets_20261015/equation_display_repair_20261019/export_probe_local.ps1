param([string]$InputDocument,[string]$PreviewDirectory)
$ErrorActionPreference='Stop'
$word=$null;$doc=$null
try {
 $word=New-Object -ComObject Word.Application
 $word.Visible=$true;$word.DisplayAlerts=0;$word.Options.PrintBackground=$false
 $doc=$word.Documents.Open($InputDocument,$false,$true)
 if(-not $doc.ReadOnly){throw 'Expected read-only temporary probe.'}
 $sample=$word.Documents.Add()
 $sample.Content.FormattedText=$doc.Range(0,$doc.Content.End-1).FormattedText
 $doc.Close(0);$doc=$sample
 $out=Join-Path $env:TEMP 'thesis_symbol_font_test_20261019.pdf'
 Write-Output 'EXPORT_STARTED'
 $doc.ExportAsFixedFormat($out,17,$false)
 Write-Output 'EXPORT_COMPLETED'
 Copy-Item -LiteralPath $out -Destination (Join-Path $PreviewDirectory 'membership_font_test.pdf') -Force
 $doc.Close(0);$doc=$null;$word.Quit();$word=$null
}
finally {
 if($null -ne $doc){$doc.Close(0)}
 if($null -ne $word){$word.Quit()}
}
