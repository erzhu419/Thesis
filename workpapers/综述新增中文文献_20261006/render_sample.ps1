param([string]$InputDoc,[string]$OutputPdf)
$ErrorActionPreference='Stop'
$before=@(Get-Process WINWORD -ErrorAction SilentlyContinue|ForEach-Object {$_.Id})
$dir=Join-Path $env:TEMP ('thesis-refsample-'+[Guid]::NewGuid().ToString('N'))
[void](New-Item -ItemType Directory -Path $dir)
$file=Join-Path $dir 'sample.docx';$pdf=Join-Path $dir 'sample.pdf'
Copy-Item -LiteralPath $InputDoc -Destination $file
$word=$null;$doc=$null;$owned=$false
try{
 $word=New-Object -ComObject Word.Application
 $new=@(Get-Process WINWORD|Where-Object {$before -notcontains $_.Id})
 if($new.Count -ne 1){throw 'Unable to identify isolated Word process.'}
 $owned=$true;Write-Output ('SAMPLE_PROCESS='+$new[0].Id)
 $word.AutomationSecurity=3;$word.DisplayAlerts=0;$word.Visible=$false
 $doc=$word.Documents.Open($file,$false,$true,$false)
 $doc.ExportAsFixedFormat($pdf,17)
 Write-Output 'SAMPLE_EXPORTED'
}finally{
 if($null -ne $doc){$doc.Close([ref]0)}
 if($owned){$word.Quit([ref]0)}
}
Copy-Item -LiteralPath $pdf -Destination $OutputPdf -Force
