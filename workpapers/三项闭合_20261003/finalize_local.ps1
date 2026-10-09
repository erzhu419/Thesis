param([string]$DocumentPath,[string]$PdfPath,[string]$InfoPath,[string]$FinalizeScript)
$ErrorActionPreference='Stop'
[Console]::OutputEncoding=[Text.UTF8Encoding]::new($false)
$work=Join-Path $env:TEMP ('thesis-cvar-'+[Guid]::NewGuid().ToString('N'))
[void](New-Item -ItemType Directory -Path $work)
$localDoc=Join-Path $work 'manuscript.docx'
$localPdf=Join-Path $work 'preview.pdf'
$localInfo=Join-Path $work 'word-info.json'
Copy-Item -LiteralPath $DocumentPath -Destination $localDoc
& $FinalizeScript -DocumentPath $localDoc -PdfPath $localPdf -InfoPath $localInfo
$info=Get-Content -LiteralPath $localInfo -Raw -Encoding UTF8 | ConvertFrom-Json
if($info.Status -ne 'completed'){throw 'Local Word finalization did not complete.'}
Copy-Item -LiteralPath $localDoc -Destination $DocumentPath -Force
Copy-Item -LiteralPath $localPdf -Destination $PdfPath -Force
$info | Add-Member -NotePropertyName DeliveryDocument -NotePropertyValue $DocumentPath
$info | Add-Member -NotePropertyName DeliveryPdf -NotePropertyValue $PdfPath
[IO.File]::WriteAllText($InfoPath,($info|ConvertTo-Json -Depth 5),[Text.UTF8Encoding]::new($false))
Write-Output ('LOCAL_STAGING='+$work)
