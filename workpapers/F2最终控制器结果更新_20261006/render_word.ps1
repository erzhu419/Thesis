param([string]$DocumentPath,[string]$PdfPath,[string]$InfoPath,[switch]$UpdateToc)
$ErrorActionPreference='Stop'
[Console]::OutputEncoding=[Text.UTF8Encoding]::new($false)
$before=@(Get-Process WINWORD -ErrorAction SilentlyContinue|ForEach-Object {$_.Id})
$work=Join-Path $env:TEMP ('thesis-f2-'+[Guid]::NewGuid().ToString('N'))
[void](New-Item -ItemType Directory -Path $work)
$localDoc=Join-Path $work 'manuscript.docx'
$localPdf=Join-Path $work 'preview.pdf'
Copy-Item -LiteralPath $DocumentPath -Destination $localDoc
$word=$null;$doc=$null;$owned=$false;$old=@{}
$info=[ordered]@{Status='starting';LocalStaging=$work}
try{
 $word=New-Object -ComObject Word.Application
 $new=@(Get-Process WINWORD|Where-Object {$before -notcontains $_.Id})
 if($new.Count -ne 1){throw 'Independent Word process could not be identified.'}
 $owned=$true;$info.ProcessId=$new[0].Id
 [IO.File]::WriteAllText($InfoPath,($info|ConvertTo-Json),[Text.UTF8Encoding]::new($false))
 foreach($name in @('UpdateLinksAtOpen','UpdateLinksAtPrint','UpdateFieldsAtPrint','Pagination')){
  $old[$name]=$word.Options.$name;$word.Options.$name=$false
 }
 $word.AutomationSecurity=3;$word.DisplayAlerts=0;$word.Visible=$false;$word.ScreenUpdating=$false
 Write-Output ('WORD_PROCESS='+$info.ProcessId)
 $doc=$word.Documents.Open($localDoc,$false,(-not $UpdateToc),$false)
 Write-Output 'DOCUMENT_OPEN'
 $doc.Repaginate()
 Write-Output 'PAGINATED'
 if($UpdateToc){
  foreach($toc in $doc.TablesOfContents){$toc.UpdatePageNumbers()}
  $doc.Save()
  Write-Output 'TOC_PAGES_UPDATED'
 }
 $doc.ExportAsFixedFormat($localPdf,17)
 $info.Pages=$doc.ComputeStatistics(2)
 $info.Status='completed';$info.TocPagesUpdated=[bool]$UpdateToc
 Copy-Item -LiteralPath $localPdf -Destination $PdfPath -Force
}finally{
 if($null -ne $doc){$doc.Close([ref]0)}
 if($owned){foreach($name in $old.Keys){$word.Options.$name=$old[$name]};$word.Quit([ref]0)}
 [IO.File]::WriteAllText($InfoPath,($info|ConvertTo-Json),[Text.UTF8Encoding]::new($false))
}
if($info.Status -ne 'completed'){throw 'Word rendering did not complete.'}
if($UpdateToc){Copy-Item -LiteralPath $localDoc -Destination $DocumentPath -Force}
Write-Output ('COMPLETED pages='+$info.Pages)
