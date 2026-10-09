param([string]$DocumentPath,[string]$PdfPath,[string]$InfoPath)
$ErrorActionPreference='Stop'
[Console]::OutputEncoding=[Text.UTF8Encoding]::new($false)
$existing=@(Get-Process WINWORD -ErrorAction SilentlyContinue|ForEach-Object {$_.Id})
$staging=Join-Path $env:TEMP ('thesis-f2-link-'+[Guid]::NewGuid().ToString('N'))
[void](New-Item -ItemType Directory -Path $staging)
$localDoc=Join-Path $staging 'manuscript.docx'
$localPdf=Join-Path $staging 'preview.pdf'
Copy-Item -LiteralPath $DocumentPath -Destination $localDoc
$word=$null;$doc=$null;$owned=$false;$old=@{}
$info=[ordered]@{Status='starting';LocalStaging=$staging;Document=$DocumentPath;Pdf=$PdfPath}
function Save-Info {
 [IO.File]::WriteAllText($InfoPath,($info|ConvertTo-Json -Depth 5),[Text.UTF8Encoding]::new($false))
}
try {
 $word=New-Object -ComObject Word.Application
 $new=@(Get-Process WINWORD|Where-Object {$existing -notcontains $_.Id})
 if($new.Count -ne 1){throw 'Cannot identify the independent Word process.'}
 $owned=$true;$info.ProcessId=$new[0].Id;$info.Version=$word.Version;$info.Build=$word.Build
 Save-Info
 Write-Output ('WORD_PROCESS='+$info.ProcessId)
 foreach($name in @('UpdateLinksAtOpen','UpdateLinksAtPrint','UpdateFieldsAtPrint','Pagination')){
  $old[$name]=$word.Options.$name;$word.Options.$name=$false
 }
 $word.AutomationSecurity=3;$word.DisplayAlerts=0;$word.Visible=$false;$word.ScreenUpdating=$false
 $doc=$word.Documents.Open($localDoc,$false,$false,$false)
 $doc.PrintRevisions=$false
 $info.Status='opened';$info.Revisions=$doc.Revisions.Count;$info.Comments=$doc.Comments.Count
 $info.Equations=$doc.OMaths.Count;$info.Sections=$doc.Sections.Count
 Save-Info
 Write-Output 'DOCUMENT_OPEN'
 $doc.Repaginate()
 foreach($toc in $doc.TablesOfContents){$toc.UpdatePageNumbers()}
 $doc.Repaginate()
 foreach($toc in $doc.TablesOfContents){$toc.UpdatePageNumbers()}
 $doc.Save()
 Copy-Item -LiteralPath $localDoc -Destination $DocumentPath -Force
 $info.Pages=$doc.ComputeStatistics(2)
 $info.Status='saved_preview_pending';$info.TocPageNumbersUpdated=$true
 Save-Info
 Write-Output ('DOCUMENT_SAVED pages='+$info.Pages)
 # Export without document-structure tagging; all mathematical and visual
 # content is rendered by Word, while DOCX remains the editable deliverable.
 $doc.ExportAsFixedFormat($localPdf,17,$false,0,0,0,0,0,$true,$false,0,$false,$true,$false)
 Copy-Item -LiteralPath $localPdf -Destination $PdfPath -Force
 $info.Status='completed'
 Save-Info
 Write-Output 'PDF_EXPORTED'
} finally {
 if($null -ne $doc){$doc.Close([ref]0)}
 if($owned){foreach($name in $old.Keys){$word.Options.$name=$old[$name]};$word.Quit([ref]0)}
 Save-Info
}
