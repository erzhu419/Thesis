param([string]$DocumentPath,[string]$InfoPath)
$ErrorActionPreference='Stop'
[Console]::OutputEncoding=[Text.UTF8Encoding]::new($false)
$before=@(Get-Process WINWORD -ErrorAction SilentlyContinue|ForEach-Object {$_.Id})
$work=Join-Path $env:TEMP ('thesis-toc-'+[Guid]::NewGuid().ToString('N'))
[void](New-Item -ItemType Directory -Path $work)
$localDoc=Join-Path $work 'manuscript.docx'
Copy-Item -LiteralPath $DocumentPath -Destination $localDoc
$word=$null;$doc=$null;$owned=$false;$old=@{};$info=[ordered]@{Status='starting';LocalStaging=$work}
try{
 $word=New-Object -ComObject Word.Application
 $new=@(Get-Process WINWORD|Where-Object {$before -notcontains $_.Id})
 if($new.Count -ne 1){throw 'Independent Word process could not be identified.'}
 $owned=$true;$info.ProcessId=$new[0].Id
 foreach($name in @('UpdateLinksAtOpen','UpdateLinksAtPrint','UpdateFieldsAtPrint','Pagination')){$old[$name]=$word.Options.$name;$word.Options.$name=$false}
 $word.AutomationSecurity=3;$word.DisplayAlerts=0;$word.Visible=$false;$word.ScreenUpdating=$false
 Write-Output ('WORD_PROCESS='+$info.ProcessId)
 $doc=$word.Documents.Open($localDoc,$false,$false,$false)
 Write-Output 'DOCUMENT_OPEN'
 $equationFields=@();$referenceFields=@()
 foreach($field in $doc.Fields){
  $code=$field.Code.Text
  if($code -match '^\s*SEQ\s+Eq2\b'){$equationFields+=,$field}
  elseif($code -match '^\s*REF\s+eq2_'){$referenceFields+=,$field}
 }
 foreach($field in $equationFields){[void]$field.Update()}
 foreach($field in $referenceFields){[void]$field.Update()}
 $info.Chapter2EquationsUpdated=$equationFields.Count
 $info.Chapter2ReferencesUpdated=$referenceFields.Count
 Write-Output 'CHAPTER_2_FIELDS_UPDATED'
 foreach($toc in $doc.TablesOfContents){$toc.Update()}
 Write-Output 'TOC_UPDATED'
 $doc.Save()
 $info.Status='saved';$info.Equations=$doc.OMaths.Count;$info.Comments=$doc.Comments.Count;$info.Revisions=$doc.Revisions.Count
 $info.TocEntries=$doc.TablesOfContents.Item(1).Range.Paragraphs.Count
}finally{
 if($null -ne $doc){$doc.Close([ref]0)}
 if($owned){foreach($name in $old.Keys){$word.Options.$name=$old[$name]};$word.Quit([ref]0)}
 [IO.File]::WriteAllText($InfoPath,($info|ConvertTo-Json),[Text.UTF8Encoding]::new($false))
}
if($info.Status -ne 'saved'){throw 'TOC update did not complete.'}
Copy-Item -LiteralPath $localDoc -Destination $DocumentPath -Force
Write-Output 'COMPLETED'
