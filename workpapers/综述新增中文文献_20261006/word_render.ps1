param([string]$DocumentPath,[string]$InfoPath,[string]$ReviewPdf,[string]$ReferencesPdf)
$ErrorActionPreference='Stop'
[Console]::OutputEncoding=[Text.UTF8Encoding]::new($false)
$before=@(Get-Process WINWORD -ErrorAction SilentlyContinue|ForEach-Object {$_.Id})
$work=Join-Path $env:TEMP ('thesis-cnrefs-'+[Guid]::NewGuid().ToString('N'))
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
 $doc.Repaginate()
 foreach($toc in $doc.TablesOfContents){$toc.UpdatePageNumbers()}
 $doc.Save()
 $info.Pages=$doc.ComputeStatistics(2)
 $info.Equations=$doc.OMaths.Count;$info.Comments=$doc.Comments.Count;$info.Revisions=$doc.Revisions.Count
 $range=$doc.Content.Duplicate;$find=$range.Find;$find.ClearFormatting();$find.Text='公交控制嵌入在由线网设计';$find.Wrap=0
 if(-not $find.Execute()){throw 'Review start not found.'}
 $first=$range.Information(3)
 $range=$doc.Content.Duplicate;$find=$range.Find;$find.ClearFormatting();$find.Text='1.3 研究目标';$find.Wrap=0
 # The end anchor occurs in the last portion of 1.2.3; include the following page.
 $range=$doc.Content.Duplicate;$find=$range.Find;$find.ClearFormatting();$find.Text='世界模型的作用方式经历了由状态压缩';$find.Wrap=0
 if(-not $find.Execute()){throw 'Review end not found.'}
 $last=$range.Information(3)+1
 $rfirst=$doc.Bookmarks.Item('ref-cn2026-160').Range.Information(3)
 $rlast=$doc.Bookmarks.Item('ref-cn2026-209').Range.Information(3)
 $info.ReviewPages=@($first,$last);$info.NewReferencePages=@($rfirst,$rlast)
 $localReview=Join-Path $work 'review.pdf';$localReferences=Join-Path $work 'references.pdf'
 $doc.ExportAsFixedFormat($localReview,17,$false,0,3,$first,$last)
 $doc.ExportAsFixedFormat($localReferences,17,$false,0,3,$rfirst,$rlast)
 Copy-Item -LiteralPath $localReview -Destination $ReviewPdf -Force
 Copy-Item -LiteralPath $localReferences -Destination $ReferencesPdf -Force
 $info.Status='saved';Write-Output ('PAGES='+$info.Pages)
}finally{
 if($null -ne $doc){$doc.Close([ref]0)}
 if($owned){foreach($name in $old.Keys){$word.Options.$name=$old[$name]};$word.Quit([ref]0)}
 [IO.File]::WriteAllText($InfoPath,($info|ConvertTo-Json),[Text.UTF8Encoding]::new($false))
}
if($info.Status -ne 'saved'){throw 'Word pagination/rendering did not complete.'}
Copy-Item -LiteralPath $localDoc -Destination $DocumentPath -Force
Write-Output 'COMPLETED'
