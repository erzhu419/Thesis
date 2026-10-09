param([string]$DocumentPath,[string]$InfoPath)
$ErrorActionPreference='Stop'
[Console]::OutputEncoding=[Text.UTF8Encoding]::new($false)
$existing=@(Get-Process WINWORD -ErrorAction SilentlyContinue|ForEach-Object {$_.Id})
$staging=Join-Path $env:TEMP ('thesis-six-figures-'+[Guid]::NewGuid().ToString('N'))
[void](New-Item -ItemType Directory -Path $staging)
$localDoc=Join-Path $staging 'manuscript.docx'
Copy-Item -LiteralPath $DocumentPath -Destination $localDoc
$word=$null;$doc=$null;$owned=$false;$old=@{}
$info=[ordered]@{Status='starting';LocalStaging=$staging}
try {
 $word=New-Object -ComObject Word.Application
 $new=@(Get-Process WINWORD|Where-Object {$existing -notcontains $_.Id})
 if($new.Count -ne 1){throw 'Cannot identify the independent Word process.'}
 $owned=$true;$info.ProcessId=$new[0].Id;$info.Version=$word.Version;$info.Build=$word.Build
 foreach($name in @('UpdateLinksAtOpen','UpdateLinksAtPrint','UpdateFieldsAtPrint','Pagination')){
  $old[$name]=$word.Options.$name;$word.Options.$name=$false
 }
 $word.AutomationSecurity=3;$word.DisplayAlerts=0;$word.Visible=$false;$word.ScreenUpdating=$false
 $doc=$word.Documents.Open($localDoc,$false,$false,$false)
 $doc.PrintRevisions=$false
 $doc.Repaginate()
 foreach($toc in $doc.TablesOfContents){$toc.UpdatePageNumbers()}
 $doc.Repaginate()
 foreach($toc in $doc.TablesOfContents){$toc.UpdatePageNumbers()}
 $doc.Save()
 $info.Pages=$doc.ComputeStatistics(2);$info.Equations=$doc.OMaths.Count
 $info.Sections=$doc.Sections.Count;$info.Revisions=$doc.Revisions.Count;$info.Comments=$doc.Comments.Count
 $info.Locations=@()
 foreach($key in @('fig1_1','fig2_1','fig2_3','fig4_1','fig5_2','fig6_1')){
  $range=$doc.Bookmarks.Item($key).Range
  $shapes=$doc.Range(0,$range.Start).InlineShapes
  $shape=$shapes.Item($shapes.Count)
  $info.Locations += [ordered]@{Bookmark=$key;CaptionPage=$range.Information(3);ImagePage=$shape.Range.Information(3);AdjustedPage=$range.Information(1);WidthPoints=$shape.Width;HeightPoints=$shape.Height}
 }
 $info.Status='completed';$info.TocPageNumbersUpdated=$true
 Copy-Item -LiteralPath $localDoc -Destination $DocumentPath -Force
 Write-Output ('PAGINATION_COMPLETE pages='+$info.Pages)
} finally {
 if($null -ne $doc){$doc.Close([ref]0)}
 if($owned){foreach($name in $old.Keys){$word.Options.$name=$old[$name]};$word.Quit([ref]0)}
 [IO.File]::WriteAllText($InfoPath,($info|ConvertTo-Json -Depth 6),[Text.UTF8Encoding]::new($false))
}
