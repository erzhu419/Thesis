param([string]$DocumentPath,[string]$PaginationPath,[string]$ResultFolder,[string]$InfoPath)
$ErrorActionPreference='Stop'
[Console]::OutputEncoding=[Text.UTF8Encoding]::new($false)
$before=@(Get-Process WINWORD -ErrorAction SilentlyContinue|ForEach-Object {$_.Id})
$word=$null;$doc=$null;$owned=$false;$old=@{}
$info=[ordered]@{Status='starting';Files=@()}
function Write-Info { [IO.File]::WriteAllText($InfoPath,($info|ConvertTo-Json -Depth 5),[Text.UTF8Encoding]::new($false)) }
try {
 $word=New-Object -ComObject Word.Application
 $new=@(Get-Process WINWORD|Where-Object {$before -notcontains $_.Id})
 if($new.Count -ne 1){throw 'Independent Word process not identified.'}
 $owned=$true;$info.ProcessId=$new[0].Id;$info.Version=$word.Version;$info.Build=$word.Build
 foreach($name in @('UpdateLinksAtOpen','UpdateLinksAtPrint','UpdateFieldsAtPrint','Pagination')){
  $old[$name]=$word.Options.$name;$word.Options.$name=$false
 }
 $word.AutomationSecurity=3;$word.DisplayAlerts=0;$word.Visible=$false;$word.ScreenUpdating=$false
 $record=Get-Content -LiteralPath $PaginationPath -Raw -Encoding UTF8|ConvertFrom-Json
 $source=Join-Path $record.LocalStaging 'manuscript.docx'
 $doc=$word.Documents.Open($source,$false,$true,$false)
 $doc.PrintRevisions=$false
 $doc.Repaginate()
 foreach($location in $record.Locations){
  $name=$location.Bookmark
  $localPdf=Join-Path $record.LocalStaging ($name+'.pdf')
  $resultPdf=Join-Path $ResultFolder ($name+'.pdf')
  $from=[Math]::Min($location.ImagePage,$location.CaptionPage)
  $to=[Math]::Max($location.ImagePage,$location.CaptionPage)
  $info.Status='exporting';$info.Current=$name;Write-Info
  Write-Output ('EXPORT '+$name+' pages='+$from+'-'+$to)
  $doc.ExportAsFixedFormat($localPdf,17,$false,0,3,$from,$to,0,$false,$true,0,$false,$false,$false)
  Copy-Item -LiteralPath $localPdf -Destination $resultPdf
  $info.Files += [ordered]@{Figure=$name;From=$from;To=$to;Pdf=$resultPdf}
  Write-Info
 }
 $info.Status='completed';Write-Output 'PREVIEW_COMPLETE'
} finally {
 if($null -ne $doc){$doc.Close([ref]0)}
 if($owned){foreach($name in $old.Keys){$word.Options.$name=$old[$name]};$word.Quit([ref]0)}
 Write-Info
}
