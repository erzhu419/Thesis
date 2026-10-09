param([string]$Source,[string]$LocalPdf,[string]$ResultPdf,[string]$InfoPath)
$ErrorActionPreference='Stop'
[Console]::OutputEncoding=[Text.UTF8Encoding]::new($false)
$before=@(Get-Process WINWORD -ErrorAction SilentlyContinue | ForEach-Object {$_.Id})
$word=$null; $doc=$null; $owned=$false; $failure=$null; $old=@{}
$timer=[Diagnostics.Stopwatch]::StartNew()
$info=[ordered]@{Status='starting';Source=$Source;RequestedReadOnly=$true;SaveCalled=$false;TocUpdateCalled=$false;ContentOrFormatChangesCalled=$false;LocalPdf=$LocalPdf;ResultPdf=$ResultPdf;Started=(Get-Date).ToString('o')}
function Write-Info { [IO.File]::WriteAllText($InfoPath,($info | ConvertTo-Json -Depth 5),[Text.UTF8Encoding]::new($false)) }
try {
 $word=New-Object -ComObject Word.Application
 $new=@(Get-Process WINWORD | Where-Object {$before -notcontains $_.Id})
 if($new.Count -ne 1){throw 'Could not identify one newly owned Word process.'}
 $owned=$true
 $info.ProcessId=$new[0].Id; $info.Version=$word.Version; $info.Build=$word.Build
 $info.FileVersion=(Get-Item (Join-Path $word.Path 'WINWORD.EXE')).VersionInfo.FileVersion
 foreach($name in @('UpdateLinksAtOpen','UpdateLinksAtPrint','UpdateFieldsAtPrint','Pagination')){
  $old[$name]=$word.Options.$name; $word.Options.$name=$false
 }
 $old.AutomationSecurity=$word.AutomationSecurity; $old.DisplayAlerts=$word.DisplayAlerts; $old.ScreenUpdating=$word.ScreenUpdating
 $word.AutomationSecurity=3; $word.DisplayAlerts=0; $word.Visible=$false; $word.ScreenUpdating=$false
 $info.Status='opening'; Write-Info
 Write-Output ('FRESH_WORD_PID='+$info.ProcessId)
 $doc=$word.Documents.Open($Source,$false,$true,$false)
 $info.OpenedReadOnly=$doc.ReadOnly
 $info.DocumentsCount=$word.Documents.Count
 $info.FullName=$doc.FullName
 $info.OpenSeconds=$timer.Elapsed.TotalSeconds
 $doc.Repaginate()
 $info.Pages=$doc.ComputeStatistics(2)
 $info.BeforeExportSeconds=$timer.Elapsed.TotalSeconds
 $info.ExportStarted=(Get-Date).ToString('o'); $info.Status='exporting'; Write-Info
 Write-Output ('FRESH_EXPORT_START pages='+$info.Pages+' elapsed='+$info.BeforeExportSeconds)
 $exportTimer=[Diagnostics.Stopwatch]::StartNew()
 $doc.ExportAsFixedFormat($LocalPdf,17)
 $exportTimer.Stop()
 $info.ExportSeconds=$exportTimer.Elapsed.TotalSeconds
 $info.Status='completed'; $info.PdfBytes=(Get-Item -LiteralPath $LocalPdf).Length
 Copy-Item -LiteralPath $LocalPdf -Destination $ResultPdf -Force
 Write-Output ('FRESH_EXPORT_COMPLETE seconds='+$info.ExportSeconds+' bytes='+$info.PdfBytes)
} catch {
 $failure=$_; $info.Status='failed'; $info.Error=$_.Exception.Message
} finally {
 if($null -ne $doc){$dontSave=0; $doc.Close([ref]$dontSave)}
 if(($null -ne $word) -and $owned){
  foreach($name in @('UpdateLinksAtOpen','UpdateLinksAtPrint','UpdateFieldsAtPrint','Pagination')){
   if($old.ContainsKey($name)){$word.Options.$name=$old[$name]}
  }
  foreach($name in @('AutomationSecurity','DisplayAlerts','ScreenUpdating')){
   if($old.ContainsKey($name)){$word.$name=$old[$name]}
  }
  $dontSave=0; $word.Quit([ref]$dontSave)
 }
 $timer.Stop(); $info.TotalSeconds=$timer.Elapsed.TotalSeconds; $info.Finished=(Get-Date).ToString('o'); Write-Info
}
if($null -ne $failure){throw $failure}
