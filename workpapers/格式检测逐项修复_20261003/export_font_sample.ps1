param([string]$Source,[string]$Pdf,[string]$InfoPath,[string]$WorkerPidPath)
$ErrorActionPreference='Stop'
[Console]::OutputEncoding=[Text.UTF8Encoding]::new($false)
$before=@(Get-Process WINWORD -ErrorAction SilentlyContinue|ForEach-Object {$_.Id})
$word=$null;$doc=$null;$owned=$false;$old=@{};$failure=$null
$testFolder=Join-Path $env:TEMP ('thesis-font-test-'+[Guid]::NewGuid().ToString('N'))
[void](New-Item -ItemType Directory -Path $testFolder)
$testCopy=Join-Path $testFolder 'sample-source.docx'
Copy-Item -LiteralPath $Source -Destination $testCopy
$info=[ordered]@{Source=$Source;Copy=$testCopy;Status='starting'}
$timer=[Diagnostics.Stopwatch]::StartNew()
try {
 $word=New-Object -ComObject Word.Application
 $new=@(Get-Process WINWORD|Where-Object {$before -notcontains $_.Id})
 if($new.Count -ne 1){throw 'Independent Word process not identified.'}
 $owned=$true;$info.ProcessId=$new[0].Id
 [IO.File]::WriteAllText($WorkerPidPath,[string]$new[0].Id)
 foreach($name in @('UpdateLinksAtOpen','UpdateLinksAtPrint','UpdateFieldsAtPrint','Pagination')){
  $old[$name]=$word.Options.$name;$word.Options.$name=$false
 }
 $old.AutomationSecurity=$word.AutomationSecurity;$old.DisplayAlerts=$word.DisplayAlerts
 $word.AutomationSecurity=3;$word.DisplayAlerts=0;$word.Visible=$false;$word.ScreenUpdating=$false
 $doc=$word.Documents.Open($testCopy,$false,$true,$false)
 $info.OpenSeconds=$timer.Elapsed.TotalSeconds
 Write-Output ('OPENED seconds='+$info.OpenSeconds)
 $range=$doc.OMaths.Item(334).Range
 $start=$range.Information(3)
 if($start -lt 1){$doc.Repaginate();$start=$range.Information(3)}
 $info.FromPage=$start;$info.ToPage=$start+2
 $info.Math=[ordered]@{Text=[string]$range.Text;Size=$range.Font.Size;Ascii=$range.Font.NameAscii;FarEast=$range.Font.NameFarEast}
 [IO.File]::WriteAllText($InfoPath,($info|ConvertTo-Json -Depth 5),[Text.UTF8Encoding]::new($false))
 $exportTimer=[Diagnostics.Stopwatch]::StartNew()
 Write-Output ('EXPORTING range='+$start+'-'+($start+2))
 $doc.ExportAsFixedFormat($Pdf,17,$false,0,3,$start,($start+2),0,$false,$true,0,$false,$false,$false)
 $info.ExportSeconds=$exportTimer.Elapsed.TotalSeconds
 $info.TotalSeconds=$timer.Elapsed.TotalSeconds;$info.Status='completed'
} catch {$failure=$_;$info.Status='failed';$info.Error=$_.Exception.Message}
finally {
 if($null -ne $doc){$dontSave=0;$doc.Close([ref]$dontSave)}
 if(($null -ne $word)-and $owned){
  foreach($name in @('UpdateLinksAtOpen','UpdateLinksAtPrint','UpdateFieldsAtPrint','Pagination')){
   if($old.ContainsKey($name)){$word.Options.$name=$old[$name]}
  }
  if($old.ContainsKey('AutomationSecurity')){$word.AutomationSecurity=$old.AutomationSecurity}
  if($old.ContainsKey('DisplayAlerts')){$word.DisplayAlerts=$old.DisplayAlerts}
  $dontSave=0;$word.Quit([ref]$dontSave)
 }
 [IO.File]::WriteAllText($InfoPath,($info|ConvertTo-Json -Depth 5),[Text.UTF8Encoding]::new($false))
}
if($null -ne $failure){throw $failure}
Write-Output 'FONT_SAMPLE_STATUS=completed'
