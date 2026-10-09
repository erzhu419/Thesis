param(
 [Parameter(Mandatory=$true)][string]$InputDocument,
 [Parameter(Mandatory=$true)][string]$OutputPdf,
 [Parameter(Mandatory=$true)][string]$InfoJson
)
$ErrorActionPreference='Stop'
[Console]::OutputEncoding=[Text.UTF8Encoding]::new($false)
$word=$null;$doc=$null;$owns=$false;$failure=$null;$old=@{}
$before=@(Get-Process WINWORD -ErrorAction SilentlyContinue|ForEach-Object {$_.Id})
$info=[ordered]@{Status='starting';InputDocument=$InputDocument;OutputPdf=$OutputPdf;Error=$null}
try {
 $word=New-Object -ComObject Word.Application
 $new=@(Get-Process WINWORD|Where-Object {($before -notcontains $_.Id)-and($_.Id -ne 45772)})
 if($new.Count -ne 1){throw 'No unique independent Word process.'}
 $owns=$true;$info.ProcessId=$new[0].Id;$info.Version=$word.Version
 foreach($n in @('UpdateLinksAtOpen','UpdateLinksAtPrint','UpdateFieldsAtPrint','Pagination')){
  $old[$n]=$word.Options.$n;$word.Options.$n=$false
 }
 $old.AutomationSecurity=$word.AutomationSecurity;$old.DisplayAlerts=$word.DisplayAlerts
 $word.AutomationSecurity=3;$word.DisplayAlerts=0;$word.Visible=$false;$word.ScreenUpdating=$false
 $doc=$word.Documents.Open($InputDocument,$false,$true,$false)
 $doc.Repaginate()
 $info.Pages=$doc.ComputeStatistics(2)
 Write-Output ('EXPORT_SAVED_DOCX pages='+$info.Pages)
 $doc.ExportAsFixedFormat($OutputPdf,17)
 $info.PagesAfterExport=$doc.ComputeStatistics(2)
 $info.Status='completed'
} catch {$failure=$_;$info.Status='failed';$info.Error=$_.Exception.Message}
finally {
 if($null -ne $doc){$dontSave=0;$doc.Close([ref]$dontSave)}
 if(($null -ne $word)-and$owns){
  foreach($n in @('UpdateLinksAtOpen','UpdateLinksAtPrint','UpdateFieldsAtPrint','Pagination')){
   if($old.ContainsKey($n)){$word.Options.$n=$old[$n]}
  }
  if($old.ContainsKey('AutomationSecurity')){$word.AutomationSecurity=$old.AutomationSecurity}
  if($old.ContainsKey('DisplayAlerts')){$word.DisplayAlerts=$old.DisplayAlerts}
  $dontSave=0;$word.Quit([ref]$dontSave)
 }
 [IO.File]::WriteAllText($InfoJson,($info|ConvertTo-Json -Depth 5),[Text.UTF8Encoding]::new($false))
}
if($null -ne $failure){throw $failure}
Write-Output ('FINAL_STATUS='+$info.Status)
