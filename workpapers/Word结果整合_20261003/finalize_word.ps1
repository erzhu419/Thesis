param(
 [Parameter(Mandatory=$true)][string]$DocumentPath,
 [Parameter(Mandatory=$true)][string]$PdfPath,
 [Parameter(Mandatory=$true)][string]$InfoPath
)
$ErrorActionPreference='Stop'
[Console]::OutputEncoding=[Text.UTF8Encoding]::new($false)
$before=@(Get-Process WINWORD -ErrorAction SilentlyContinue|ForEach-Object {$_.Id})
$word=$null;$doc=$null;$owned=$false;$failure=$null;$old=@{}
$info=[ordered]@{Status='starting';Document=$DocumentPath;Pdf=$PdfPath}
try {
 $word=New-Object -ComObject Word.Application
 $new=@(Get-Process WINWORD|Where-Object {$before -notcontains $_.Id})
 if($new.Count -ne 1){throw 'Cannot identify the independent Word process.'}
 $owned=$true;$info.ProcessId=$new[0].Id
 foreach($name in @('UpdateLinksAtOpen','UpdateLinksAtPrint','UpdateFieldsAtPrint','Pagination')){
  $old[$name]=$word.Options.$name;$word.Options.$name=$false
 }
 $old.AutomationSecurity=$word.AutomationSecurity;$old.DisplayAlerts=$word.DisplayAlerts
 $word.AutomationSecurity=3;$word.DisplayAlerts=0;$word.Visible=$false;$word.ScreenUpdating=$false
 $doc=$word.Documents.Open($DocumentPath,$false,$false,$false)
 Write-Output ('OPENED sections='+$doc.Sections.Count+' equations='+$doc.OMaths.Count)
 $info.Sections=$doc.Sections.Count;$info.Equations=$doc.OMaths.Count
 for($i=1;$i -le $doc.TablesOfContents.Count;$i++){
  $doc.TablesOfContents.Item($i).Update()
 }
 Write-Output 'TOC_UPDATED'
 $doc.Repaginate()
 for($i=1;$i -le $doc.TablesOfContents.Count;$i++){
  $doc.TablesOfContents.Item($i).UpdatePageNumbers()
 }
 foreach($section in $doc.Sections){
  foreach($header in $section.Headers){[void]$header.Range.Fields.Update()}
  foreach($footer in $section.Footers){[void]$footer.Range.Fields.Update()}
 }
 $doc.Repaginate()
 $doc.Save()
 $info.Pages=$doc.ComputeStatistics(2)
 Write-Output ('SAVED pages='+$info.Pages)
 $doc.ExportAsFixedFormat($PdfPath,17)
 $info.Status='completed'
} catch {$failure=$_;$info.Status='failed';$info.Error=$_.Exception.Message}
finally {
 if($null -ne $doc){$dontSave=0;$doc.Close([ref]$dontSave)}
 if(($null -ne $word)-and$owned){
  foreach($name in @('UpdateLinksAtOpen','UpdateLinksAtPrint','UpdateFieldsAtPrint','Pagination')){
   if($old.ContainsKey($name)){$word.Options.$name=$old[$name]}
  }
  $word.AutomationSecurity=$old.AutomationSecurity;$word.DisplayAlerts=$old.DisplayAlerts
  $dontSave=0;$word.Quit([ref]$dontSave)
 }
 [IO.File]::WriteAllText($InfoPath,($info|ConvertTo-Json -Depth 5),[Text.UTF8Encoding]::new($false))
}
if($null -ne $failure){throw $failure}
Write-Output 'FINAL_STATUS=completed'
