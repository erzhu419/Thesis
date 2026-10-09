param([string]$DocumentPath,[string]$PdfPath,[string]$InfoPath)
$ErrorActionPreference='Stop'
[Console]::OutputEncoding=[Text.UTF8Encoding]::new($false)
$before=@(Get-Process WINWORD -ErrorAction SilentlyContinue|ForEach-Object {$_.Id})
$word=$null;$doc=$null;$owned=$false;$failure=$null;$old=@{}
$info=[ordered]@{Status='starting';Document=$DocumentPath;Pdf=$PdfPath}
try {
 $word=New-Object -ComObject Word.Application
 $new=@(Get-Process WINWORD|Where-Object {$before -notcontains $_.Id})
 if($new.Count -ne 1){throw 'Cannot identify the independent Word process.'}
 $owned=$true
 $info.ProcessId=$new[0].Id;$info.Version=$word.Version;$info.Build=$word.Build
 $info.FileVersion=(Get-Item (Join-Path $word.Path 'WINWORD.EXE')).VersionInfo.FileVersion
 $fonts=@();foreach($font in $word.FontNames){$fonts += [string]$font}
 $info.AvailableFonts=@($fonts|Sort-Object -Unique)
 foreach($name in @('UpdateLinksAtOpen','UpdateLinksAtPrint','UpdateFieldsAtPrint','Pagination')){
  $old[$name]=$word.Options.$name;$word.Options.$name=$false
 }
 $old.AutomationSecurity=$word.AutomationSecurity;$old.DisplayAlerts=$word.DisplayAlerts
 $word.AutomationSecurity=3;$word.DisplayAlerts=0;$word.Visible=$false;$word.ScreenUpdating=$false
 $doc=$word.Documents.Open($DocumentPath,$false,$false,$false)
 $info.Sections=$doc.Sections.Count;$info.Equations=$doc.OMaths.Count
 $rounds=@();$stable=$false
 for($round=1;$round -le 3;$round++){
  foreach($toc in $doc.TablesOfContents){$toc.Update()}
  $doc.Repaginate()
  foreach($toc in $doc.TablesOfContents){$toc.UpdatePageNumbers()}
  foreach($section in $doc.Sections){
   foreach($part in @($section.Headers)+@($section.Footers)){
    foreach($field in $part.Range.Fields){
     if($field.Code.Text -match '^\s*(PAGE|NUMPAGES|SECTIONPAGES|PAGEREF|REF|STYLEREF)\b'){
      [void]$field.Update()
     }
    }
   }
  }
  $doc.Repaginate()
  $pages=$doc.ComputeStatistics(2)
  $tocText=($doc.TablesOfContents.Item(1).Range.Text)
  $rounds += [ordered]@{Round=$round;Pages=$pages}
  Write-Output ('PAGINATION round='+$round+' pages='+$pages)
  if($round -gt 1 -and $pages -eq $lastPages -and $tocText -eq $lastToc){$stable=$true;break}
  $lastPages=$pages;$lastToc=$tocText
 }
 $info.PaginationRounds=$rounds;$info.Stable=$stable
 if(-not $stable){throw 'Pagination/TOC did not stabilize within three passes.'}
 $doc.Save()
 $info.Pages=$pages
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
