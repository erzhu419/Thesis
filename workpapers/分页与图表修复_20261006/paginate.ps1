param([string]$DocumentPath,[string]$PdfPath,[string]$InfoPath)
$ErrorActionPreference='Stop'
[Console]::OutputEncoding=[Text.UTF8Encoding]::new($false)
$before=@(Get-Process WINWORD -ErrorAction SilentlyContinue|ForEach-Object {$_.Id})
$word=$null;$doc=$null;$owned=$false;$failure=$null;$old=@{}
$work=Join-Path $env:TEMP ('thesis-format-'+[Guid]::NewGuid().ToString('N'))
[void](New-Item -ItemType Directory -Path $work)
$localDoc=Join-Path $work 'manuscript.docx'
$localPdf=Join-Path $work 'preview.pdf'
Copy-Item -LiteralPath $DocumentPath -Destination $localDoc
$info=[ordered]@{Status='starting';Document=$DocumentPath;Pdf=$PdfPath;LocalStaging=$work}
function Format-Toc($Document){
 for($level=1;$level -le 4;$level++){
  $style=$Document.Styles.Item(-19-$level)
  $style.Font.Size=12;$style.Font.NameAscii='Times New Roman';$style.Font.NameOther='Times New Roman'
  if($level -eq 1){$style.Font.NameFarEast=[string][char]0x9ED1+[char]0x4F53}
  else{$style.Font.NameFarEast=[string][char]0x5B8B+[char]0x4F53}
  $f=$style.ParagraphFormat
  $f.LineSpacingRule=4;$f.LineSpacing=20;$f.SpaceBeforeAuto=0;$f.SpaceAfterAuto=0
  $f.SpaceBefore=0;$f.SpaceAfter=0
  if($level -eq 1){$f.SpaceBefore=6;$f.SpaceAfter=6}
  $f.CharacterUnitLeftIndent=0;$f.CharacterUnitFirstLineIndent=0
  $f.FirstLineIndent=0;$f.LeftIndent=10.5*($level-1);$f.RightIndent=0
 }
 foreach($toc in $Document.TablesOfContents){
  $range=$toc.Range
  $range.Font.Size=12;$range.Font.NameAscii='Times New Roman';$range.Font.NameOther='Times New Roman'
  $range.ParagraphFormat.LineSpacingRule=4;$range.ParagraphFormat.LineSpacing=20
  $range.ParagraphFormat.TabStops.ClearAll()
  [void]$range.ParagraphFormat.TabStops.Add(414.8,2,1)
  $closing=$Document.Range($range.End,$range.End+1).Paragraphs.Item(1)
  if(([string]$closing.Range.Text -match '^\s*$') -and ([string]$closing.Range.WordOpenXML -match 'fldCharType="end"')){
   $closing.Range.Font.Size=1
   $closing.Format.LineSpacingRule=4;$closing.Format.LineSpacing=1
   $closing.Format.SpaceBeforeAuto=0;$closing.Format.SpaceAfterAuto=0
   $closing.Format.SpaceBefore=0;$closing.Format.SpaceAfter=0;$closing.Format.KeepWithNext=0
  }
 }
}
try{
 $word=New-Object -ComObject Word.Application
 $new=@(Get-Process WINWORD|Where-Object {$before -notcontains $_.Id})
 if($new.Count -ne 1){throw 'Cannot identify the independent Word process.'}
 $owned=$true;$info.ProcessId=$new[0].Id;$info.Version=$word.Version;$info.Build=$word.Build
 $info.FileVersion=(Get-Item (Join-Path $word.Path 'WINWORD.EXE')).VersionInfo.FileVersion
 $fonts=@();foreach($font in $word.FontNames){$fonts += [string]$font}
 $info.AvailableFonts=@($fonts|Sort-Object -Unique)
 foreach($name in @('UpdateLinksAtOpen','UpdateLinksAtPrint','UpdateFieldsAtPrint','Pagination')){
  $old[$name]=$word.Options.$name;$word.Options.$name=$false
 }
 $old.AutomationSecurity=$word.AutomationSecurity;$old.DisplayAlerts=$word.DisplayAlerts
 $word.AutomationSecurity=3;$word.DisplayAlerts=0;$word.Visible=$false;$word.ScreenUpdating=$false
 $doc=$word.Documents.Open($localDoc,$false,$false,$false)
 $info.Sections=$doc.Sections.Count;$info.Equations=$doc.OMaths.Count
 $rounds=@();$stable=$false
 for($round=1;$round -le 4;$round++){
  foreach($toc in $doc.TablesOfContents){$toc.Update()}
  Format-Toc $doc
  $doc.Repaginate()
  foreach($toc in $doc.TablesOfContents){$toc.UpdatePageNumbers()}
  foreach($section in $doc.Sections){
   foreach($part in @($section.Headers)+@($section.Footers)){
    foreach($field in $part.Range.Fields){
     if($field.Code.Text -match '^\s*(PAGE|NUMPAGES|SECTIONPAGES|PAGEREF|REF|STYLEREF)\b'){[void]$field.Update()}
    }
   }
  }
  Format-Toc $doc
  $doc.Repaginate()
  $pages=$doc.ComputeStatistics(2);$tocText=$doc.TablesOfContents.Item(1).Range.Text
  $rounds += [ordered]@{Round=$round;Pages=$pages;TocParagraphs=$doc.TablesOfContents.Item(1).Range.Paragraphs.Count}
  Write-Output ('PAGINATION round='+$round+' pages='+$pages)
  if($round -gt 1 -and $pages -eq $lastPages -and $tocText -eq $lastToc){$stable=$true;break}
  $lastPages=$pages;$lastToc=$tocText
 }

 $tableSpans=@()
 $indices=Get-Content -LiteralPath (Join-Path (Split-Path $InfoPath) 'table_indices.json') -Raw -Encoding UTF8|ConvertFrom-Json
 foreach($item in $indices){
  $t=$doc.Tables.Item($item.index);$r=$t.Range.Duplicate;$r.Collapse(1)
  $sp=$r.Information(3);$sy=$r.Information(6)
  $r=$t.Range.Duplicate;$r.End=$r.End-1;$r.Collapse(0)
  $ep=$r.Information(3);$ey=$r.Information(6)
  $tableSpans += [ordered]@{Index=$item.index;Caption=$item.caption;StartPage=$sp;StartY=$sy;EndPage=$ep;EndY=$ey}
 }
 $info.TableSpans=$tableSpans

 $info.PaginationRounds=$rounds;$info.Stable=$stable
 if(-not $stable){throw 'Pagination and TOC did not stabilize within four passes.'}
 $doc.Save();$info.Pages=$pages;$info.Status='document_saved_preview_pending'
 [IO.File]::WriteAllText($InfoPath,($info|ConvertTo-Json -Depth 8),[Text.UTF8Encoding]::new($false))
 $info.Status='saved'
}catch{$failure=$_;$info.Status='failed';$info.Error=$_.Exception.Message}
finally{
 if($null -ne $doc){$dontSave=0;$doc.Close([ref]$dontSave)}
 if(($null -ne $word)-and $owned){
  foreach($name in @('UpdateLinksAtOpen','UpdateLinksAtPrint','UpdateFieldsAtPrint','Pagination')){
   if($old.ContainsKey($name)){$word.Options.$name=$old[$name]}
  }
  $word.AutomationSecurity=$old.AutomationSecurity;$word.DisplayAlerts=$old.DisplayAlerts
  $dontSave=0;$word.Quit([ref]$dontSave)
 }
 [IO.File]::WriteAllText($InfoPath,($info|ConvertTo-Json -Depth 8),[Text.UTF8Encoding]::new($false))
}
if($null -ne $failure){throw $failure}
Copy-Item -LiteralPath $localDoc -Destination $DocumentPath -Force

Write-Output 'FINAL_STATUS=completed'
