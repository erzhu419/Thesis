param([string]$DocumentPath,[string]$NotesPath,[string]$InfoPath,[string]$PreviewFolder)
$ErrorActionPreference='Stop'
[Console]::OutputEncoding=[Text.UTF8Encoding]::new($false)
$existing=@(Get-Process WINWORD -ErrorAction SilentlyContinue|ForEach-Object {$_.Id})
$staging=Join-Path $env:TEMP ('thesis-code-links-'+[Guid]::NewGuid().ToString('N'))
[void](New-Item -ItemType Directory -Path $staging)
$local=Join-Path $staging 'manuscript.docx'
Copy-Item -LiteralPath $DocumentPath -Destination $local
$notes=Get-Content -LiteralPath $NotesPath -Raw -Encoding UTF8|ConvertFrom-Json
$word=$null;$doc=$null;$owned=$false;$old=@{}
$info=[ordered]@{Status='starting';LocalStaging=$staging;Notes=@();References=@();PreviewPages=@()}
function Find-Text($Text) {
 $range=$doc.Bookmarks.Item('ThesisMainText').Range.Duplicate
 $range.Find.ClearFormatting();$range.Find.Format=$false
 $range.Find.Text=[string]$Text;$range.Find.Wrap=0;$range.Find.MatchWildcards=$false
 if(-not $range.Find.Execute()){throw ('Text not found: '+$Text)}
 return $range
}
function Render-Page($Page,$Path) {
 $bits=[byte[]]$Page.EnhMetaFileBits
 $stream=[IO.MemoryStream]::new($bits,$false)
 $metafile=[Drawing.Imaging.Metafile]::new($stream)
 $bitmap=[Drawing.Bitmap]::new(1191,1684)
 $graphics=[Drawing.Graphics]::FromImage($bitmap)
 $graphics.Clear([Drawing.Color]::White)
 $graphics.DrawImage($metafile,[Drawing.Rectangle]::new(0,0,1191,1684))
 $bitmap.Save($Path,[Drawing.Imaging.ImageFormat]::Png)
 $graphics.Dispose();$bitmap.Dispose();$metafile.Dispose();$stream.Dispose()
}
try {
 $word=New-Object -ComObject Word.Application
 $new=@(Get-Process WINWORD|Where-Object {$existing -notcontains $_.Id})
 if($new.Count -ne 1){throw 'Cannot identify independent Word process.'}
 $owned=$true;$info.ProcessId=$new[0].Id;$info.Version=$word.Version;$info.Build=$word.Build
 foreach($name in @('UpdateLinksAtOpen','UpdateLinksAtPrint','UpdateFieldsAtPrint','Pagination')){
  $old[$name]=$word.Options.$name;$word.Options.$name=$false
 }
 $word.AutomationSecurity=3;$word.DisplayAlerts=0;$word.Visible=$false
 $doc=$word.Documents.Open($local,$false,$false,$false)
 $doc.PrintRevisions=$false;$doc.ActiveWindow.View.Type=3
 foreach($note in $notes){
  $range=Find-Text $note.text
  $range.Paragraphs.Item(1).Format.Alignment=0
 }
 foreach($key in @('ref-categorical-code','ref-resac-code','ref-bapr-code','ref-bus-h2o-code','ref-mcwm-code')){
  $doc.Bookmarks.Item($key).Range.Paragraphs.Item(1).Format.Alignment=0
 }
 $doc.Repaginate()
 foreach($toc in $doc.TablesOfContents){$toc.UpdatePageNumbers()}
 $doc.Repaginate()
 foreach($toc in $doc.TablesOfContents){$toc.UpdatePageNumbers()}
 $info.Pages=$doc.ComputeStatistics(2);$info.Equations=$doc.OMaths.Count;$info.Sections=$doc.Sections.Count
 $selected=[Collections.Generic.HashSet[int]]::new()
 foreach($note in $notes){
  $range=Find-Text $note.text
  $para=$range.Paragraphs.Item(1).Range.Duplicate
  $end=$para.Duplicate;$end.End--;$end.Collapse(0)
  $from=$range.Information(3);$to=$end.Information(3)
  $info.Notes += [ordered]@{Chapter=$note.chapter;From=$from;To=$to;Links=$para.Hyperlinks.Count}
  for($n=$from;$n -le $to;$n++){[void]$selected.Add($n)}
 }
 foreach($key in @('ref-categorical-code','ref-resac-code','ref-bapr-code','ref-bus-h2o-code','ref-mcwm-code')){
  $range=$doc.Bookmarks.Item($key).Range
  $from=$range.Information(3);$end=$range.Duplicate;$end.Collapse(0);$to=$end.Information(3)
  $info.References += [ordered]@{Bookmark=$key;From=$from;To=$to;Links=$range.Hyperlinks.Count}
  for($n=$from;$n -le $to;$n++){[void]$selected.Add($n)}
 }
 $doc.Save();Copy-Item -LiteralPath $local -Destination $DocumentPath -Force
 Write-Output ('SAVED pages='+$info.Pages)
 Add-Type -AssemblyName System.Drawing
 [void](New-Item -ItemType Directory -Path $PreviewFolder -Force)
 $pages=$doc.ActiveWindow.Panes.Item(1).Pages
 foreach($n in @($selected|Sort-Object)){
  Render-Page $pages.Item($n) (Join-Path $PreviewFolder ('page_'+$n.ToString('000')+'.png'))
  $info.PreviewPages += $n
 }
 $info.Status='completed';$info.TocPageNumbersUpdated=$true
 Write-Output ('PREVIEWS '+$info.PreviewPages.Count)
} finally {
 if($null -ne $doc){$doc.Close([ref]0)}
 if($owned){foreach($name in $old.Keys){$word.Options.$name=$old[$name]};$word.Quit([ref]0)}
 [IO.File]::WriteAllText($InfoPath,($info|ConvertTo-Json -Depth 7),[Text.UTF8Encoding]::new($false))
}
