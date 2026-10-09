param([string]$DocumentPath,[string]$TargetsPath,[string]$InfoPath,[string]$PreviewFolder,[switch]$FinalizeOnly,[string]$BaselineInfoPath)
$ErrorActionPreference='Stop'
[Console]::OutputEncoding=[Text.UTF8Encoding]::new($false)
$existing=@(Get-Process WINWORD -ErrorAction SilentlyContinue|ForEach-Object {$_.Id})
$dir=Join-Path $env:TEMP ('thesis-compact-'+[Guid]::NewGuid().ToString('N'))
[void](New-Item -ItemType Directory -Path $dir)
$local=Join-Path $dir 'manuscript.docx'
Copy-Item -LiteralPath $DocumentPath -Destination $local
$targets=Get-Content -LiteralPath $TargetsPath -Raw -Encoding UTF8|ConvertFrom-Json
$word=$null;$doc=$null;$owned=$false;$old=@{}
$info=[ordered]@{Status='starting';LocalStaging=$dir}
function Write-Info { [IO.File]::WriteAllText($InfoPath,($info|ConvertTo-Json -Depth 9),[Text.UTF8Encoding]::new($false)) }
function Caption-Range($Document,$Item,$StyleName) {
 $range=$Document.Bookmarks.Item('ThesisMainText').Range.Duplicate
 $range.Find.ClearFormatting();$range.Find.Style=[string]$targets.Styles.$StyleName;$range.Find.Format=$true
 $range.Find.Text=[string]$Item.Label+' ';$range.Find.Wrap=0
 $range.Find.Forward=$true;$range.Find.MatchWildcards=$false;$range.Find.MatchWholeWord=$false
 if(-not $range.Find.Execute()){throw ('Caption missing: '+$Item.Label)}
 return $range
}
function Read-Pages($Document) {
 $result=@();$pages=$Document.ActiveWindow.Panes.Item(1).Pages
 for($i=1;$i -le $pages.Count;$i++){
  $first=$null;$last=$null;$bottom=0;$imageCount=0;$paragraphs=@()
  foreach($rect in $pages.Item($i).Rectangles){
   $range=$rect.Range
   if($range.StoryType -ne 1){continue}
   $type=$rect.RectangleType;$top=$rect.Top;$height=$rect.Height
   $end=$top+$height
   if($end -gt $bottom){$bottom=$end}
   if($type -eq 1){$imageCount++}
   if($type -eq 0){
    $t=[string]$range.Text
    if($null -eq $first){$first=$t}
    $last=$t
    $paragraphs += [ordered]@{Start=$range.Start;End=$range.End;Top=$top;Height=$height;TextStart=$t.Substring(0,[Math]::Min(90,$t.Length))}
   }
  }
  $result += [ordered]@{Page=$i;Bottom=$bottom;Gap=[Math]::Max(0,($Document.PageSetup.PageHeight-$Document.PageSetup.BottomMargin)-$bottom);Images=$imageCount;First=$first;Last=$last;Blocks=$paragraphs}
  if($i%80 -eq 0){Write-Host ('GEOMETRY '+$i+'/'+$pages.Count)}
 }
 return ,$result
}
function Read-Tables($Document) {
 $result=@()
 foreach($item in $targets.Tables){
  $cap=Caption-Range $Document $item 'tablecaption'
  $table=$Document.Range($cap.End,$Document.Content.End).Tables.Item(1)
  $first=$table.Range.Duplicate;$first.Collapse(1)
  $last=$table.Range.Duplicate;$last.End--;$last.Collapse(0)
  $firstBody=$table.Rows.Item(2).Range.Duplicate;$firstBody.Collapse(1)
  $result += [ordered]@{Bookmark=$item.Bookmark;Caption=$item.Caption;CaptionPage=$cap.Information(3);StartPage=$first.Information(3);EndPage=$last.Information(3);FirstBodyPage=$firstBody.Information(3);Rows=$table.Rows.Count}
 }
 return ,$result
}
function Read-Figures($Document) {
 $result=@()
 foreach($item in $targets.Figures){
  $cap=Caption-Range $Document $item 'figurecaption'
  $shapes=$Document.Range(0,$cap.Start).InlineShapes
  $shape=$shapes.Item($shapes.Count)
  $result += [ordered]@{Bookmark=$item.Bookmark;CaptionPage=$cap.Information(3);ImagePage=$shape.Range.Information(3);Width=$shape.Width;Height=$shape.Height}
 }
 return ,$result
}
function Read-Chapters($Document) {
 $result=@()
 foreach($title in $targets.ChapterTitles){
  $range=$Document.Bookmarks.Item('ThesisMainText').Range.Duplicate
  $range.Find.ClearFormatting();$range.Find.Format=$false;$range.Find.Text=[string]$title
  $range.Find.Wrap=0;$range.Find.Forward=$true;$range.Find.MatchWildcards=$false
  if(-not $range.Find.Execute()){throw ('Chapter title missing: '+$title)}
  $result += [ordered]@{Title=$title;Page=$range.Information(3)}
 }
 return ,$result
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
 if($FinalizeOnly){
  $previous=Get-Content -LiteralPath $BaselineInfoPath -Raw -Encoding UTF8|ConvertFrom-Json
  foreach($key in @('BeforePages','BeforeTables','BeforeFigures','BeforeChapters','BeforeGeometry')){
   $info[$key]=$previous.$key
  }
 } else {
  $doc.Repaginate();$info.BeforePages=$doc.ComputeStatistics(2)
  $info.BeforeTables=Read-Tables $doc;$info.BeforeFigures=Read-Figures $doc
  $info.BeforeChapters=Read-Chapters $doc;$info.BeforeGeometry=Read-Pages $doc
  $info.Status='before_geometry_complete';Write-Info
  foreach($item in $targets.Tables){
   $cap=Caption-Range $doc $item 'tablecaption'
   $table=$doc.Range($cap.End,$doc.Content.End).Tables.Item(1)
   $table.Range.ParagraphFormat.KeepWithNext=0
   $table.Rows.AllowBreakAcrossPages=0
   $table.Rows.Item(1).HeadingFormat=-1
   $table.Rows.Item(1).Range.ParagraphFormat.KeepWithNext=-1
   $cap.Paragraphs.Item(1).Format.KeepWithNext=-1
  }
  $figureStyle=$doc.Styles.Item([string]$targets.Styles.figure).ParagraphFormat
  $figureStyle.LineUnitBefore=0;$figureStyle.SpaceBefore=4
  $captionStyle=$doc.Styles.Item([string]$targets.Styles.figurecaption).ParagraphFormat
  $captionStyle.LineUnitAfter=0;$captionStyle.SpaceAfter=4
  $tableStyle=$doc.Styles.Item([string]$targets.Styles.tablecaption).ParagraphFormat
  $tableStyle.LineUnitBefore=0;$tableStyle.SpaceBefore=6;$tableStyle.SpaceAfter=3
 }
 $doc.Repaginate()
 foreach($toc in $doc.TablesOfContents){$toc.UpdatePageNumbers()}
 $doc.Repaginate()
 foreach($toc in $doc.TablesOfContents){$toc.UpdatePageNumbers()}
 $info.AfterPages=$doc.ComputeStatistics(2)
 $info.AfterTables=Read-Tables $doc;$info.AfterFigures=Read-Figures $doc
 $info.AfterChapters=Read-Chapters $doc;$info.AfterGeometry=Read-Pages $doc
 $info.Equations=$doc.OMaths.Count;$info.Sections=$doc.Sections.Count
 $doc.Save();Copy-Item -LiteralPath $local -Destination $DocumentPath -Force
 $info.Status='saved';Write-Info
 Write-Output ('COMPACT_SAVED before='+$info.BeforePages+' after='+$info.AfterPages)
 Add-Type -AssemblyName System.Drawing
 [void](New-Item -ItemType Directory -Path $PreviewFolder -Force)
 $main=$doc.Bookmarks.Item('ThesisMainText').Range.Duplicate;$main.Collapse(1)
 $firstPage=$main.Information(3);$info.PreviewPages=@()
 $chapterPages=@($info.AfterChapters|ForEach-Object {$_.Page})
 $selected=[Collections.Generic.HashSet[int]]::new()
 foreach($p in $info.AfterGeometry){
  if($p.Page -ge $firstPage -and $p.Gap -ge 100 -and $chapterPages -notcontains ($p.Page+1)){
   [void]$selected.Add($p.Page)
   if($p.Page -lt $info.AfterPages){[void]$selected.Add($p.Page+1)}
  }
 }
 foreach($t in $info.AfterTables){
  if($t.StartPage -ne $t.EndPage){[void]$selected.Add($t.StartPage);[void]$selected.Add($t.EndPage)}
 }
 foreach($name in @('fig1_1','fig2_1','fig2_3','fig4_1','fig5_2','fig6_1')){
  $figure=$info.AfterFigures|Where-Object {$_.Bookmark -eq $name}
  [void]$selected.Add($figure.ImagePage)
 }
 $pages=$doc.ActiveWindow.Panes.Item(1).Pages
 foreach($n in @($selected|Sort-Object)){
  Render-Page $pages.Item($n) (Join-Path $PreviewFolder ('page_'+$n.ToString('000')+'.png'))
  $info.PreviewPages += $n
 }
 $info.Status='completed';Write-Output ('PREVIEWS_COMPLETE count='+$info.PreviewPages.Count)
} finally {
 if($null -ne $doc){$doc.Close([ref]0)}
 if($owned){foreach($name in $old.Keys){$word.Options.$name=$old[$name]};$word.Quit([ref]0)}
 Write-Info
}
