param([string]$Source,[string]$InfoPath,[string]$PreviewPath)
$ErrorActionPreference='Stop'
[Console]::OutputEncoding=[Text.UTF8Encoding]::new($false)
$existing=@(Get-Process WINWORD -ErrorAction SilentlyContinue|ForEach-Object {$_.Id})
$dir=Join-Path $env:TEMP ('thesis-layout-probe-'+[Guid]::NewGuid().ToString('N'))
[void](New-Item -ItemType Directory -Path $dir)
$local=Join-Path $dir 'source.docx'
Copy-Item -LiteralPath $Source -Destination $local
$word=$null;$doc=$null;$owned=$false;$old=@{}
$info=[ordered]@{Status='starting';LocalStaging=$dir}
function Write-Info { [IO.File]::WriteAllText($InfoPath,($info|ConvertTo-Json -Depth 8),[Text.UTF8Encoding]::new($false)) }
try {
 $word=New-Object -ComObject Word.Application
 $new=@(Get-Process WINWORD|Where-Object {$existing -notcontains $_.Id})
 if($new.Count -ne 1){throw 'Cannot identify independent Word process.'}
 $owned=$true;$info.ProcessId=$new[0].Id;$info.Version=$word.Version;$info.Build=$word.Build
 foreach($name in @('UpdateLinksAtOpen','UpdateLinksAtPrint','UpdateFieldsAtPrint','Pagination')){
  $old[$name]=$word.Options.$name;$word.Options.$name=$false
 }
 $word.AutomationSecurity=3;$word.DisplayAlerts=0;$word.Visible=$false
 $doc=$word.Documents.Open($local,$false,$true,$false)
 $doc.PrintRevisions=$false;$doc.ActiveWindow.View.Type=3
 $doc.Repaginate();$info.Pages=$doc.ComputeStatistics(2)
 $pages=$doc.ActiveWindow.Panes.Item(1).Pages
 $info.PageCollectionCount=$pages.Count;$info.Geometry=@()
 foreach($number in @(33,38,43,60,127,159,197)){
  $page=$pages.Item($number);$rectangles=@()
  foreach($rect in $page.Rectangles){
   $r=$rect.Range;$t=[string]$r.Text
   $rectangles += [ordered]@{Type=$rect.RectangleType;Left=$rect.Left;Top=$rect.Top;Width=$rect.Width;Height=$rect.Height;Story=$r.StoryType;Start=$r.Start;End=$r.End;TextStart=$t.Substring(0,[Math]::Min(100,$t.Length));TextEnd=$t.Substring([Math]::Max(0,$t.Length-100))}
  }
  $info.Geometry += [ordered]@{Page=$number;Rectangles=$rectangles}
 }
 $info.Status='geometry_complete';Write-Info
 Write-Output ('GEOMETRY_COMPLETE pages='+$info.Pages)
 $info.Status='rendering_page';Write-Info
 $bits=[byte[]]$pages.Item(38).EnhMetaFileBits
 Add-Type -AssemblyName System.Drawing
 $stream=[IO.MemoryStream]::new($bits,$false)
 $metafile=[Drawing.Imaging.Metafile]::new($stream)
 $bitmap=[Drawing.Bitmap]::new(1191,1684)
 $graphics=[Drawing.Graphics]::FromImage($bitmap)
 $graphics.Clear([Drawing.Color]::White)
 $graphics.DrawImage($metafile,[Drawing.Rectangle]::new(0,0,1191,1684))
 $bitmap.Save($PreviewPath,[Drawing.Imaging.ImageFormat]::Png)
 $graphics.Dispose();$bitmap.Dispose();$metafile.Dispose();$stream.Dispose()
 $info.Status='completed';$info.RenderedPreview=$PreviewPath;Write-Info
 Write-Output 'PAGE_PREVIEW_COMPLETE'
} finally {
 if($null -ne $doc){$doc.Close([ref]0)}
 if($owned){foreach($name in $old.Keys){$word.Options.$name=$old[$name]};$word.Quit([ref]0)}
 Write-Info
}
