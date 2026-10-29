param([string]$InputDocument,[string]$WorkDirectory)
$ErrorActionPreference='Stop'
Add-Type -AssemblyName System.Drawing
Add-Type @'
using System;
using System.Runtime.InteropServices;
public static class SymbolTableUi {
 [StructLayout(LayoutKind.Sequential)] public struct RECT {public int Left,Top,Right,Bottom;}
 [DllImport("user32.dll")] public static extern bool SetProcessDPIAware();
 [DllImport("user32.dll")] public static extern bool GetWindowRect(IntPtr h,out RECT r);
 [DllImport("user32.dll")] public static extern bool PrintWindow(IntPtr h,IntPtr dc,uint flags);
}
'@
[void][SymbolTableUi]::SetProcessDPIAware()
$imageDirectory=Join-Path $WorkDirectory 'table_ui'
[IO.Directory]::CreateDirectory($imageDirectory) | Out-Null
$word=$null;$doc=$null;$owned=$false
$rows=@();$failures=@()
try {
 $word=New-Object -ComObject Word.Application
 if($word.Documents.Count -ne 0){throw 'Expected isolated Word instance.'}
 $owned=$true;$word.Visible=$true;$word.DisplayAlerts=0
 $doc=$word.Documents.Open($InputDocument,$false,$true,$false)
 if(-not $doc.ReadOnly){throw 'Read-only rendering required.'}
 $word.WindowState=0;$word.Width=950;$word.Height=400
 $window=$doc.ActiveWindow
 $window.DocumentMap=$false;$window.View.Type=3;$window.View.ShowAll=$false
 $window.View.Zoom.Percentage=125
 $doc.Repaginate()
 $table=$doc.Tables.Item(3)
 if($table.Rows.Count -ne 104){throw 'Expected 103 notation rows plus header.'}
 $hwnd=[IntPtr]$window.Hwnd
 foreach($i in 1..$table.Rows.Count){
  $range=$table.Rows.Item($i).Range
  $window.ScrollIntoView($range,$true)
  $word.ScreenRefresh();[Threading.Thread]::Sleep(80)
  [int]$left=0;[int]$top=0;[int]$width=0;[int]$height=0
  $rect=New-Object SymbolTableUi+RECT
  [void][SymbolTableUi]::GetWindowRect($hwnd,[ref]$rect)
  try {
   $window.GetPoint([ref]$left,[ref]$top,[ref]$width,[ref]$height,$range)
   if($top -lt $rect.Top+125 -or $top+$height -gt $rect.Bottom-60){
    $window.ScrollIntoView($range,$false)
    $word.ScreenRefresh();[Threading.Thread]::Sleep(100)
    $window.GetPoint([ref]$left,[ref]$top,[ref]$width,[ref]$height,$range)
   }
   if($top+$height -gt $rect.Bottom-60){
    $window.SmallScroll(3,0,0,0);$word.ScreenRefresh()
    $window.GetPoint([ref]$left,[ref]$top,[ref]$width,[ref]$height,$range)
   }
  } catch {$failures+=@{row=$i;error=$_.Exception.Message};continue}
  $bitmap=[Drawing.Bitmap]::new($rect.Right-$rect.Left,$rect.Bottom-$rect.Top,[Drawing.Imaging.PixelFormat]::Format32bppRgb)
  $graphics=[Drawing.Graphics]::FromImage($bitmap);$dc=$graphics.GetHdc()
  try {$ok=[SymbolTableUi]::PrintWindow($hwnd,$dc,0)} finally {$graphics.ReleaseHdc($dc);$graphics.Dispose()}
  if(-not $ok){throw 'PrintWindow failed.'}
  $x=$left-$rect.Left;$y=$top-$rect.Top
  $cropRect=[Drawing.Rectangle]::FromLTRB([Math]::Max(0,$x-8),[Math]::Max(0,$y-8),[Math]::Min($bitmap.Width,$x+$width+8),[Math]::Min($bitmap.Height,$y+$height+8))
  $crop=$bitmap.Clone($cropRect,[Drawing.Imaging.PixelFormat]::Format32bppRgb)
  $name='row_'+$i.ToString('000')+'.png'
  try {$crop.Save((Join-Path $imageDirectory $name),[Drawing.Imaging.ImageFormat]::Png)} finally {$crop.Dispose()}
  if($i -eq 1 -or $i -eq 104){$bitmap.Save((Join-Path $imageDirectory ('viewport_'+$i.ToString('000')+'.png')),[Drawing.Imaging.ImageFormat]::Png)}
  $bitmap.Dispose()
  $rows+=@{row=$i;file=$name;page=$range.Information(3);rect=@($x,$y,$width,$height)}
  if(($i % 20) -eq 0){Write-Output ('TABLE_ROWS='+$i+'/'+$table.Rows.Count)}
 }
 $pages=@()
 foreach($toc in $doc.TablesOfContents){
  $toc.UpdatePageNumbers()
  foreach($field in $toc.Range.Fields){
   if($field.Code.Text -match 'PAGEREF\s+(_Toc[0-9]+)'){
    $pages+=@{bookmark=$Matches[1];page=$field.Result.Text.Trim()}
   }
  }
 }
 @{source=$InputDocument;rows=$rows;failures=$failures;readonly=$doc.ReadOnly;math_objects=$doc.OMaths.Count;physical_pages=$doc.ComputeStatistics(2);toc_pages=$pages} | ConvertTo-Json -Depth 7 | Out-File -Encoding utf8 (Join-Path $WorkDirectory 'native_table_review.json')
 Write-Output ('TABLE_DONE failures='+$failures.Count+' pages='+$doc.ComputeStatistics(2))
 $doc.Close(0);$doc=$null;$word.Quit();$word=$null
} finally {
 if($null -ne $doc){$doc.Close(0)}
 if($null -ne $word -and $owned){$word.Quit()}
}
