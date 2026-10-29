param([string]$InputDocument,[string]$WorkDirectory)
$ErrorActionPreference='Stop'
Add-Type -AssemblyName System.Drawing
Add-Type @'
using System;
using System.Runtime.InteropServices;
public static class FormulaUiProbe {
 [StructLayout(LayoutKind.Sequential)] public struct RECT {public int Left,Top,Right,Bottom;}
 [DllImport("user32.dll")] public static extern bool SetProcessDPIAware();
 [DllImport("user32.dll")] public static extern bool GetWindowRect(IntPtr h,out RECT r);
 [DllImport("user32.dll")] public static extern bool PrintWindow(IntPtr h,IntPtr dc,uint flags);
}
'@
[void][FormulaUiProbe]::SetProcessDPIAware()
$imageDirectory=Join-Path $WorkDirectory 'native_crops'
$viewportDirectory=Join-Path $WorkDirectory 'native_viewports'
[IO.Directory]::CreateDirectory($imageDirectory) | Out-Null
[IO.Directory]::CreateDirectory($viewportDirectory) | Out-Null
$writer=[IO.StreamWriter]::new((Join-Path $WorkDirectory 'native_capture_refinements.jsonl'),$false,[Text.UTF8Encoding]::new($false))
$word=$null;$doc=$null;$owned=$false;$bitmap=$null
$results=@();$viewport=30000;$scrolls=0;$failures=@()
try {
 $word=New-Object -ComObject Word.Application
 if($word.Documents.Count -ne 0){throw 'Expected isolated Word instance.'}
 $owned=$true;$word.Visible=$true;$word.DisplayAlerts=0
 $doc=$word.Documents.Open($InputDocument,$false,$true,$false)
 if(-not $doc.ReadOnly){throw 'UI probe requires read-only input.'}
 $word.WindowState=0;$word.Width=1000;$word.Height=690
 $doc.ActiveWindow.DocumentMap=$false
 $doc.ActiveWindow.View.Type=3;$doc.ActiveWindow.View.ShowAll=$false
 $doc.ActiveWindow.View.Zoom.Percentage=125
 $refUpdates=@()
 foreach($field in $doc.Fields){
  $code=$field.Code.Text
  if($code -match '^\s*REF\s+(eqA_00[45])\b'){
   $bookmark=$Matches[1];$oldResult=$field.Result.Text
   [void]$field.Update();$newResult=$field.Result.Text
   if($newResult -match '错误|Error!'){
    $field.Code.Text=(' REF '+$bookmark+' '+[char]92+'h ')
    [void]$field.Update();$newResult=$field.Result.Text
   }
   $refUpdates+=@{bookmark=$bookmark;old_code=$code;new_code=$field.Code.Text;old_result=$oldResult;new_result=$newResult}
  }
 }
 if($refUpdates.Count -ne 2){throw 'Expected exactly two authorized REF updates.'}
 $doc.Repaginate()
 $mathCollection=$doc.OMaths
 $hwnd=[IntPtr]$doc.ActiveWindow.Hwnd
 $window=New-Object FormulaUiProbe+RECT
 [void][FormulaUiProbe]::GetWindowRect($hwnd,[ref]$window)
 if($doc.OMaths.Count -ne 4738){throw 'Expected all 4738 equations.'}
 foreach($mathIndex in (Get-Content -Raw -Encoding utf8 (Join-Path $WorkDirectory 'final_review_targets.json') | ConvertFrom-Json)){
  $case=@{name=('M'+$mathIndex.ToString('000000'));kind='original_position';first_math_index_1based=$mathIndex;last_math_index_1based=$mathIndex}
  $r=$mathCollection.Item($mathIndex).Range
  [int]$left=0;[int]$top=0;[int]$width=0;[int]$height=0
  $scrolled=$false
  try {
   $doc.ActiveWindow.GetPoint([ref]$left,[ref]$top,[ref]$width,[ref]$height,$r)
   if($top -lt $window.Top+150 -or $top+$height -gt $window.Bottom-55 -or $left -lt $window.Left -or $left+$width -gt $window.Right){throw 'Equation outside visible viewport.'}
  } catch {
   $doc.ActiveWindow.ScrollIntoView($r,$true)
   $word.ScreenRefresh();[Threading.Thread]::Sleep(100)
   try {
    $doc.ActiveWindow.GetPoint([ref]$left,[ref]$top,[ref]$width,[ref]$height,$r)
    if($top -lt $window.Top+150 -or $top+$height -gt $window.Bottom-55){throw 'Equation still outside document viewport.'}
   } catch {
    $doc.ActiveWindow.ScrollIntoView($r,$false)
    $word.ScreenRefresh();[Threading.Thread]::Sleep(200)
    try {$doc.ActiveWindow.GetPoint([ref]$left,[ref]$top,[ref]$width,[ref]$height,$r)}
    catch {$failures+=@{math_index=$mathIndex;error=$_.Exception.Message};continue}
   }
   if($top+$height -gt $window.Bottom-55){$doc.ActiveWindow.SmallScroll(3,0,0,0);$word.ScreenRefresh();$doc.ActiveWindow.GetPoint([ref]$left,[ref]$top,[ref]$width,[ref]$height,$r)}
   if($top -lt $window.Top+150){$doc.ActiveWindow.SmallScroll(0,3,0,0);$word.ScreenRefresh();$doc.ActiveWindow.GetPoint([ref]$left,[ref]$top,[ref]$width,[ref]$height,$r)}
   $scrolled=$true;$scrolls++
  }
  if($width -le 0 -or $height -le 0){throw ('Empty GetPoint rectangle: '+$case.name)}
  if($scrolled -or $null -eq $bitmap){
   if($null -ne $bitmap){$bitmap.Dispose()}
   $window=New-Object FormulaUiProbe+RECT
   if(-not [FormulaUiProbe]::GetWindowRect($hwnd,[ref]$window)){throw 'GetWindowRect failed.'}
   $word.ScreenRefresh();[Threading.Thread]::Sleep(75)
   $bitmap=[Drawing.Bitmap]::new($window.Right-$window.Left,$window.Bottom-$window.Top,[Drawing.Imaging.PixelFormat]::Format32bppRgb)
   $g=[Drawing.Graphics]::FromImage($bitmap);$dc=$g.GetHdc()
   try { $ok=[FormulaUiProbe]::PrintWindow($hwnd,$dc,0) } finally {$g.ReleaseHdc($dc);$g.Dispose()}
   if(-not $ok){throw 'PrintWindow failed.'}
   $viewport++
   $viewportFile=Join-Path $viewportDirectory ('viewport_'+$viewport.ToString('0000')+'.png')
   $bitmap.Save($viewportFile,[Drawing.Imaging.ImageFormat]::Png)
  }
  $x=$left-$window.Left;$y=$top-$window.Top
  if($x -lt 0 -or $y -lt 0 -or $x+$width -gt $bitmap.Width -or $y+$height -gt $bitmap.Height){throw ('GetPoint/window coordinate mismatch: '+$case.name)}
  $strict=[Drawing.Rectangle]::new($x,$y,$width,$height)
  $padded=[Drawing.Rectangle]::FromLTRB([Math]::Max(0,$x-10),[Math]::Max(0,$y-10),[Math]::Min($bitmap.Width,$x+$width+10),[Math]::Min($bitmap.Height,$y+$height+10))
  $stem=Join-Path $imageDirectory $case.name
  foreach($item in @(@{suffix='.png';rect=$padded})){
   $crop=$bitmap.Clone($item.rect,[Drawing.Imaging.PixelFormat]::Format32bppRgb)
   try {$crop.Save(($stem+$item.suffix),[Drawing.Imaging.ImageFormat]::Png)} finally {$crop.Dispose()}
  }
  $pageNumber=$r.Information(3)
  $row=@{name=$case.name;math_index=$mathIndex;start_page=$pageNumber;end_page=$pageNumber;getpoint_local_rect=@($x,$y,$width,$height);viewport=$viewport;crop=($case.name+'.png')}
  $writer.WriteLine(($row | ConvertTo-Json -Depth 4 -Compress));$writer.Flush();$results+=$row
  if(($mathIndex % 250) -eq 0){Write-Output ('CAPTURED='+$mathIndex+'/4738 viewports='+$viewport)}
 }
 $tocPages=@()
 foreach($toc in $doc.TablesOfContents){
  $toc.UpdatePageNumbers()
  foreach($field in $toc.Range.Fields){
   if($field.Code.Text -match 'PAGEREF\s+(_Toc[0-9]+)'){
    $tocPages+=@{bookmark=$Matches[1];page=$field.Result.Text.Trim()}
   }
  }
 }
 @{ref_updates=$refUpdates;toc_pages=$tocPages;pages=$doc.ComputeStatistics(2);math_objects=$doc.OMaths.Count;readonly=$doc.ReadOnly} | ConvertTo-Json -Depth 6 | Out-File -Encoding utf8 (Join-Path $WorkDirectory 'native_final_field_results.json')
 @{source=$InputDocument;read_only=$doc.ReadOnly;omath=$doc.OMaths.Count;zoom=$doc.ActiveWindow.View.Zoom.Percentage;captured=$results.Count;failures=$failures;pages=$doc.ComputeStatistics(2);viewport_count=$viewport;scroll_count=$scrolls;formula_selection_used=$false;pdf_generated=$false;emf_used=$false} |
  ConvertTo-Json -Depth 7 | Out-File -Encoding utf8 (Join-Path $WorkDirectory 'native_refinement_summary.json')
 $writer.Close();$writer=$null
 Write-Output ('ALL_FORMULA_CAPTURE_DONE failures='+$failures.Count+' viewports='+$viewport)
 $doc.Close(0);$doc=$null;$word.Quit();$word=$null
} finally {
 if($null -ne $writer){$writer.Close()}
 if($null -ne $bitmap){$bitmap.Dispose()}
 if($null -ne $doc){$doc.Close(0)}
 if(($null -ne $word) -and $owned){$word.Quit()}
}
