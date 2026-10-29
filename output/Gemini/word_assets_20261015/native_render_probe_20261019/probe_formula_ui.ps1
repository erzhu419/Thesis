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
$cases=Get-Content -Raw (Join-Path $WorkDirectory 'ui_probe_targets.json') | ConvertFrom-Json
$word=$null;$doc=$null;$owned=$false;$bitmap=$null
$results=@();$viewport=0;$scrolls=0
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
 $doc.Repaginate()
 $hwnd=[IntPtr]$doc.ActiveWindow.Hwnd
 foreach($case in ($cases.targets | Sort-Object first_math_index_1based)){
  $first=$doc.OMaths.Item([int]$case.first_math_index_1based).Range
  $last=$doc.OMaths.Item([int]$case.last_math_index_1based).Range
  $r=$doc.Range($first.Start,$last.End)
  [int]$left=0;[int]$top=0;[int]$width=0;[int]$height=0
  $scrolled=$false
  try {
   $doc.ActiveWindow.GetPoint([ref]$left,[ref]$top,[ref]$width,[ref]$height,$r)
  } catch {
   $doc.ActiveWindow.ScrollIntoView($r,$true)
   $word.ScreenRefresh();[Threading.Thread]::Sleep(350)
   $doc.ActiveWindow.GetPoint([ref]$left,[ref]$top,[ref]$width,[ref]$height,$r)
   $scrolled=$true;$scrolls++
  }
  if($width -le 0 -or $height -le 0){throw ('Empty GetPoint rectangle: '+$case.name)}
  if($scrolled -or $null -eq $bitmap){
   if($null -ne $bitmap){$bitmap.Dispose()}
   $window=New-Object FormulaUiProbe+RECT
   if(-not [FormulaUiProbe]::GetWindowRect($hwnd,[ref]$window)){throw 'GetWindowRect failed.'}
   $word.ScreenRefresh();[Threading.Thread]::Sleep(150)
   $bitmap=[Drawing.Bitmap]::new($window.Right-$window.Left,$window.Bottom-$window.Top,[Drawing.Imaging.PixelFormat]::Format32bppRgb)
   $g=[Drawing.Graphics]::FromImage($bitmap);$dc=$g.GetHdc()
   try { $ok=[FormulaUiProbe]::PrintWindow($hwnd,$dc,0) } finally {$g.ReleaseHdc($dc);$g.Dispose()}
   if(-not $ok){throw 'PrintWindow failed.'}
   $viewport++
   $viewportFile=Join-Path $WorkDirectory ('ui_viewport_'+$viewport.ToString('000')+'.png')
   $bitmap.Save($viewportFile,[Drawing.Imaging.ImageFormat]::Png)
  }
  $x=$left-$window.Left;$y=$top-$window.Top
  if($x -lt 0 -or $y -lt 0 -or $x+$width -gt $bitmap.Width -or $y+$height -gt $bitmap.Height){throw ('GetPoint/window coordinate mismatch: '+$case.name)}
  $strict=[Drawing.Rectangle]::new($x,$y,$width,$height)
  $padded=[Drawing.Rectangle]::FromLTRB([Math]::Max(0,$x-10),[Math]::Max(0,$y-10),[Math]::Min($bitmap.Width,$x+$width+10),[Math]::Min($bitmap.Height,$y+$height+10))
  $stem=Join-Path $WorkDirectory ('ui_'+$case.name)
  foreach($item in @(@{suffix='_strict.png';rect=$strict},@{suffix='_padded.png';rect=$padded})){
   $crop=$bitmap.Clone($item.rect,[Drawing.Imaging.PixelFormat]::Format32bppRgb)
   try {$crop.Save(($stem+$item.suffix),[Drawing.Imaging.ImageFormat]::Png)} finally {$crop.Dispose()}
  }
  $start=$r.Duplicate;$start.Collapse(1);$end=$r.Duplicate;$end.Collapse(0)
  $results+=@{name=$case.name;kind=$case.kind;first_math_index_1based=$case.first_math_index_1based;last_math_index_1based=$case.last_math_index_1based;actual_range_text=$r.Text;start_page=$start.Information(3);end_page=$end.Information(3);getpoint_screen_rect=@($left,$top,$width,$height);getpoint_local_rect=@($x,$y,$width,$height);window_rect=@($window.Left,$window.Top,$window.Right,$window.Bottom);viewport=$viewportFile;scrolled=$scrolled;strict_png=($stem+'_strict.png');padded_png=($stem+'_padded.png')}
 }
 @{source=$InputDocument;read_only=$doc.ReadOnly;omath=$doc.OMaths.Count;zoom=$doc.ActiveWindow.View.Zoom.Percentage;cases=$results;viewport_count=$viewport;scroll_count=$scrolls;formula_selection_used=$false;pdf_generated=$false;emf_used=$false} |
  ConvertTo-Json -Depth 7 | Out-File -Encoding utf8 (Join-Path $WorkDirectory 'ui_formula_probe.json')
 Write-Output ('UI_FORMULA_PROBE_DONE cases='+$results.Count+' viewports='+$viewport)
 $doc.Close(0);$doc=$null;$word.Quit();$word=$null
} finally {
 if($null -ne $bitmap){$bitmap.Dispose()}
 if($null -ne $doc){$doc.Close(0)}
 if(($null -ne $word) -and $owned){$word.Quit()}
}
