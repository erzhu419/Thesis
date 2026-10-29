param([string]$InputDocument,[string]$WorkDirectory,[int]$First=1,[int]$Last=4825,[int]$Shard=1)
$ErrorActionPreference='Stop'
Add-Type -AssemblyName System.Drawing
Add-Type @'
using System; using System.Runtime.InteropServices;
public static class FormulaNative {
[StructLayout(LayoutKind.Sequential)] public struct RECT {public int Left,Top,Right,Bottom;}
[DllImport("user32.dll")] public static extern bool SetProcessDPIAware();
[DllImport("user32.dll")] public static extern bool GetWindowRect(IntPtr h,out RECT r);
[DllImport("user32.dll")] public static extern bool PrintWindow(IntPtr h,IntPtr dc,uint flags);
}
'@
[void][FormulaNative]::SetProcessDPIAware()
$vd=Join-Path $WorkDirectory 'native_viewports';[IO.Directory]::CreateDirectory($vd)|Out-Null
$writer=[IO.StreamWriter]::new((Join-Path $WorkDirectory ('manifest_'+$Shard+'.jsonl')),$false,[Text.UTF8Encoding]::new($false))
$word=$null;$doc=$null;$bitmap=$null;$owned=$false;$view=$Shard*100000;$failures=@();$count=0;$zoom=90
try {
 $word=New-Object -ComObject Word.Application
 if($word.Documents.Count -ne 0){throw 'Expected isolated Word process.'}
 $owned=$true;$word.Visible=$true;$word.DisplayAlerts=0
 $doc=$word.Documents.Open($InputDocument,$false,$true,$false)
 if(-not $doc.ReadOnly){throw 'Read-only source required.'}
 $word.WindowState=0;$word.Width=950;$word.Height=400
 $win=$doc.ActiveWindow;$win.DocumentMap=$false;$win.View.Type=3;$win.View.ShowAll=$false;$win.View.Zoom.Percentage=$zoom
 $doc.Repaginate();$math=$doc.OMaths
 if($math.Count -ne 4825){throw 'Expected 4825 math objects.'}
 $hwnd=[IntPtr]$win.Hwnd;$box=New-Object FormulaNative+RECT
 [void][FormulaNative]::GetWindowRect($hwnd,[ref]$box)
 for($i=$First;$i -le $Last;$i++){
  try {
   $r=$math.Item($i).Range;[int]$l=0;[int]$t=0;[int]$w=0;[int]$h=0;$changed=$false
   if($zoom -ne 90){$zoom=90;$win.View.Zoom.Percentage=$zoom;$changed=$true}
   $visible=$false
   try {$win.GetPoint([ref]$l,[ref]$t,[ref]$w,[ref]$h,$r);$visible=($t -ge $box.Top+145 -and $t+$h -le $box.Bottom-65 -and $l -ge $box.Left -and $l+$w -le $box.Right)}catch{}
   if(-not $visible){
    $win.ScrollIntoView($r,$true);$word.ScreenRefresh();[Threading.Thread]::Sleep(90);$changed=$true
    $win.GetPoint([ref]$l,[ref]$t,[ref]$w,[ref]$h,$r)
    if($h -gt ($box.Bottom-$box.Top-225)){$zoom=90;$win.View.Zoom.Percentage=$zoom;$win.ScrollIntoView($r,$true);$word.ScreenRefresh();[Threading.Thread]::Sleep(120);$win.GetPoint([ref]$l,[ref]$t,[ref]$w,[ref]$h,$r)}
    if($t+$h -gt $box.Bottom-65){$win.ScrollIntoView($r,$false);$word.ScreenRefresh();[Threading.Thread]::Sleep(80);$win.GetPoint([ref]$l,[ref]$t,[ref]$w,[ref]$h,$r)}
    for($q=0;$q -lt 10 -and ($t -lt $box.Top+140 -or $t+$h -gt $box.Bottom-65);$q++){
     if($t -lt $box.Top+140){$win.SmallScroll(0,1,0,0)}else{$win.SmallScroll(1,0,0,0)}
     $word.ScreenRefresh();[Threading.Thread]::Sleep(30);$win.GetPoint([ref]$l,[ref]$t,[ref]$w,[ref]$h,$r)
    }
   }
   if($w -le 0 -or $h -le 0 -or $t -lt $box.Top+135 -or $t+$h -gt $box.Bottom-55){throw ('Incomplete viewport '+$l+','+$t+','+$w+','+$h)}
   if($changed -or $null -eq $bitmap){
    if($null -ne $bitmap){$bitmap.Dispose()}
    [void][FormulaNative]::GetWindowRect($hwnd,[ref]$box)
    $word.ScreenRefresh();[Threading.Thread]::Sleep(100)
    $bitmap=[Drawing.Bitmap]::new($box.Right-$box.Left,$box.Bottom-$box.Top,[Drawing.Imaging.PixelFormat]::Format32bppRgb)
    $g=[Drawing.Graphics]::FromImage($bitmap);$dc=$g.GetHdc()
    try{$ok=[FormulaNative]::PrintWindow($hwnd,$dc,0)}finally{$g.ReleaseHdc($dc);$g.Dispose()}
    if(-not $ok){throw 'PrintWindow failed.'}
    $view++;$bitmap.Save((Join-Path $vd ('viewport_'+$view.ToString('0000')+'.png')),[Drawing.Imaging.ImageFormat]::Png)
   }
   $x=$l-$box.Left;$y=$t-$box.Top
   if($x -lt 0 -or $y -lt 0 -or $x+$w -gt $bitmap.Width -or $y+$h -gt $bitmap.Height){throw 'Coordinate outside captured image.'}
   $row=@{math_index=$i;start_page=$r.Information(3);getpoint_local_rect=@($x,$y,$w,$h);viewport=$view;zoom=$zoom}
   $writer.WriteLine(($row|ConvertTo-Json -Compress));$writer.Flush();$count++
  }catch{$failures+=@{math_index=$i;error=$_.Exception.Message};Write-Output ('FAILED '+$i+' '+$_.Exception.Message)}
  if(($i % 200) -eq 0){Write-Output ('PROGRESS '+$i+' captured='+$count+' views='+($view-$Shard*100000))}
 }
 @{first=$First;last=$Last;captured=$count;failures=$failures;math_objects=$math.Count;pages=$doc.ComputeStatistics(2);readonly=$doc.ReadOnly}|ConvertTo-Json -Depth 6|Out-File -Encoding utf8 (Join-Path $WorkDirectory ('summary_'+$Shard+'.json'))
 Write-Output ('DONE '+$First+'-'+$Last+' failures='+$failures.Count)
}finally{if($null -ne $writer){$writer.Close()};if($null -ne $bitmap){$bitmap.Dispose()};if($null -ne $doc){$doc.Close(0)};if($null -ne $word -and $owned){$word.Quit()}}
