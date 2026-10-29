param([string]$InputDocument,[string]$PreviewDirectory)
$ErrorActionPreference='Stop'
Add-Type -AssemblyName System.Drawing
Add-Type @'
using System;
using System.Runtime.InteropServices;
public class ProbeWindow {
 [StructLayout(LayoutKind.Sequential)] public struct RECT {public int Left,Top,Right,Bottom;}
 [DllImport("user32.dll")] public static extern bool GetWindowRect(IntPtr hwnd,out RECT r);
 [DllImport("user32.dll")] public static extern bool SetForegroundWindow(IntPtr hwnd);
 [DllImport("user32.dll")] public static extern bool PrintWindow(IntPtr hwnd, IntPtr hdc, uint flags);
}
'@
$word=$null;$doc=$null;$owned=$false
try {
 $word=New-Object -ComObject Word.Application
 if($word.Documents.Count -ne 0){throw 'New Word instance unexpectedly contains documents.'}
 $owned=$true;$word.Visible=$true;$word.DisplayAlerts=0
 $doc=$word.Documents.Open($InputDocument,$false,$true)
 if(-not $doc.ReadOnly){throw 'Expected read-only probe.'}
 $doc.ActiveWindow.View.Type=3;$doc.Repaginate()
 $word.WindowState=0;$word.Left=0;$word.Top=0;$word.Width=850;$word.Height=650
 $doc.ActiveWindow.DocumentMap=$false;$doc.ActiveWindow.View.Zoom.Percentage=95
 foreach($show in @($false,$true)){
  $doc.ActiveWindow.View.ShowAll=$show
  $mode=if($show){'screen_on'}else{'screen_off'}
  for($pageNo=1;$pageNo -le 5;$pageNo++){
   $r=$doc.Content.Duplicate;$r.Find.ClearFormatting()
   $prefix=([string][char](64+$pageNo))+' - '
   if(-not $r.Find.Execute($prefix)){throw ('Missing probe label '+$prefix)}
   $r.Select();$doc.ActiveWindow.ScrollIntoView($r,$true);$word.ScreenRefresh()
   [void][ProbeWindow]::SetForegroundWindow([IntPtr]$doc.ActiveWindow.Hwnd)
   [System.Threading.Thread]::Sleep(900)
   $rect=New-Object ProbeWindow+RECT
   [void][ProbeWindow]::GetWindowRect([IntPtr]$doc.ActiveWindow.Hwnd,[ref]$rect)
   $width=$rect.Right-$rect.Left;$height=$rect.Bottom-$rect.Top
   $bitmap=[System.Drawing.Bitmap]::new($width,$height)
   $g=[System.Drawing.Graphics]::FromImage($bitmap)
   $hdc=$g.GetHdc()
   try{[void][ProbeWindow]::PrintWindow([IntPtr]$doc.ActiveWindow.Hwnd,$hdc,0)}finally{$g.ReleaseHdc($hdc)}
   $bitmap.Save((Join-Path $PreviewDirectory ($mode+'_'+$pageNo.ToString('000')+'.png')),[System.Drawing.Imaging.ImageFormat]::Png)
   $g.Dispose();$bitmap.Dispose()
  }
 }
 Write-Output 'SCREEN_PROBES_DONE'
 $doc.Close(0);$doc=$null;$word.Quit();$word=$null
}
finally {
 if($null -ne $doc){$doc.Close(0)}
 if(($null -ne $word) -and $owned){$word.Quit()}
}
