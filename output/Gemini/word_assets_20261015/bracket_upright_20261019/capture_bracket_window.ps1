param([string]$InputDocument,[string]$WorkDirectory)
$ErrorActionPreference='Stop'
Add-Type -AssemblyName System.Drawing
Add-Type @'
using System;
using System.Runtime.InteropServices;
public class BracketPreviewWindow {
 [StructLayout(LayoutKind.Sequential)] public struct RECT {public int Left,Top,Right,Bottom;}
 [DllImport("user32.dll")] public static extern bool GetWindowRect(IntPtr h,out RECT r);
 [DllImport("user32.dll")] public static extern bool SetForegroundWindow(IntPtr h);
 [DllImport("user32.dll")] public static extern bool PrintWindow(IntPtr h,IntPtr dc,uint flags);
}
'@
$word=$null;$doc=$null;$owned=$false
try {
 $word=New-Object -ComObject Word.Application
 if($word.Documents.Count -ne 0){throw 'Expected isolated Word instance.'}
 $owned=$true;$word.Visible=$true;$word.DisplayAlerts=0
 $doc=$word.Documents.Open($InputDocument,$false,$true)
 if(-not $doc.ReadOnly){throw 'Expected read-only repair copy.'}
 $word.WindowState=0;$word.Left=0;$word.Top=0;$word.Width=850;$word.Height=650
 $doc.ActiveWindow.DocumentMap=$false;$doc.ActiveWindow.View.Type=3
 $doc.ActiveWindow.View.ShowAll=$false;$doc.ActiveWindow.View.Zoom.Percentage=110
 $doc.Repaginate()
 $r=$doc.OMaths.Item(4139).Range
 $r.Select();$word.Selection.Collapse(1)
 $doc.ActiveWindow.ScrollIntoView($r,$true)
 [void][BracketPreviewWindow]::SetForegroundWindow([IntPtr]$doc.ActiveWindow.Hwnd)
 $doc.ActiveWindow.SmallScroll(25,0,0,0)
 $word.ScreenRefresh();[System.Threading.Thread]::Sleep(1200)
 @{mathText=$r.Text;page=$r.Information(3);noProofing=$r.NoProofing} | ConvertTo-Json | Out-File -Encoding utf8 (Join-Path $WorkDirectory 'C31_window_target.json')
 $rect=New-Object BracketPreviewWindow+RECT
 [void][BracketPreviewWindow]::GetWindowRect([IntPtr]$doc.ActiveWindow.Hwnd,[ref]$rect)
 $bitmap=[System.Drawing.Bitmap]::new($rect.Right-$rect.Left,$rect.Bottom-$rect.Top)
 $g=[System.Drawing.Graphics]::FromImage($bitmap);$dc=$g.GetHdc()
 try{[void][BracketPreviewWindow]::PrintWindow([IntPtr]$doc.ActiveWindow.Hwnd,$dc,0)}finally{$g.ReleaseHdc($dc)}
 $bitmap.Save((Join-Path $WorkDirectory 'native_preview\C31_actual_Word_window.png'),[System.Drawing.Imaging.ImageFormat]::Png)
 $g.Dispose();$bitmap.Dispose()
 Write-Output 'C31_WINDOW_CAPTURE_DONE'
 $doc.Close(0);$doc=$null;$word.Quit();$word=$null
}
finally{if($null -ne $doc){$doc.Close(0)};if(($null -ne $word) -and $owned){$word.Quit()}}
