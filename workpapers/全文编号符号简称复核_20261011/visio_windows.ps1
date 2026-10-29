$ErrorActionPreference = 'Stop'
Add-Type @'
using System;
using System.Text;
using System.Runtime.InteropServices;
public static class OfficeWindows {
  public delegate bool WindowCallback(IntPtr h, IntPtr p);
  [DllImport("user32.dll")] public static extern bool EnumWindows(WindowCallback f, IntPtr p);
  [DllImport("user32.dll")] public static extern uint GetWindowThreadProcessId(IntPtr h, out uint p);
  [DllImport("user32.dll", CharSet=CharSet.Unicode)] public static extern int GetWindowText(IntPtr h, StringBuilder s, int n);
  [DllImport("user32.dll", CharSet=CharSet.Unicode)] public static extern int GetClassName(IntPtr h, StringBuilder s, int n);
  [DllImport("user32.dll")] public static extern bool IsWindowVisible(IntPtr h);
}
'@
$targets = @(Get-Process VISIO -ErrorAction SilentlyContinue | ForEach-Object Id)
$script:windows = [Collections.Generic.List[string]]::new()
[OfficeWindows]::EnumWindows({
    param($handle, $unused)
    [uint32]$processId = 0
    [void][OfficeWindows]::GetWindowThreadProcessId($handle, [ref]$processId)
    $title = [Text.StringBuilder]::new(512)
    $class = [Text.StringBuilder]::new(256)
    [void][OfficeWindows]::GetWindowText($handle, $title, 512)
    [void][OfficeWindows]::GetClassName($handle, $class, 256)
    if ($targets -contains $processId) {
        $script:windows.Add('PID='+$processId+' HWND='+$handle+' Visible='+[OfficeWindows]::IsWindowVisible($handle)+' Class='+$class+' Title='+$title)
    }
    return $true
}, [IntPtr]::Zero) | Out-Null
$script:windows
