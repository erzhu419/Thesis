$ErrorActionPreference='Stop'
Add-Type @'
using System;
using System.Text;
using System.Runtime.InteropServices;
public static class ThesisWordWindows {
 public delegate bool EnumProc(IntPtr h,IntPtr p);
 [DllImport("user32.dll")] public static extern bool EnumWindows(EnumProc f,IntPtr p);
 [DllImport("user32.dll")] public static extern bool EnumChildWindows(IntPtr h,EnumProc f,IntPtr p);
 [DllImport("user32.dll")] public static extern uint GetWindowThreadProcessId(IntPtr h,out uint p);
 [DllImport("user32.dll",CharSet=CharSet.Unicode)] public static extern int GetWindowText(IntPtr h,StringBuilder b,int n);
 [DllImport("user32.dll")] public static extern bool IsWindowVisible(IntPtr h);
}
'@
$taskWordPid=42548
$items=[Collections.Generic.List[object]]::new()
$callback=[ThesisWordWindows+EnumProc]{param($handle,$parameter)
 $owner=0
 [void][ThesisWordWindows]::GetWindowThreadProcessId($handle,[ref]$owner)
 if($owner -eq $taskWordPid){
  $buffer=[Text.StringBuilder]::new(512)
  [void][ThesisWordWindows]::GetWindowText($handle,$buffer,512)
  if($buffer.Length){$items.Add([ordered]@{Handle=$handle.ToInt64();Visible=[ThesisWordWindows]::IsWindowVisible($handle);Text=$buffer.ToString()})}
 }
 return $true
}
[void][ThesisWordWindows]::EnumWindows($callback,[IntPtr]::Zero)
$items | ConvertTo-Json -Depth 3
