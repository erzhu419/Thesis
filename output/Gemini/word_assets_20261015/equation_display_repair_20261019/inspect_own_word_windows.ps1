$ErrorActionPreference='Stop'
Add-Type @'
using System;
using System.Runtime.InteropServices;
using System.Text;
public class WordWindows {
 public delegate bool EnumProc(IntPtr h,IntPtr p);
 [DllImport("user32.dll")] public static extern bool EnumWindows(EnumProc cb,IntPtr p);
 [DllImport("user32.dll")] public static extern bool EnumChildWindows(IntPtr h,EnumProc cb,IntPtr p);
 [DllImport("user32.dll")] public static extern uint GetWindowThreadProcessId(IntPtr h,out uint pid);
 [DllImport("user32.dll",CharSet=CharSet.Unicode)] public static extern int GetWindowText(IntPtr h,StringBuilder s,int n);
}
'@
$processIds=@(Get-Process WINWORD -ErrorAction SilentlyContinue | ForEach-Object {$_.Id})
$items=[System.Collections.Generic.List[object]]::new()
$cb=[WordWindows+EnumProc]{param($h,$p)
 [uint32]$wpid=0;[void][WordWindows]::GetWindowThreadProcessId($h,[ref]$wpid)
 if($processIds -contains $wpid){
  $s=[System.Text.StringBuilder]::new(1024);[void][WordWindows]::GetWindowText($h,$s,1024)
  $items.Add(@{pid=$wpid;hwnd=$h.ToInt64();text=$s.ToString()})
  $child=[WordWindows+EnumProc]{param($c,$q)
   $t=[System.Text.StringBuilder]::new(1024);[void][WordWindows]::GetWindowText($c,$t,1024)
   if($t.Length -gt 0){$items.Add(@{child=$c.ToInt64();text=$t.ToString()})}
   return $true
  }
  [void][WordWindows]::EnumChildWindows($h,$child,[IntPtr]::Zero)
 }
 return $true
}
[void][WordWindows]::EnumWindows($cb,[IntPtr]::Zero)
$items | ConvertTo-Json -Depth 3
