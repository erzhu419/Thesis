Add-Type -TypeDefinition @'
using System;
using System.Text;
using System.Runtime.InteropServices;
public class ThesisWin {
 public delegate bool Callback(IntPtr h, IntPtr p);
 [DllImport("user32.dll")] public static extern bool EnumWindows(Callback c,IntPtr p);
 [DllImport("user32.dll")] public static extern uint GetWindowThreadProcessId(IntPtr h,out uint p);
 [DllImport("user32.dll")] public static extern int GetWindowText(IntPtr h,StringBuilder s,int n);
 [DllImport("user32.dll")] public static extern bool IsWindowVisible(IntPtr h);
}
'@
[ThesisWin]::EnumWindows({
 param($h,$p)
 $owner=0
 [ThesisWin]::GetWindowThreadProcessId($h,[ref]$owner)|Out-Null
 if($owner -eq 30668){
  $s=New-Object Text.StringBuilder 1024
  [ThesisWin]::GetWindowText($h,$s,1024)|Out-Null
  [pscustomobject]@{Handle=$h;Visible=[ThesisWin]::IsWindowVisible($h);Title=$s.ToString()}|ConvertTo-Json -Compress|Write-Host
 }
 $true
},[IntPtr]::Zero)|Out-Null
