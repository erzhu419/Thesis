param([string]$InputEmf,[string]$OutputStem,[int]$Width=1700)
$ErrorActionPreference='Stop'
Add-Type -AssemblyName System.Drawing
if(-not ('NativeEmfProbe' -as [type])){
Add-Type -ReferencedAssemblies System.Drawing -TypeDefinition @'
using System;
using System.Drawing;
using System.Drawing.Imaging;
using System.IO;
using System.Runtime.InteropServices;
public static class NativeEmfProbe {
  [StructLayout(LayoutKind.Sequential)] public struct RECT {public int L,T,R,B;}
  [DllImport("gdi32.dll",SetLastError=true)] static extern IntPtr SetEnhMetaFileBits(uint n,byte[] data);
  [DllImport("gdi32.dll",SetLastError=true)] static extern bool PlayEnhMetaFile(IntPtr dc,IntPtr emf,ref RECT rect);
  [DllImport("gdi32.dll")] static extern bool DeleteEnhMetaFile(IntPtr emf);
  public static void Render(byte[] data,int width,int height,string stem) {
    // Native GDI playback, using GDI+ only as the bitmap/PNG container.
    IntPtr emf=SetEnhMetaFileBits((uint)data.Length,data);
    if(emf==IntPtr.Zero) throw new System.ComponentModel.Win32Exception();
    try {
      using(Bitmap bmp=new Bitmap(width,height,PixelFormat.Format32bppRgb)) {
        using(Graphics g=Graphics.FromImage(bmp)) {
          g.Clear(Color.White);
          IntPtr dc=g.GetHdc();
          try {
            RECT r=new RECT {L=0,T=0,R=width-1,B=height-1};
            if(!PlayEnhMetaFile(dc,emf,ref r)) throw new System.ComponentModel.Win32Exception();
          } finally {g.ReleaseHdc(dc);}
        }
        bmp.Save(stem+"_gdi.png",ImageFormat.Png);
      }
    } finally {DeleteEnhMetaFile(emf);}
    // The current thesis-preview route, from the exact same EMF and pixel size.
    using(MemoryStream ms=new MemoryStream(data))
    using(Metafile mf=new Metafile(ms))
    using(Bitmap bmp=new Bitmap(width,height,PixelFormat.Format32bppRgb)) {
      using(Graphics g=Graphics.FromImage(bmp)) {
        g.Clear(Color.White);g.DrawImage(mf,0,0,width,height);
      }
      bmp.Save(stem+"_gdiplus.png",ImageFormat.Png);
    }
  }
}
'@
}
[byte[]]$bits=[IO.File]::ReadAllBytes($InputEmf)
$frameWidth=[BitConverter]::ToInt32($bits,32)-[BitConverter]::ToInt32($bits,24)
$frameHeight=[BitConverter]::ToInt32($bits,36)-[BitConverter]::ToInt32($bits,28)
$height=[int][Math]::Round($Width*$frameHeight/[double]$frameWidth)
[NativeEmfProbe]::Render($bits,$Width,$height,$OutputStem)
# EMF-specified weights distinguish an already-bold recording from a replay issue.
$fonts=@();$records=@{};$offset=0
while($offset -lt $bits.Length){
  $kind=[BitConverter]::ToUInt32($bits,$offset)
  $size=[BitConverter]::ToUInt32($bits,$offset+4)
  if($size -lt 8){throw 'Invalid EMF record size.'}
  if(-not $records.ContainsKey([string]$kind)){$records[[string]$kind]=0}
  $records[[string]$kind]++
  if($kind -eq 82){
    $face=[Text.Encoding]::Unicode.GetString($bits,$offset+40,64).Split([char]0)[0]
    $fonts+=@{face=$face;height=[BitConverter]::ToInt32($bits,$offset+12);weight=[BitConverter]::ToInt32($bits,$offset+28);italic=$bits[$offset+32]}
  }
  $offset+=$size
}
@{input_emf=$InputEmf;width=$Width;height=$height;frame_0_01mm=@($frameWidth,$frameHeight);fonts=$fonts;record_counts=$records;pdf_generated=$false} |
 ConvertTo-Json -Depth 5 | Out-File -Encoding utf8 ($OutputStem+'_replay.json')
