param([string]$InputDocument,[string]$PreviewDirectory)
$ErrorActionPreference='Stop'
Add-Type -AssemblyName System.Drawing
$word=$null;$doc=$null
try {
 $word=New-Object -ComObject Word.Application
 $word.Visible=$false;$word.DisplayAlerts=0
 $doc=$word.Documents.Open($InputDocument,$false,$true)
 if(-not $doc.ReadOnly){throw 'Expected read-only probe.'}
 $doc.ActiveWindow.View.Type=3;$doc.Repaginate()
 $pages=$doc.ComputeStatistics(2)
 foreach($show in @($false,$true)){
  $doc.ActiveWindow.View.ShowAll=$show
  $doc.Repaginate()
  $mode=if($show){'marks_on'}else{'marks_off'}
  for($pageNo=1;$pageNo -le $pages;$pageNo++){
   $page=$doc.ActiveWindow.Panes.Item(1).Pages.Item($pageNo)
   [byte[]]$bits=$page.EnhMetaFileBits
   $stream=[System.IO.MemoryStream]::new($bits)
   $metafile=[System.Drawing.Imaging.Metafile]::new($stream)
   $bitmap=[System.Drawing.Bitmap]::new(1700,2404)
   $g=[System.Drawing.Graphics]::FromImage($bitmap)
   $g.Clear([System.Drawing.Color]::White);$g.DrawImage($metafile,0,0,1700,2404)
   $bitmap.Save((Join-Path $PreviewDirectory ($mode+'_'+$pageNo.ToString('000')+'.png')),[System.Drawing.Imaging.ImageFormat]::Png)
   $g.Dispose();$bitmap.Dispose();$metafile.Dispose();$stream.Dispose()
  }
 }
 @{pages=$pages;read_only=$doc.ReadOnly;omath=$doc.OMaths.Count} | ConvertTo-Json | Out-File -Encoding utf8 (Join-Path $PreviewDirectory 'probe_native.json')
 $doc.ActiveWindow.View.ShowAll=$false
 $doc.ExportAsFixedFormat((Join-Path $PreviewDirectory 'membership_font_test.pdf'),17)
 Write-Output ('PAGES='+$pages)
 $doc.Close(0);$doc=$null;$word.Quit();$word=$null
}
finally {
 if($null -ne $doc){$doc.Close(0)}
 if($null -ne $word){$word.Quit()}
}
