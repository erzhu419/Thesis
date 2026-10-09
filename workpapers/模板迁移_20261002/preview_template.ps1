param([string]$InputDocument,[string]$OutputPdf,[string]$InfoJson)
$ErrorActionPreference='Stop'
[Console]::OutputEncoding=[Text.UTF8Encoding]::new($false)
$word=$null; $doc=$null
try {
  $word=New-Object -ComObject Word.Application
  $word.Visible=$false; $word.DisplayAlerts=0
  $oldLinks=$word.Options.UpdateLinksAtOpen
  $word.Options.UpdateLinksAtOpen=$false
  Write-Output 'OPEN_TEMPLATE_COPY'
  $doc=$word.Documents.Open($InputDocument,$false,$true)
  $doc.Repaginate()
  $sections=@()
  for($i=1;$i -le $doc.Sections.Count;$i++) {
    $s=$doc.Sections.Item($i); $p=$s.PageSetup
    $sections+= [ordered]@{Index=$i; Start=$s.Range.Start; End=$s.Range.End; StartPage=$s.Range.Information(3);
      Width=$p.PageWidth;Height=$p.PageHeight;Top=$p.TopMargin;Bottom=$p.BottomMargin;Left=$p.LeftMargin;Right=$p.RightMargin;
      Gutter=$p.Gutter;HeaderDistance=$p.HeaderDistance;FooterDistance=$p.FooterDistance;StartType=$p.SectionStart;
      DifferentFirst=$p.DifferentFirstPageHeaderFooter;OddEven=$p.OddAndEvenPagesHeaderFooter}
  }
  [ordered]@{Version=$word.Version;Pages=$doc.ComputeStatistics(2);Sections=$sections} | ConvertTo-Json -Depth 5 | Set-Content -Encoding UTF8 $InfoJson
  Write-Output 'EXPORT_TEMPLATE_PREVIEW'
  $doc.ExportAsFixedFormat($OutputPdf,17)
  Write-Output ('PAGES='+$doc.ComputeStatistics(2))
} finally {
  if($null -ne $doc){$doc.Close(0);[void][Runtime.InteropServices.Marshal]::ReleaseComObject($doc)}
  if($null -ne $word){$word.Options.UpdateLinksAtOpen=$oldLinks;$word.Quit();[void][Runtime.InteropServices.Marshal]::ReleaseComObject($word)}
}
