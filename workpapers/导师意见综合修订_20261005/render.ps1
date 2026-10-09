$ErrorActionPreference = 'Stop'
$folder = '\\wsl.localhost\Ubuntu-22.04\home\erzhu419\mine_code\Thesis\workpapers\导师意见综合修订_20261005'
$inputPath = Join-Path $folder '修订待分页.docx'
$pdfPath = Join-Path $folder '修订预览.pdf'
$renderDir = Join-Path $env:TEMP ('thesis-render-' + [guid]::NewGuid().ToString('N'))
New-Item -ItemType Directory -Path $renderDir | Out-Null
$localDoc = Join-Path $renderDir 'thesis.docx'
$localPdf = Join-Path $renderDir 'thesis.pdf'
Copy-Item -LiteralPath $inputPath -Destination $localDoc
$word = $null
$doc = $null
try {
  $word = New-Object -ComObject Word.Application
  $word.Visible = $false
  $word.DisplayAlerts = 0
  $version = $word.Version
  Write-Output 'WORD_OPEN'
  $doc = $word.Documents.Open($localDoc, $false, $false, $false)
  if ($null -eq $doc) { throw 'Word returned no document' }
  Write-Output 'WORD_FIELDS'
  $doc.Fields.Update() | Out-Null
  foreach ($toc in $doc.TablesOfContents) { $toc.Update() | Out-Null }
  foreach ($section in $doc.Sections) {
    foreach ($header in $section.Headers) { $header.Range.Fields.Update() | Out-Null }
    foreach ($footer in $section.Footers) { $footer.Range.Fields.Update() | Out-Null }
  }
  $doc.Repaginate()
  foreach ($toc in $doc.TablesOfContents) { $toc.UpdatePageNumbers() | Out-Null }
  $doc.Save()
  Write-Output 'WORD_EXPORT'
  $doc.ExportAsFixedFormat($localPdf,17,$false,0,0,0,0,0,$true,$false,0,$true,$true,$false)
  $info = @{wordVersion=$version; pages=$doc.ComputeStatistics(2); equations=$doc.OMaths.Count; sections=$doc.Sections.Count; comments=$doc.Comments.Count; revisions=$doc.Revisions.Count; pdf=$pdfPath}
  $info | ConvertTo-Json | Set-Content (Join-Path $folder 'word_render.json') -Encoding UTF8
  $info | ConvertTo-Json
  $doc.Close(0); $doc=$null
  $word.Quit(); $word=$null
  Copy-Item -LiteralPath $localDoc -Destination $inputPath -Force
  Copy-Item -LiteralPath $localPdf -Destination $pdfPath -Force
  Remove-Item -LiteralPath $renderDir -Recurse
} finally {
  if ($null -ne $doc) { $doc.Close(0) }
  if ($null -ne $word) { $word.Quit() }
}
