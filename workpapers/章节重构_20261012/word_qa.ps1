param([string]$InputFile,[string]$PreviewFile)
$ErrorActionPreference='Stop'
$word=New-Object -ComObject Word.Application
$initial=$word.Documents.Count
$document=$null
try {
    if ($initial -eq 0) { $word.Visible=$false; $word.DisplayAlerts=0 }
    $document=$word.Documents.Open($InputFile,$false,$false)
    $document.Repaginate()
    foreach ($toc in $document.TablesOfContents) { $toc.UpdatePageNumbers() }
    $document.Repaginate()
    foreach ($toc in $document.TablesOfContents) { $toc.UpdatePageNumbers() }
    $document.Save()
    $document.ExportAsFixedFormat($PreviewFile,17)
    [PSCustomObject]@{Application='Microsoft Word';Version=$word.Version;Build=$word.Build;Pages=$document.ComputeStatistics(2);TocCount=$document.TablesOfContents.Count} | ConvertTo-Json -Compress
} finally {
    if ($document) { Start-Sleep -Seconds 2; $document.Close(0) }
    if ($initial -eq 0 -and $word.Documents.Count -eq 0) { $word.Quit() }
}
