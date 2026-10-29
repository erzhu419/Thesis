param(
    [Parameter(Mandatory=$true)][string]$InputDocument,
    [Parameter(Mandatory=$true)][string]$OutputPdf,
    [switch]$UpdateToc
)
$ErrorActionPreference = "Stop"
$word = $null
$document = $null
try {
    $word = New-Object -ComObject Word.Application
    $initialDocuments = $word.Documents.Count
    if ($initialDocuments -eq 0) {
        $word.Visible = $false
        $word.DisplayAlerts = 0
    }
    $document = $word.Documents.Open($InputDocument, $false, (-not $UpdateToc))
    $document.Repaginate()
    if ($UpdateToc) {
        foreach ($toc in $document.TablesOfContents) {
            $toc.UpdatePageNumbers()
        }
        $document.Repaginate()
        $document.Save()
    }
    $document.ExportAsFixedFormat($OutputPdf, 17)
    $pages = $document.ComputeStatistics(2)
    $headings = @()
    $chapterBookmarks = @("_Toc242183007", "_Toc242183017", "_Toc242183038", "_Toc242183058", "_Toc242183075", "_Toc242183093", "_Toc242183111")
    for ($i = 1; $i -le 7; $i++) {
        $range = $document.Bookmarks.Item($chapterBookmarks[$i-1]).Range
        $headings += [pscustomobject]@{Chapter=$i; Page=$range.Information(3)}
    }
    [pscustomobject]@{
        WordVersion = $word.Version
        WordBuild = $word.Build
        Pages = $pages
        TocPageNumbersUpdated = [bool]$UpdateToc
        ChapterPages = $headings
    } | ConvertTo-Json -Depth 5 -Compress
}
finally {
    if ($null -ne $document) {
        $document.Close(0)
        [void][Runtime.InteropServices.Marshal]::ReleaseComObject($document)
    }
    if ($null -ne $word) {
        if ($initialDocuments -eq 0) { $word.Quit() }
        [void][Runtime.InteropServices.Marshal]::ReleaseComObject($word)
    }
}
