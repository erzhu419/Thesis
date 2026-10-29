param(
    [Parameter(Mandatory=$true)][string]$InputDocument,
    [Parameter(Mandatory=$true)][string]$OutputPdf
)

$ErrorActionPreference = "Stop"
$word = $null
$document = $null
try {
    $word = New-Object -ComObject Word.Application
    $word.Visible = $false
    $word.DisplayAlerts = 0
    $document = $word.Documents.Open($InputDocument, $false, $true)
    $document.Repaginate()
    $start = $document.Bookmarks.Item("_Toc242183022").Range
    $next = $document.Bookmarks.Item("_Toc242183027").Range
    $first = $document.Range($start.Start, $start.Start).Information(3)
    $last = $document.Range($next.Start - 1, $next.Start - 1).Information(3)
    $from = [Math]::Max(1, $first - 1)
    $to = [Math]::Min($document.ComputeStatistics(2), $last + 1)
    $document.ExportAsFixedFormat($OutputPdf, 17, $false, 0, 3, $from, $to)
    [pscustomobject]@{
        WordVersion = $word.Version
        WordBuild = $word.Build
        SectionStartPage = $first
        SectionEndPage = $last
        PreviewFirstPage = $from
        PreviewLastPage = $to
        FieldsUpdated = $false
        DocumentSavedByWord = $false
    } | ConvertTo-Json -Compress
}
finally {
    if ($null -ne $document) {
        $document.Close(0)
        [void][Runtime.InteropServices.Marshal]::ReleaseComObject($document)
    }
    if ($null -ne $word) {
        $word.Quit()
        [void][Runtime.InteropServices.Marshal]::ReleaseComObject($word)
    }
}
