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
    $section = $document.Bookmarks.Item(([char]0x7814).ToString() + [char]0x7A76 + [char]0x80CC + [char]0x666F + [char]0x4E0E + [char]0x610F + [char]0x4E49).Range
    $first = $document.Range($section.Start, $section.Start).Information(3)
    $last = $document.Range($section.End - 1, $section.End - 1).Information(3)
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
        [void][System.Runtime.InteropServices.Marshal]::ReleaseComObject($document)
    }
    if ($null -ne $word) {
        $word.Quit()
        [void][System.Runtime.InteropServices.Marshal]::ReleaseComObject($word)
    }
}
