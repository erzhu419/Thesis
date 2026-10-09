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
    $document = $word.Documents.Open($InputDocument, $false, $false)
    foreach ($toc in $document.TablesOfContents) {
        $toc.Update()
    }
    $document.Fields.Update() | Out-Null
    $document.Repaginate()
    $document.Save()
    $document.ExportAsFixedFormat($OutputPdf, 17)
    $pages = $document.ComputeStatistics(2)
    Write-Output ("WORD_RENDER_PAGES=" + $pages)
    Write-Output ("WORD_RENDER_PDF=" + $OutputPdf)
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
