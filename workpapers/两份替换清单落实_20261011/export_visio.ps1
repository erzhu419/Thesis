param([string]$InputFile,[string]$OutputBase)
$ErrorActionPreference='Stop'
$app=New-Object -ComObject Visio.Application
$initial=$app.Documents.Count
$doc=$null
try {
    $doc=$app.Documents.OpenEx($InputFile,2)
    $page=$doc.Pages.Item(1)
    $page.Export($OutputBase+'.emf')
    $page.Export($OutputBase+'.png')
    [PSCustomObject]@{Application='Microsoft Visio';Version=$app.Version;Pages=$doc.Pages.Count;Shapes=$page.Shapes.Count} | ConvertTo-Json -Compress
} finally {
    if ($doc) { $doc.Close() }
    if ($initial -eq 0 -and $app.Documents.Count -eq 0) { $app.Quit() }
}
