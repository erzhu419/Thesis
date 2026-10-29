param([Parameter(Mandatory=$true)][string]$Folder)
$ErrorActionPreference = 'Stop'
$app = $null
$doc = $null
try {
    Write-Output 'CONNECT VISIO'
    $app = [Runtime.InteropServices.Marshal]::GetActiveObject('Visio.Application')
    Write-Output 'VISIO CREATED'
    $initial = $app.Documents.Count
    Write-Output ('DOCUMENTS '+$initial)
    if ($initial -eq 0) {
        $app.Visible = $true
        $app.AlertResponse = 1
        $app.EventsEnabled = $false
    }
    foreach ($name in @('f21','f41','f52','f11')) {
        Write-Output ('OPEN '+$name)
        $doc = $app.Documents.OpenEx((Join-Path $Folder ($name+'.vsdx')),2)
        Write-Output ('OPENED '+$name)
        $page = $doc.Pages.Item(1)
        $page.Export((Join-Path $Folder ($name+'.emf')))
        Write-Output ('EMF '+$name)
        $page.Export((Join-Path $Folder ($name+'.png')))
        $doc.Close()
        $doc = $null
        Write-Output ('EXPORTED '+$name)
    }
    Write-Output ('Visio '+$app.Version)
}
finally {
    if ($null -ne $doc) { $doc.Close() }
    if ($null -ne $app -and $initial -eq 0) { $app.Quit() }
}
