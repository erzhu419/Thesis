$ErrorActionPreference = 'Stop'

$docxPath = '\\wsl.localhost\Ubuntu-22.04\home\erzhu419\mine_code\Thesis\output\_word_finalize_input.docx'
$pdfPath = '\\wsl.localhost\Ubuntu-22.04\home\erzhu419\mine_code\Thesis\output\_word_finalize_output.pdf'

$word = $null
$doc = $null
try {
    $word = New-Object -ComObject Word.Application
    $word.Visible = $false
    $word.DisplayAlerts = 0
    $doc = $word.Documents.Open($docxPath)
    if ($doc -eq $null) {
        throw "Word returned a null document object while opening $docxPath"
    }
    $doc.TrackRevisions = $false

    $revisionCountBefore = $doc.Revisions.Count
    $commentCountBefore = $doc.Comments.Count
    if ($revisionCountBefore -gt 0) {
        $doc.Revisions.AcceptAll()
    }
    while ($doc.Comments.Count -gt 0) {
        $doc.Comments.Item(1).Delete()
    }

    $doc.Fields.Update() | Out-Null
    foreach ($toc in $doc.TablesOfContents) {
        $toc.Update() | Out-Null
        $toc.UpdatePageNumbers() | Out-Null
    }
    foreach ($section in $doc.Sections) {
        foreach ($header in $section.Headers) {
            $header.Range.Fields.Update() | Out-Null
        }
        foreach ($footer in $section.Footers) {
            $footer.Range.Fields.Update() | Out-Null
        }
    }

    if ($doc.Revisions.Count -gt 0) {
        $doc.Revisions.AcceptAll()
    }

    $doc.Save()
    $doc.ExportAsFixedFormat($pdfPath, 17, $false, 0, 0, 0, 0, 0, $true, $false, 0, $true, $true, $false)
    $doc.Close(0)
    $doc = $null
    $word.Quit()
    $word = $null
    Write-Output ("WORD_VERSION=" + $word.Version)
    Write-Output ("REVISIONS_BEFORE=" + $revisionCountBefore)
    Write-Output ("COMMENTS_BEFORE=" + $commentCountBefore)
    Write-Output "WORD_FINALIZE=OK"
}
finally {
    if ($doc -ne $null) {
        try { $doc.Close(0) } catch {}
    }
    if ($word -ne $null) {
        try { $word.Quit() } catch {}
    }
    if ($doc -ne $null) { [void][System.Runtime.InteropServices.Marshal]::ReleaseComObject($doc) }
    if ($word -ne $null) { [void][System.Runtime.InteropServices.Marshal]::ReleaseComObject($word) }
}
