param([string]$Stage)
$ErrorActionPreference = 'Stop'
$word = $null
$doc = $null
try {
    $word = New-Object -ComObject Word.Application
    $initialDocuments = $word.Documents.Count
    if ($initialDocuments -eq 0) { $word.Visible = $false; $word.DisplayAlerts = 0 }
    $results = @()
    foreach ($name in @('current', 'template')) {
        $doc = $word.Documents.Open("$Stage\$name.docx", $false, $true)
        $range = $doc.Content.Duplicate
        $query = if ($name -eq 'current') { 'DAGANZO C F. A headway-based' } else { 'HOU S, WEI J.' }
        $found = $range.Find.Execute($query)
        if (-not $found) { throw "Reference not found in $name" }
        $para = $range.Paragraphs.Item(1).Range
        $format = $para.ParagraphFormat
        $tabs = @()
        foreach ($tab in $format.TabStops) {
            $tabs += [pscustomobject]@{Position=$tab.Position;Alignment=$tab.Alignment;Leader=$tab.Leader}
        }
        $results += [pscustomobject]@{
            Document=$name; Page=$para.Information(3); Text=$para.Text.Trim()
            Alignment=$format.Alignment; LeftIndent=$format.LeftIndent
            FirstLineIndent=$format.FirstLineIndent; RightIndent=$format.RightIndent
            SpaceBefore=$format.SpaceBefore; SpaceAfter=$format.SpaceAfter
            LineSpacingRule=$format.LineSpacingRule; LineSpacing=$format.LineSpacing
            WordWrap=$format.WordWrap; DisableLineHeightGrid=$format.DisableLineHeightGrid
            KeepWithNext=$format.KeepWithNext; KeepTogether=$format.KeepTogether
            FontAscii=$para.Font.NameAscii; FontFarEast=$para.Font.NameFarEast; FontSize=$para.Font.Size
            DefaultTabStop=$word.Options.DefaultTabStop; Tabs=$tabs
        }
        if ($name -eq 'template') {
            $page = $para.Information(3)
            $doc.ExportAsFixedFormat("$Stage\template_refs.pdf", 17, $false, 0, 3, $page, ($page + 2))
        }
        $doc.Close(0)
        [void][Runtime.InteropServices.Marshal]::ReleaseComObject($doc)
        $doc = $null
    }
    $results | ConvertTo-Json -Depth 5 -Compress
} finally {
    if ($null -ne $doc) { $doc.Close(0); [void][Runtime.InteropServices.Marshal]::ReleaseComObject($doc) }
    if ($null -ne $word) {
        if ($initialDocuments -eq 0) { $word.Quit() }
        [void][Runtime.InteropServices.Marshal]::ReleaseComObject($word)
    }
}
