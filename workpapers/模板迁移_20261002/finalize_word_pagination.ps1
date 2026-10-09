[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)][string]$InputDocument,
    [Parameter(Mandatory = $true)][string]$OutputPdf,
    [Parameter(Mandatory = $true)][string]$InfoJson
)

Set-StrictMode -Version 2.0
$ErrorActionPreference = 'Stop'
[Console]::OutputEncoding = [Text.UTF8Encoding]::new($false)

# Run only on the already-created migration copy, never on the source/template.
# For visual-only revisions: refresh only TOC/page layout; leave SEQ/REF unchanged.
$existingWordPids = @(Get-Process WINWORD -ErrorAction SilentlyContinue | ForEach-Object { $_.Id })
$word = $null
$doc = $null
$ownsWord = $false
$oldOptions = @{}
$failure = $null
$info = [ordered]@{
    InputDocument = $InputDocument
    OutputPdf = $OutputPdf
    Status = 'starting'
    StartedAt = (Get-Date).ToString('o')
    Word = $null
    Iterations = [Collections.Generic.List[object]]::new()
    Stable = $false
    Pages = $null
    TocText = $null
    Sections = @()
    FloatingShapes = @()
    InlineShapeCount = $null
    Error = $null
    CleanupErrors = [Collections.Generic.List[string]]::new()
}

function Format-Toc($Document) {
    for ($level = 1; $level -le 3; $level++) {
        $style = $Document.Styles.Item(-19 - $level)
        $style.Font.Size = 12
        if ($level -eq 1) { $style.Font.NameFarEast = [string][char]0x9ED1 + [char]0x4F53 }
        else { $style.Font.NameFarEast = [string][char]0x5B8B + [char]0x4F53 }
        $format = $style.ParagraphFormat
        $format.LineSpacingRule = 4
        $format.LineSpacing = 20
        $format.SpaceBeforeAuto = 0
        $format.SpaceAfterAuto = 0
        $format.SpaceBefore = 0
        $format.SpaceAfter = 0
        if ($level -eq 1) { $format.SpaceBefore = 6; $format.SpaceAfter = 6 }
        $format.CharacterUnitLeftIndent = 0
        $format.CharacterUnitFirstLineIndent = 0
        $format.FirstLineIndent = 0
        $format.LeftIndent = 10.5 * ($level - 1)
        $format.RightIndent = 0
    }
    for ($i = 1; $i -le $Document.TablesOfContents.Count; $i++) {
        $range = $Document.TablesOfContents.Item($i).Range
        $range.Font.Size = 12
        $format = $range.ParagraphFormat
        $format.LineSpacingRule = 4
        $format.LineSpacing = 20
        $format.TabStops.ClearAll()
        [void]$format.TabStops.Add(414.8, 2, 1)
        # The TOC field closes in its own empty paragraph; keep that paragraph tiny.
        $closing = $Document.Range($range.End, $range.End + 1).Paragraphs.Item(1)
        if (([string]$closing.Range.Text -match '^\s*$') -and
            ([string]$closing.Range.WordOpenXML -match 'fldCharType="end"')) {
            $closing.Range.Font.Size = 1
            $closing.Format.LineSpacingRule = 4
            $closing.Format.LineSpacing = 1
            $closing.Format.SpaceBeforeAuto = 0
            $closing.Format.SpaceAfterAuto = 0
            $closing.Format.SpaceBefore = 0
            $closing.Format.SpaceAfter = 0
            $closing.Format.KeepWithNext = 0
        }
    }
}

function Get-TocText($Document) {
    $parts = [Collections.Generic.List[string]]::new()
    for ($i = 1; $i -le $Document.TablesOfContents.Count; $i++) {
        $parts.Add([string]$Document.TablesOfContents.Item($i).Range.Text)
    }
    return [string]::Join("`n<TOC>`n", $parts.ToArray())
}

function Get-SectionInfo($Document) {
    $result = [Collections.Generic.List[object]]::new()
    for ($i = 1; $i -le $Document.Sections.Count; $i++) {
        $section = $Document.Sections.Item($i)
        $start = $section.Range.Duplicate
        $start.Collapse(1) # wdCollapseStart
        $end = $section.Range.Duplicate
        $end.SetRange([Math]::Max($end.Start, $end.End - 1), $end.End)
        $end.Collapse(1)
        $setup = $section.PageSetup
        $headersFooters = [Collections.Generic.List[object]]::new()
        foreach ($kind in @('Headers', 'Footers')) {
            $collection = $section.$kind
            for ($variant = 1; $variant -le 3; $variant++) {
                $hf = $collection.Item($variant)
                $headersFooters.Add([ordered]@{
                    Kind = $kind; Variant = $variant; Exists = [bool]$hf.Exists
                    LinkToPrevious = [bool]$hf.LinkToPrevious
                    RestartNumberingAtSection = [bool]$hf.PageNumbers.RestartNumberingAtSection
                    StartingNumber = $hf.PageNumbers.StartingNumber
                    NumberStyle = [int]$hf.PageNumbers.NumberStyle
                })
            }
        }
        $result.Add([ordered]@{
            Index = $i; Start = $section.Range.Start; End = $section.Range.End
            FirstPhysicalPage = $start.Information(3); LastPhysicalPage = $end.Information(3)
            FirstAdjustedPage = $start.Information(1); LastAdjustedPage = $end.Information(1)
            StartType = [int]$setup.SectionStart
            DifferentFirstPage = $setup.DifferentFirstPageHeaderFooter
            OddAndEvenPages = $setup.OddAndEvenPagesHeaderFooter
            HeadersFooters = $headersFooters.ToArray()
        })
    }
    return $result.ToArray()
}

function Get-ShapeInfo($Shape, [string]$Scope) {
    return [ordered]@{
        Scope = $Scope; Name = [string]$Shape.Name; Type = [int]$Shape.Type
        AnchorStory = [int]$Shape.Anchor.StoryType; AnchorStart = $Shape.Anchor.Start
        AnchorPhysicalPage = $Shape.Anchor.Information(3)
        Left = $Shape.Left; Top = $Shape.Top; Width = $Shape.Width; Height = $Shape.Height
        RelativeHorizontalPosition = [int]$Shape.RelativeHorizontalPosition
        RelativeVerticalPosition = [int]$Shape.RelativeVerticalPosition
        WrapType = [int]$Shape.WrapFormat.Type
    }
}

function Get-FloatingShapes($Document) {
    $result = [Collections.Generic.List[object]]::new()
    foreach ($shape in $Document.Shapes) { $result.Add((Get-ShapeInfo $shape 'Main')) }
    for ($i = 1; $i -le $Document.Sections.Count; $i++) {
        $section = $Document.Sections.Item($i)
        foreach ($kind in @('Headers', 'Footers')) {
            for ($variant = 1; $variant -le 3; $variant++) {
                $hf = $section.$kind.Item($variant)
                # Linked headers/footers are represented by their owning section.
                if (($i -gt 1) -and $hf.LinkToPrevious) { continue }
                foreach ($shape in $hf.Shapes) {
                    $scope = 'Section{0}.{1}.{2}' -f $i, $kind, $variant
                    $result.Add((Get-ShapeInfo $shape $scope))
                }
            }
        }
    }
    return $result.ToArray()
}

try {
    $word = New-Object -ComObject Word.Application
    $newWordPids = @(Get-Process WINWORD -ErrorAction SilentlyContinue |
        Where-Object { ($existingWordPids -notcontains $_.Id) -and ($_.Id -ne 45772) } |
        ForEach-Object { $_.Id })
    if ($newWordPids.Count -ne 1) {
        throw "COM activation did not yield exactly one new independent Word process (new PIDs: $newWordPids)."
    }
    $wordProcessId = $newWordPids[0]
    $ownsWord = $true
    $info.Word = [ordered]@{
        Name = $word.Name; Version = $word.Version; Build = $word.Build
        Path = $word.Path; ProcessId = $wordProcessId
        PreexistingProcessIds = $existingWordPids
    }
    foreach ($name in @('UpdateLinksAtOpen', 'UpdateLinksAtPrint', 'UpdateFieldsAtPrint', 'Pagination')) {
        $oldOptions[$name] = $word.Options.$name
        $word.Options.$name = $false
    }
    $oldOptions['AutomationSecurity'] = $word.AutomationSecurity
    $oldOptions['DisplayAlerts'] = $word.DisplayAlerts
    $oldOptions['ScreenUpdating'] = $word.ScreenUpdating
    $word.AutomationSecurity = 3 # msoAutomationSecurityForceDisable
    $word.DisplayAlerts = 0
    $word.ScreenUpdating = $false
    $word.Visible = $false

    Write-Output 'OPEN_MIGRATION_COPY'
    $doc = $word.Documents.Open($InputDocument, $false, $false, $false)
    if ($doc.ReadOnly) { throw 'The migration copy opened read-only.' }
    $info.Status = 'updating'
    $previousPages = -1
    $previousToc = $null
    for ($iteration = 1; $iteration -le 4; $iteration++) {
        for ($i = 1; $i -le $doc.TablesOfContents.Count; $i++) {
            $doc.TablesOfContents.Item($i).Update()
        }
        Format-Toc $doc
        $doc.Repaginate()
        for ($i = 1; $i -le $doc.TablesOfContents.Count; $i++) {
            $doc.TablesOfContents.Item($i).UpdatePageNumbers()
        }
        Format-Toc $doc
        $doc.Repaginate()
        $pages = [int]$doc.ComputeStatistics(2)
        $tocText = Get-TocText $doc
        $samePages = $pages -eq $previousPages
        $sameToc = ($null -ne $previousToc) -and
            [string]::Equals($tocText, $previousToc, [StringComparison]::Ordinal)
        $info.Iterations.Add([ordered]@{
            Iteration = $iteration; Pages = $pages; TocCharacters = $tocText.Length
            SamePageCount = $samePages; SameTocText = $sameToc
        })
        $info.Pages = $pages
        $info.TocText = $tocText
        Write-Output ('PAGINATION iteration={0} pages={1} samePages={2} sameToc={3}' -f
            $iteration, $pages, $samePages, $sameToc)
        if ($samePages -and $sameToc) { $info.Stable = $true; break }
        $previousPages = $pages
        $previousToc = $tocText
    }

    $info.Sections = @(Get-SectionInfo $doc)
    $info.FloatingShapes = @(Get-FloatingShapes $doc)
    $info.InlineShapeCount = $doc.InlineShapes.Count
    Write-Output 'SAVE_MIGRATION_COPY'
    $doc.Save()
    Write-Output 'EXPORT_FINAL_PDF'
    # Keep the two-argument COM export that is verified in this Word installation.
    $doc.ExportAsFixedFormat($OutputPdf, 17)
    $info.PagesAfterExport = [int]$doc.ComputeStatistics(2)
    if ($info.PagesAfterExport -ne $info.Pages) { $info.Stable = $false }
    $info.Status = 'completed'
    if (-not $info.Stable) { $info.Status = 'completed_unstable' }
} catch {
    $failure = $_
    $info.Status = 'failed'
    $info.Error = $_.Exception.Message
} finally {
    if ($null -ne $doc) {
        try { $dontSave = 0; $doc.Close([ref]$dontSave) } catch { $info.CleanupErrors.Add($_.Exception.Message) }
        [void][Runtime.InteropServices.Marshal]::ReleaseComObject($doc)
    }
    if (($null -ne $word) -and $ownsWord) {
        foreach ($name in @('UpdateLinksAtOpen', 'UpdateLinksAtPrint', 'UpdateFieldsAtPrint', 'Pagination')) {
            if ($oldOptions.ContainsKey($name)) {
                try { $word.Options.$name = $oldOptions[$name] }
                catch { $info.CleanupErrors.Add($_.Exception.Message) }
            }
        }
        foreach ($name in @('AutomationSecurity', 'DisplayAlerts', 'ScreenUpdating')) {
            if ($oldOptions.ContainsKey($name)) {
                try { $word.$name = $oldOptions[$name] }
                catch { $info.CleanupErrors.Add($_.Exception.Message) }
            }
        }
        try { $dontSave = 0; $word.Quit([ref]$dontSave) } catch { $info.CleanupErrors.Add($_.Exception.Message) }
    }
    if ($null -ne $word) { [void][Runtime.InteropServices.Marshal]::ReleaseComObject($word) }
    $info.FinishedAt = (Get-Date).ToString('o')
    [IO.File]::WriteAllText($InfoJson, ($info | ConvertTo-Json -Depth 20),
        [Text.UTF8Encoding]::new($false))
    [GC]::Collect()
    [GC]::WaitForPendingFinalizers()
}
if ($null -ne $failure) { throw $failure }
Write-Output ('FINAL_STATUS={0}; PAGES={1}; STABLE={2}' -f $info.Status, $info.Pages, $info.Stable)
