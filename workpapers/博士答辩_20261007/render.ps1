param([string]$InputFile, [string]$OutputBase)
$ErrorActionPreference = 'Stop'
$app = New-Object -ComObject PowerPoint.Application
$pres = $null
try {
    $pres = $app.Presentations.Open($InputFile, $true, $false, $false)
    $pres.SaveAs($OutputBase + '.pdf', 32)
    $pngDir = $OutputBase + '_pages'
    New-Item -ItemType Directory -Path $pngDir -Force | Out-Null
    $layout = @()
    foreach ($slide in $pres.Slides) {
        $slide.Export((Join-Path $pngDir ('slide_{0:D2}.png' -f $slide.SlideIndex)), 'PNG', 1600, 900)
        foreach ($shape in $slide.Shapes) {
            if ($shape.HasTextFrame -and $shape.TextFrame.HasText) {
                $t = $shape.TextFrame2.TextRange
                $usableH = $shape.Height - $shape.TextFrame2.MarginTop - $shape.TextFrame2.MarginBottom
                $usableW = $shape.Width - $shape.TextFrame2.MarginLeft - $shape.TextFrame2.MarginRight
                if ($t.BoundHeight -gt ($usableH + 3) -or $t.BoundWidth -gt ($usableW + 3)) {
                    $layout += [PSCustomObject]@{
                        slide = $slide.SlideIndex; shape = $shape.Name
                        text = $shape.TextFrame.TextRange.Text
                        boundH = $t.BoundHeight; usableH = $usableH
                        boundW = $t.BoundWidth; usableW = $usableW
                    }
                }
            }
        }
    }
    $result = [PSCustomObject]@{
        software = 'Microsoft PowerPoint'; version = $app.Version; build = $app.Build
        slides = $pres.Slides.Count; textOverflow = $layout
    } | ConvertTo-Json -Depth 5
    [System.IO.File]::WriteAllText($OutputBase + '_pagination.json', $result, [System.Text.UTF8Encoding]::new($false))
} finally {
    if ($pres) { $pres.Close() }
    if ($app.Presentations.Count -eq 0) { $app.Quit() }
}
