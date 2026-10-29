param([string]$InputFile, [string]$OutputBase)
$ErrorActionPreference = 'Stop'
$app = New-Object -ComObject PowerPoint.Application
$pres = $null
try {
    $pres = $app.Presentations.Open($InputFile, $true, $false, $false)
    $pres.SaveAs($OutputBase + '.pdf', 32)
    $pngDir = $OutputBase + '_pages'
    New-Item -ItemType Directory -Path $pngDir -Force | Out-Null
    $overflow = @()
    foreach ($page in @(14,19)) {
        $slide = $pres.Slides.Item($page)
        $slide.Export((Join-Path $pngDir ('slide_{0:D2}.png' -f $page)), 'PNG', 1600, 900)
        foreach ($shape in $slide.Shapes) {
            if ($shape.HasTextFrame -and $shape.TextFrame.HasText) {
                $t = $shape.TextFrame2.TextRange
                $h = $shape.Height - $shape.TextFrame2.MarginTop - $shape.TextFrame2.MarginBottom
                $w = $shape.Width - $shape.TextFrame2.MarginLeft - $shape.TextFrame2.MarginRight
                if ($t.BoundHeight -gt ($h + 3) -or $t.BoundWidth -gt ($w + 3)) {
                    $overflow += [PSCustomObject]@{page=$page; shape=$shape.Name}
                }
            }
        }
    }
    [PSCustomObject]@{
        software='Microsoft PowerPoint'; version=$app.Version; build=$app.Build
        slides=$pres.Slides.Count; inspectedPages=@(14,19); textOverflow=$overflow
    } | ConvertTo-Json -Depth 5 -Compress
} finally {
    if ($pres) { $pres.Close() }
    if ($app.Presentations.Count -eq 0) { $app.Quit() }
}
