param([string]$OutputJson)
$ErrorActionPreference = 'Stop'
[Console]::OutputEncoding = [System.Text.UTF8Encoding]::new($false)
Add-Type -AssemblyName System.Drawing
$installed = New-Object System.Drawing.Text.InstalledFontCollection
$word = $null
try {
    $word = New-Object -ComObject Word.Application
    $word.Visible = $false
    $word.DisplayAlerts = 0
    $names = @()
    foreach ($font in $word.FontNames) { $names += [string]$font }
    $path = Join-Path $word.Path 'WINWORD.EXE'
    $result = [ordered]@{
        WordCallable = $true
        WordVersion = [string]$word.Version
        WordBuild = [string]$word.Build
        WordExecutable = $path
        WordFileVersion = [string](Get-Item $path).VersionInfo.FileVersion
        WordFonts = @($names | Sort-Object -Unique)
        InstalledFontFamilies = @($installed.Families | ForEach-Object { $_.Name } | Sort-Object -Unique)
    }
    $result | ConvertTo-Json -Depth 5 | Set-Content -Encoding UTF8 $OutputJson
    Write-Output ('WORD_VERSION=' + $result.WordFileVersion)
    Write-Output ('INSTALLED_FONT_FAMILIES=' + $result.InstalledFontFamilies.Count)
} finally {
    if ($null -ne $word) { $word.Quit(); [void][Runtime.InteropServices.Marshal]::ReleaseComObject($word) }
    $installed.Dispose()
}
