param([string]$InfoPath,[string]$ResultPdf,[string]$ExportInfo)
$ErrorActionPreference='Stop'
$j=Get-Content -LiteralPath $InfoPath -Raw -Encoding UTF8|ConvertFrom-Json
$source=Join-Path $j.LocalStaging 'manuscript.docx'
$pdf=Join-Path $j.LocalStaging 'preview_readonly.pdf'
& '\\wsl.localhost\Ubuntu-22.04\home\erzhu419\mine_code\Thesis\workpapers\格式检测逐项修复_20261003\export_fresh_readonly.ps1' -Source $source -LocalPdf $pdf -ResultPdf $ResultPdf -InfoPath $ExportInfo
