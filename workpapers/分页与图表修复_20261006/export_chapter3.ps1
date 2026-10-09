param([string]$InfoPath,[string]$ResultPdf,[string]$ExportInfo)
$ErrorActionPreference='Stop'
$j=Get-Content -LiteralPath $InfoPath -Raw -Encoding UTF8|ConvertFrom-Json
$source=Join-Path $j.LocalStaging 'manuscript.docx'
$pdf=Join-Path $j.LocalStaging 'preview_readonly.pdf'
& '\\wsl.localhost\Ubuntu-22.04\home\erzhu419\mine_code\Thesis\workpapers\分页与图表修复_20261006\export_chapter3_readonly.ps1' -Source $source -LocalPdf $pdf -ResultPdf $ResultPdf -InfoPath $ExportInfo
