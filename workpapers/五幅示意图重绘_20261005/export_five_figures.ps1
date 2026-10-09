$ErrorActionPreference='Stop'
[Console]::OutputEncoding=[Text.UTF8Encoding]::new($false)
$root='\\wsl.localhost\Ubuntu-22.04\home\erzhu419\mine_code\Thesis'
$local=Join-Path $env:TEMP 'thesis_five_figures_20261005'
New-Item -ItemType Directory -Force -Path $local | Out-Null
$source=Join-Path $root '排版完成_最终核对_五图重绘_20261005.docx'
$localDoc=Join-Path $local 'five_figures.docx'
$localPdf=Join-Path $local 'five_figures.pdf'
Copy-Item -LiteralPath $source -Destination $localDoc -Force
& (Join-Path $root 'workpapers\格式检测逐项修复_20261003\export_fresh_readonly.ps1') -Source $localDoc -LocalPdf $localPdf -ResultPdf (Join-Path $root '排版完成_最终核对_五图重绘_20261005.pdf') -InfoPath (Join-Path $root 'workpapers\五幅示意图重绘_20261005\Word导出记录.json')
