$ErrorActionPreference='Stop'
[Console]::OutputEncoding=[Text.UTF8Encoding]::new($false)
$root='\\wsl.localhost\Ubuntu-22.04\home\erzhu419\mine_code\Thesis'
$local=Join-Path $env:TEMP ('thesis_bus_schematic_'+[Guid]::NewGuid().ToString('N'))
New-Item -ItemType Directory -Path $local | Out-Null
$source=Join-Path $root '排版完成_五图公交示意版_20261006.docx'
$localDoc=Join-Path $local 'manuscript.docx'
$localPdf=Join-Path $local 'manuscript.pdf'
Copy-Item -LiteralPath $source -Destination $localDoc
& (Join-Path $root 'workpapers\格式检测逐项修复_20261003\export_fresh_readonly.ps1') -Source $localDoc -LocalPdf $localPdf -ResultPdf (Join-Path $root '排版完成_五图公交示意版_20261006.pdf') -InfoPath (Join-Path $root 'workpapers\五图示意风格重绘_20261006\Word导出记录.json')
