$ErrorActionPreference='Stop'
[Console]::OutputEncoding=[Text.UTF8Encoding]::new($false)
$localFolder='C:\Users\erzhu\AppData\Local\Temp\thesis-abstract-locked-4aadffd09d9f4757b8a7b5dcffa55b0b'
$localDoc=Join-Path $localFolder 'thesis.docx'
$localPdf=Join-Path $localFolder 'thesis.pdf'
$target='D:\Paper\Thesis\排版完成_重新修订_保留摘要_20261005.docx'
$resultPdf='D:\Paper\Thesis\排版完成_重新修订_保留摘要_20261005.pdf'
Copy-Item -LiteralPath $target -Destination $localDoc -Force
& '\\wsl.localhost\Ubuntu-22.04\home\erzhu419\mine_code\Thesis\workpapers\格式检测逐项修复_20261003\export_fresh_readonly.ps1' -Source $localDoc -LocalPdf $localPdf -ResultPdf $resultPdf -InfoPath '\\wsl.localhost\Ubuntu-22.04\home\erzhu419\mine_code\Thesis\workpapers\重新修订_保留摘要_20261005\Word导出.json'
Remove-Item -LiteralPath $localFolder -Recurse -Force
