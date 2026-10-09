$ErrorActionPreference='Stop'
$root='\\wsl.localhost\Ubuntu-22.04\home\erzhu419\mine_code\Thesis'
$work=Join-Path $root 'workpapers\找回底稿完整修订_20261005'
$folder=[IO.File]::ReadAllText((Join-Path $work '本地导出路径.txt'))
$source=Join-Path $folder 'locked.docx'
Copy-Item -LiteralPath (Join-Path $root 'output\排版完成_导师意见复核_20261005.docx') -Destination $source -Force
& (Join-Path $root 'workpapers\格式检测逐项修复_20261003\export_fresh_readonly.ps1') -Source $source -LocalPdf (Join-Path $folder 'thesis.pdf') -ResultPdf (Join-Path $root 'output\排版完成_导师意见复核_20261005.pdf') -InfoPath (Join-Path $work 'Word导出.json')
