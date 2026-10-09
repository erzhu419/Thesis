$ErrorActionPreference='Stop'
[Console]::OutputEncoding=[Text.UTF8Encoding]::new($false)
$root='\\wsl.localhost\Ubuntu-22.04\home\erzhu419\mine_code\Thesis'
$work=Join-Path $root 'workpapers\作者审阅接受与批注解释_20261006'
$stem='排版完成_接受作者修订_全文统一_20261006'
$doc=Join-Path $root ($stem+'.docx')
$pdf=Join-Path $root ($stem+'.pdf')
& (Join-Path $work 'refresh_layout.ps1') -DocumentPath $doc -PdfPath $pdf -InfoPath (Join-Path $work 'Word分页记录.json')
$info=Get-Content -LiteralPath (Join-Path $work 'Word分页记录.json') -Raw -Encoding UTF8|ConvertFrom-Json
& (Join-Path $root 'workpapers\格式检测逐项修复_20261003\export_fresh_readonly.ps1') -Source (Join-Path $info.LocalStaging 'manuscript.docx') -LocalPdf (Join-Path $info.LocalStaging 'preview.pdf') -ResultPdf $pdf -InfoPath (Join-Path $work 'Word导出记录.json')
