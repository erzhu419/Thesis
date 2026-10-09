param([string]$Name)
$ErrorActionPreference='Stop'
$work='\\wsl.localhost\Ubuntu-22.04\home\erzhu419\mine_code\Thesis\workpapers\分页与图表修复_20261006'
$doc=Join-Path $work ($Name+'.docx')
$pdf=Join-Path $work ($Name+'_第3章.pdf')
$info=Join-Path $work ($Name+'_Word.json')
& (Join-Path $work 'paginate.ps1') -DocumentPath $doc -PdfPath $pdf -InfoPath $info
& (Join-Path $work 'export_chapter3.ps1') -InfoPath $info -ResultPdf $pdf -ExportInfo (Join-Path $work ($Name+'_导出.json'))
