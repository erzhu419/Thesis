param([string]$WorkFolder)
$ErrorActionPreference='Stop'
[Console]::OutputEncoding=[Text.UTF8Encoding]::new($false)
$local=Join-Path $env:TEMP ('thesis-figure-label-'+[Guid]::NewGuid().ToString('N'))
[void](New-Item -ItemType Directory -Path $local)
$source=Join-Path $local 'source.vsdx'
Copy-Item -LiteralPath (Join-Path $WorkFolder '图6-1修改前.vsdx') -Destination $source
$app=$null;$doc=$null
try {
 $app=New-Object -ComObject Visio.Application
 $app.Visible=$false
 $app.AlertResponse=1
 $app.EventsEnabled=$false
 Write-Output ('OPEN '+$source)
 $doc=$app.Documents.OpenEx($source,192)
 Write-Output 'OPENED'
 $page=$doc.Pages.Item(1)
 $upper=$page.Shapes.ItemFromID(457)
 $lower=$page.Shapes.ItemFromID(458)
 if($upper.Text.Trim() -ne '(a)源结构复用与模型选择' -or $lower.Text.Trim() -ne '(a)源结构复用与模型选择'){
  throw 'Unexpected figure labels.'
 }
 $lower.Text='(b)模型生成与置信加权更新'
 Write-Output 'LABEL CHANGED'
 $doc.SaveAs((Join-Path $local 'corrected.vsdx'))
 Write-Output 'SAVED'
 $page.Export((Join-Path $local 'corrected.emf'))
 $page.Export((Join-Path $local 'corrected.png'))
 foreach($extension in @('vsdx','emf','png')){
  Copy-Item -LiteralPath (Join-Path $local ('corrected.'+$extension)) -Destination (Join-Path $WorkFolder ('图6-1标记校订.'+$extension))
 }
 Write-Output ('Visio '+$app.Version+'; corrected lower-panel label.')
} finally {
 if($doc){$doc.Close()}
 if($app){$app.Quit()}
}
