$ErrorActionPreference='Stop'
[Console]::OutputEncoding=[Text.UTF8Encoding]::new($false)
$root='\\wsl.localhost\Ubuntu-22.04\home\erzhu419\mine_code\Thesis'
$work=Join-Path $root 'workpapers\本人论文引用补充_20261005'
$target=Join-Path $root 'output\排版完成_导师意见复核_20261005.docx'
$resultPdf=Join-Path $root 'output\排版完成_导师意见复核_20261005.pdf'
$localFolder=Join-Path $env:TEMP ('thesis-citations-'+[guid]::NewGuid().ToString('N'))
New-Item -ItemType Directory -Path $localFolder | Out-Null
$localDoc=Join-Path $localFolder 'thesis.docx'
$localPdf=Join-Path $localFolder 'thesis.pdf'
Copy-Item -LiteralPath $target -Destination $localDoc
[IO.File]::WriteAllText((Join-Path $work '本地导出路径.txt'),$localFolder)
$word=$null;$doc=$null
try {
 $word=New-Object -ComObject Word.Application
 $word.Visible=$false;$word.DisplayAlerts=0
 Write-Output 'UPDATE_OPEN'
 $doc=$word.Documents.Open($localDoc,$false,$false,$false)
 $doc.Fields.Update()|Out-Null
 foreach($toc in $doc.TablesOfContents){$toc.Update()|Out-Null}
 $doc.Repaginate()
 foreach($toc in $doc.TablesOfContents){$toc.UpdatePageNumbers()|Out-Null}
 $doc.Save()
 Write-Output ('UPDATED_PAGES='+$doc.ComputeStatistics(2))
 $doc.Close(0);$doc=$null
 $word.Quit();$word=$null
 Copy-Item -LiteralPath $localDoc -Destination $target -Force
} finally {
 if($null -ne $doc){$doc.Close(0)}
 if($null -ne $word){$word.Quit()}
}
$exporter=Join-Path $root 'workpapers\格式检测逐项修复_20261003\export_fresh_readonly.ps1'
& $exporter -Source $localDoc -LocalPdf $localPdf -ResultPdf $resultPdf -InfoPath (Join-Path $work 'Word导出.json')

Remove-Item -LiteralPath $localFolder -Recurse -Force
