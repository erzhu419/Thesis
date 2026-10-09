$ErrorActionPreference='Stop'
[Console]::OutputEncoding=[Text.UTF8Encoding]::new($false)
$root='\\wsl.localhost\Ubuntu-22.04\home\erzhu419\mine_code\Thesis'
$work=Join-Path $root 'workpapers\找回底稿完整修订_20261005'
$target=Join-Path $root 'output\排版完成_导师意见复核_20261005.docx'
$folder=Join-Path $env:TEMP ('thesis-recovered-'+[guid]::NewGuid().ToString('N'))
New-Item -ItemType Directory -Path $folder | Out-Null
$localDoc=Join-Path $folder 'thesis.docx'
Copy-Item -LiteralPath $target -Destination $localDoc
[IO.File]::WriteAllText((Join-Path $work '本地导出路径.txt'),$folder)
$word=$null;$doc=$null
$before=@(Get-Process WINWORD -ErrorAction SilentlyContinue | ForEach-Object {$_.Id})
try {
 $word=New-Object -ComObject Word.Application
 $new=@(Get-Process WINWORD | Where-Object {$before -notcontains $_.Id})
 if($new.Count -ne 1){throw 'Could not identify one newly owned Word process.'}
 $word.Visible=$false;$word.DisplayAlerts=0;$word.AutomationSecurity=3
 Write-Output ('UPDATE_OPEN PID='+$new[0].Id)
 $doc=$word.Documents.Open($localDoc,$false,$false,$false)
 $doc.Fields.Update()|Out-Null
 foreach($toc in $doc.TablesOfContents){$toc.Update()|Out-Null}
 $doc.Repaginate()
 foreach($toc in $doc.TablesOfContents){$toc.UpdatePageNumbers()|Out-Null}
 $doc.Save()
 $pages=$doc.ComputeStatistics(2)
 [IO.File]::WriteAllText((Join-Path $work 'Word分页.json'),(@{Pages=$pages;FieldsUpdated=$true;TOCUpdated=$true;LocalDoc=$localDoc;ProcessId=$new[0].Id}|ConvertTo-Json),[Text.UTF8Encoding]::new($false))
 Write-Output ('UPDATED_PAGES='+$pages)
 $doc.Close(0);$doc=$null
 $word.Quit();$word=$null
 Copy-Item -LiteralPath $localDoc -Destination $target -Force
} finally {
 if($null -ne $doc){$doc.Close(0)}
 if($null -ne $word){$word.Quit()}
}
