$ErrorActionPreference='Stop'
[Console]::OutputEncoding=[Text.UTF8Encoding]::new($false)
$root='\\wsl.localhost\Ubuntu-22.04\home\erzhu419\mine_code\Thesis'
$before=@(Get-Process WINWORD -ErrorAction SilentlyContinue|ForEach-Object {$_.Id})
$word=$null;$doc=$null;$owned=$false
$local=Join-Path $env:TEMP ('thesis-comments-'+[Guid]::NewGuid().ToString('N'))
New-Item -ItemType Directory -Path $local|Out-Null
Copy-Item -LiteralPath (Join-Path $root '作者批注逐条解释_20261006.docx') -Destination (Join-Path $local 'explanations.docx')
try {
 $word=New-Object -ComObject Word.Application
 $new=@(Get-Process WINWORD|Where-Object {$before -notcontains $_.Id})
 if($new.Count -ne 1){throw 'Cannot identify own Word process.'}
 $owned=$true;$word.Visible=$false;$word.DisplayAlerts=0
 $doc=$word.Documents.Open((Join-Path $local 'explanations.docx'),$false,$true,$false)
 $doc.Repaginate();Write-Output ('EXPLANATION_PAGES='+$doc.ComputeStatistics(2))
 $doc.ExportAsFixedFormat((Join-Path $local 'explanations.pdf'),17)
 Copy-Item -LiteralPath (Join-Path $local 'explanations.pdf') -Destination (Join-Path $root '作者批注逐条解释_20261006.pdf')
} finally {
 if($doc -ne $null){$doc.Close(0)}
 if($word -ne $null -and $owned){$word.Quit(0)}
}
