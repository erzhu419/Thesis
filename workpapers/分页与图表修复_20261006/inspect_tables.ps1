param([string]$Source,[string]$InfoPath,[string]$IndicesPath)
$ErrorActionPreference='Stop'
[Console]::OutputEncoding=[Text.UTF8Encoding]::new($false)
$before=@(Get-Process WINWORD -ErrorAction SilentlyContinue|ForEach-Object {$_.Id})
$word=$null;$doc=$null;$owned=$false
try{
 $word=New-Object -ComObject Word.Application
 $new=@(Get-Process WINWORD|Where-Object {$before -notcontains $_.Id})
 if($new.Count -ne 1){throw 'Cannot identify owned Word process'}
 $owned=$true;$word.Visible=$false;$word.DisplayAlerts=0;$word.ScreenUpdating=$false
 $doc=$word.Documents.Open($Source,$false,$true,$false)
 $doc.Repaginate()
 $data=@();$indices=Get-Content -LiteralPath $IndicesPath -Raw -Encoding UTF8|ConvertFrom-Json
 foreach($item in $indices){
  $t=$doc.Tables.Item($item.index);$r=$t.Range.Duplicate;$r.Collapse(1)
  $sp=$r.Information(3);$sy=$r.Information(6)
  $r=$t.Range.Duplicate;$r.End=$r.End-1;$r.Collapse(0)
  $ep=$r.Information(3);$ey=$r.Information(6)
  $data += [ordered]@{Index=$item.index;Caption=$item.caption;StartPage=$sp;StartY=$sy;EndPage=$ep;EndY=$ey}
 }
 [IO.File]::WriteAllText($InfoPath,($data|ConvertTo-Json -Depth 5),[Text.UTF8Encoding]::new($false))
 Write-Output ('TABLES='+$data.Count+' SPLIT='+@($data|Where-Object {$_.StartPage -ne $_.EndPage}).Count)
}finally{
 if($null -ne $doc){$save=0;$doc.Close([ref]$save)}
 if($owned){$save=0;$word.Quit([ref]$save)}
}
