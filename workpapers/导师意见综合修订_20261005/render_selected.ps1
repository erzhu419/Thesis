$ErrorActionPreference='Stop'
$folder='\\wsl.localhost\Ubuntu-22.04\home\erzhu419\mine_code\Thesis\workpapers\导师意见综合修订_20261005'
$localFolder=Join-Path $env:TEMP 'thesis-render-cca3060154d44a18939ccb5b554f543f'
$word=$null
$doc=$null
try {
 $word=New-Object -ComObject Word.Application
 $word.Visible=$false
 $word.DisplayAlerts=0
 $doc=$word.Documents.Open((Join-Path $localFolder 'thesis.docx'),$false,$true,$false)
 $pages=New-Object 'System.Collections.Generic.HashSet[int]'
 foreach($term in @('表6-16','B-25','D-58','对角执行器增益','运行长度索引','F.4 补充实验')) {
  $range=$doc.Content.Duplicate
  $range.Find.ClearFormatting()
  $range.Find.Text=$term
  $range.Find.Forward=$true
  $range.Find.Wrap=0
  if($range.Find.Execute()) {
   $page=$range.Information(3)
   [void]$pages.Add([int]$page)
   Write-Output ($term+': '+$page)
  }
 }
 foreach($page in ($pages | Sort-Object)) {
  $file=Join-Path $localFolder ('QA_'+$page+'.pdf')
  Write-Output ('EXPORT_PAGE '+$page)
  $doc.ExportAsFixedFormat($file,17,$false,0,3,$page,$page,0,$true,$false,0,$false,$true,$false)
  Copy-Item -LiteralPath $file -Destination $folder -Force
 }
 $doc.Close(0);$doc=$null
 $word.Quit();$word=$null
 Write-Output 'QA_DONE'
} finally {
 if($null -ne $doc){$doc.Close(0)}
 if($null -ne $word){$word.Quit()}
}
