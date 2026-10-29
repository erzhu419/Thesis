param([string]$InputDocument,[string]$WorkDirectory)
$ErrorActionPreference='Stop';$word=$null;$doc=$null
try {
 $word=New-Object -ComObject Word.Application;$word.Visible=$false;$word.DisplayAlerts=0
 $doc=$word.Documents.Open($InputDocument,$false,$true)
 if(-not $doc.ReadOnly){throw 'Expected repair copy read-only.'}
 if($word.Documents.Count -ne 1){throw "Unexpected Word ownership."}
 $doc.Repaginate();$items=@();$samples=@()
 foreach($i in @(4139,2362,2363,2364,2486,2497,2498,772,132,149,153)) { $mr=$doc.OMaths.Item($i).Range;$samples+=@{math_index=$i;noProofing=$mr.NoProofing;page=$mr.Information(3)} }
 foreach($toc in $doc.TablesOfContents){
  $toc.UpdatePageNumbers()
  foreach($field in $toc.Range.Fields){
   $code=$field.Code.Text
   if($code -match 'PAGEREF\s+(_Toc[0-9]+)'){
    $items+=@{bookmark=$Matches[1];code=$code;page=$field.Result.Text.Trim()}
   }
  }
 }
 @{toc_count=$doc.TablesOfContents.Count;pages=$doc.ComputeStatistics(2);items=$items;original_saved=$false;formula_samples=$samples;math_objects=$doc.OMaths.Count} | ConvertTo-Json -Depth 5 | Out-File -Encoding utf8 (Join-Path $WorkDirectory 'toc_native_pages.json')
 Write-Output ('TOC_PAGE_FIELDS='+$items.Count)
 $doc.Close(0);$doc=$null;$word.Quit();$word=$null
}
finally{if($null -ne $doc){$doc.Close(0)};if($null -ne $word){$word.Quit()}}
