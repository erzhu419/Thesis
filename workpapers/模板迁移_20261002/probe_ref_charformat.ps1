param(
    [Parameter(Mandatory=$true)][string]$InputDocument,
    [Parameter(Mandatory=$true)][string]$InfoJson
)
Set-StrictMode -Version 2.0
$ErrorActionPreference='Stop'
[Console]::OutputEncoding=[Text.UTF8Encoding]::new($false)
$word=$null; $doc=$null; $owns=$false; $failure=$null
$beforePids=@(Get-Process WINWORD -ErrorAction SilentlyContinue | ForEach-Object {$_.Id})
$old=@{}
$report=[ordered]@{InputDocument=$InputDocument;Status='starting';Samples=@();Error=$null}
function Read-Font($Range) {
    return [ordered]@{EastAsia=[string]$Range.Font.NameFarEast;Ascii=[string]$Range.Font.NameAscii;
        Size=[double]$Range.Font.Size;Bold=[int]$Range.Font.Bold;Italic=[int]$Range.Font.Italic}
}
try {
    $word=New-Object -ComObject Word.Application
    $new=@(Get-Process WINWORD | Where-Object {($beforePids -notcontains $_.Id) -and ($_.Id -ne 45772)})
    if($new.Count -ne 1){throw 'No unique independent Word process.'}
    $owns=$true; $report.ProcessId=$new[0].Id
    foreach($n in @('UpdateLinksAtOpen','UpdateLinksAtPrint','UpdateFieldsAtPrint','Pagination')) {
        $old[$n]=$word.Options.$n; $word.Options.$n=$false
    }
    $old.AutomationSecurity=$word.AutomationSecurity
    $old.DisplayAlerts=$word.DisplayAlerts
    $word.AutomationSecurity=3; $word.DisplayAlerts=0; $word.Visible=$false
    $word.ScreenUpdating=$false
    $doc=$word.Documents.Open($InputDocument,$false,$false,$false)
    if($doc.ReadOnly){throw 'Input copy is read-only.'}
    $toc=$doc.TablesOfContents.Item(1)
    for($offset=-1;$offset -le 1;$offset++) {
        $pos=$toc.Range.End+$offset
        $range=$doc.Range($pos,$pos+1)
        $paragraph=$range.Paragraphs.Item(1)
        $codes=@(([string]$paragraph.Range.Text).ToCharArray() | ForEach-Object {[int]$_}) -join ','
        Write-Output ('TOC_END offset={0} codes={1} endField={2}' -f $offset,$codes,
            (([string]$paragraph.Range.WordOpenXML) -match 'fldCharType="end"'))
    }
    $selected=@{}
    $specs=@(
        @{Role='body_appendix';Needle='鲁棒强化学习从模型集合';Bookmark='appA';Size=12},
        @{Role='table_appendix';Needle='固定贝尔曼算子，';Bookmark='appA';Size=10.5}
    )
    foreach($spec in $specs) {
        $range=$doc.Content.Duplicate
        $find=$range.Find
        $find.ClearFormatting()
        $find.Text=$spec.Needle; $find.Forward=$true; $find.Wrap=0
        $find.MatchWildcards=$false; $find.MatchWholeWord=$false; $find.MatchCase=$false
        if(-not $find.Execute()){throw "Cannot find sample paragraph: $($spec.Role)"}
        $paragraph=$range.Paragraphs.Item(1).Range
        $context=[string]$paragraph.Text
        Write-Output ('SAMPLE_FOUND {0} fields={1}' -f $spec.Role,$paragraph.Fields.Count)
        for($i=1;$i -le $paragraph.Fields.Count;$i++) {
            $f=$paragraph.Fields.Item($i)
            if(([string]$f.Code.Text) -match ('^\s*REF\s+'+[regex]::Escape($spec.Bookmark)+'\b')) {
                $selected[$spec.Role]=[ordered]@{Field=$f;ExpectedSize=$spec.Size;Context=$context}
                break
            }
        }
        if(-not $selected.ContainsKey($spec.Role)){throw "Sample paragraph has no expected REF: $($spec.Role): $context"}
    }
    $samples=[Collections.Generic.List[object]]::new()
    foreach($role in @('body_appendix','table_appendix')) {
        $sample=$selected[$role]; $f=$sample.Field
        $code=[string]$f.Code.Text
        if($code -notmatch '\\\*\s+CHARFORMAT'){throw "Missing CHARFORMAT: $code"}
        $textBefore=[string]$f.Result.Text
        $fontBefore=Read-Font $f.Result
        $ok=$f.Update()
        $fontAfter=Read-Font $f.Result
        $textAfter=[string]$f.Result.Text
        $pass=($ok -ne $false) -and ($textBefore -ceq $textAfter) -and
            ($fontAfter.EastAsia -eq '宋体') -and ($fontAfter.Ascii -eq 'Times New Roman') -and
            ($fontAfter.Size -eq $sample.ExpectedSize) -and ($fontAfter.Bold -eq 0)
        foreach($key in @('EastAsia','Ascii','Size','Bold','Italic')) {
            if($fontBefore[$key] -ne $fontAfter[$key]){$pass=$false}
        }
        $samples.Add([ordered]@{Role=$role;Code=$code;Context=$sample.Context;
            BeforeText=$textBefore;AfterText=$textAfter;BeforeFont=$fontBefore;AfterFont=$fontAfter;Pass=$pass})
        $report.Samples=$samples.ToArray()
        Write-Output ('REF_SAMPLE {0} pass={1} size={2}' -f $role,$pass,$fontAfter.Size)
        if(-not $pass){throw "Representative REF format changed: $role"}
    }
    $doc.Save()
    $report.Status='passed_two_actual_ref_updates'
} catch {
    $failure=$_; $report.Status='failed'; $report.Error=$_.Exception.Message
} finally {
    if($null -ne $doc){$dontSave=0;$doc.Close([ref]$dontSave)}
    if(($null -ne $word) -and $owns) {
        foreach($n in @('UpdateLinksAtOpen','UpdateLinksAtPrint','UpdateFieldsAtPrint','Pagination')) {
            if($old.ContainsKey($n)){$word.Options.$n=$old[$n]}
        }
        if($old.ContainsKey('AutomationSecurity')){$word.AutomationSecurity=$old.AutomationSecurity}
        if($old.ContainsKey('DisplayAlerts')){$word.DisplayAlerts=$old.DisplayAlerts}
        $dontSave=0;$word.Quit([ref]$dontSave)
    }
    [IO.File]::WriteAllText($InfoJson,($report|ConvertTo-Json -Depth 10),[Text.UTF8Encoding]::new($false))
}
if($null -ne $failure){throw $failure}
