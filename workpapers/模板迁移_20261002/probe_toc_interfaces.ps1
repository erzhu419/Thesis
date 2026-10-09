param([string]$InputDocument)
$ErrorActionPreference='Stop'
[Console]::OutputEncoding=[Text.UTF8Encoding]::new($false)
$word=$null; $doc=$null; $owns=$false
$before=@(Get-Process WINWORD -ErrorAction SilentlyContinue | ForEach-Object {$_.Id})
try {
    $word=New-Object -ComObject Word.Application
    $new=@(Get-Process WINWORD | Where-Object {($before -notcontains $_.Id) -and ($_.Id -ne 45772)})
    if($new.Count -ne 1){throw 'No independent Word PID'}
    $owns=$true
    $old=$word.Options.UpdateLinksAtOpen
    $word.Options.UpdateLinksAtOpen=$false
    $word.DisplayAlerts=0; $word.Visible=$false
    $doc=$word.Documents.Open($InputDocument,$false,$true,$false)
    Write-Output ('WORD_PID='+$new[0].Id)
    for($level=1;$level -le 3;$level++) {
        $style=$doc.Styles.Item(-19-$level)
        Write-Output ('STYLE '+$level+' '+$style.NameLocal)
    }
    $count=0
    for($pi=1;$pi -le 3;$pi++) {
        $p=$doc.TablesOfContents.Item(1).Range.Paragraphs.Item($pi)
        Write-Output ("PARA_TEXT "+$p.Range.Text)
        $style=$p.Style
        Write-Output ("STYLE_VALUE_TYPE "+$style.GetType().FullName+" VALUE "+[string]$style)
        if($style -is [string]){$name=$style}else{$name=$style.NameLocal}
        Write-Output ('PARAGRAPH_STYLE '+$name)
        $p.Range.Font.Size=12
        $f=$p.Format
        $f.LineSpacingRule=4; $f.LineSpacing=20
        $f.SpaceBeforeAuto=0; $f.SpaceAfterAuto=0
        $f.SpaceBefore=6; $f.SpaceAfter=6
        $f.CharacterUnitLeftIndent=0; $f.CharacterUnitFirstLineIndent=0
        $f.FirstLineIndent=0; $f.LeftIndent=0; $f.RightIndent=0
        $f.TabStops.ClearAll(); [void]$f.TabStops.Add(414.8,2,1)
        $count++
        if($count -ge 3){break}
    }
    Write-Output 'TOC_INTERFACE_OK'
} finally {
    if($null -ne $doc){$dontSave=0; $doc.Close([ref]$dontSave)}
    if(($null -ne $word)-and $owns){$word.Options.UpdateLinksAtOpen=$old;$dontSave=0;$word.Quit([ref]$dontSave)}
}
