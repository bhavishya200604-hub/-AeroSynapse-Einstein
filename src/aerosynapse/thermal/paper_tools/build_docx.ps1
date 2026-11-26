# Build a Word document (.docx) and a PDF exported by Word from the JSON document model of paper/tools/tex2json.py.
#   powershell -NoProfile -ExecutionPolicy Bypass -File paper\tools\build_docx.ps1 -Json paper\paper2\build\doc.json -Out paper\dist\paper2
# Equations are Word equations (built up from linear format); figures are the generated PNG previews; numbers, tables
# and references come unchanged from the LaTeX sources. ASCII-only script: all text arrives through the UTF-8 JSON.
# Layout: Cambria body (matches Cambria Math), Calibri headings in navy, a title block with highlights and abstract
# panels, nomenclature / glossary as two-column tables, a running header from page 2 and page numbers.
param([Parameter(Mandatory = $true)][string]$Json, [Parameter(Mandatory = $true)][string]$Out)
$ErrorActionPreference = "Stop"
$doc = Get-Content -Raw -Encoding UTF8 $Json | ConvertFrom-Json
$outDocx = [System.IO.Path]::GetFullPath("$Out.docx")
$outPdf = [System.IO.Path]::GetFullPath("$Out.pdf")
$font = "Cambria"
$sans = "Calibri"
function RGB([int]$r, [int]$g, [int]$b) { return [int]($r + 256 * $g + 65536 * $b) }
$navy = RGB 31 56 100        # headings, labels, rules
$rule = RGB 176 192 214      # light rules
$light = RGB 236 241 248     # panel and table-header fill
$grey = RGB 95 95 95         # running header
$auto = -16777216            # wdColorAutomatic

$w = New-Object -ComObject Word.Application
$w.Visible = $false
$w.DisplayAlerts = 0
try {
    $d = $w.Documents.Add()
    $ps = $d.PageSetup
    $ps.PageWidth = 595.3; $ps.PageHeight = 841.9
    $ps.TopMargin = 72; $ps.BottomMargin = 68; $ps.LeftMargin = 68; $ps.RightMargin = 68
    $ps.HeaderDistance = 34; $ps.FooterDistance = 30

    $textW = 595.3 - 136

    $st = $d.Styles.Item(-1)            # Normal
    $st.Font.Name = $font; $st.Font.Size = 11
    $st.ParagraphFormat.SpaceAfter = 6; $st.ParagraphFormat.SpaceBefore = 0
    $st.ParagraphFormat.LineSpacingRule = 5; $st.ParagraphFormat.LineSpacing = 13.8
    $st.ParagraphFormat.Alignment = 3   # justified
    foreach ($h in @(@(-2, 13.5, 16), @(-3, 11.5, 10))) {
        $s = $d.Styles.Item([int]$h[0]); $s.Font.Name = $sans; $s.Font.Size = [single]$h[1]; $s.Font.Bold = -1
        $s.Font.Italic = 0; $s.Font.Color = $navy
        $s.ParagraphFormat.SpaceBefore = [single]$h[2]; $s.ParagraphFormat.SpaceAfter = 6; $s.ParagraphFormat.Alignment = 0
        $s.ParagraphFormat.KeepWithNext = -1
    }
    $h1b = $d.Styles.Item(-2).ParagraphFormat.Borders.Item(-3)   # a light rule under every section heading
    $h1b.LineStyle = 1; $h1b.LineWidth = 4; $h1b.Color = $rule
    $sel = $w.Selection

    # equations arrive as LaTeX: Word must be set to LaTeX equation input (Equation > Conversion > LaTeX)
    $sel.TypeText('\frac{a}{b}')
    $probe = $d.Range(0, $sel.End)
    $null = $probe.OMaths.Add($probe)
    $probe.OMaths.Item(1).BuildUp()
    $isFrac = $probe.OMaths.Item(1).Functions.Count -eq 1 -and $probe.OMaths.Item(1).Functions.Item(1).Type -eq 7
    $d.Content.Delete() | Out-Null
    if (-not $isFrac) { throw "Word's equation input is not LaTeX: set Insert > Equation > Conversion > LaTeX and rerun" }

    function Set-Fmt($r) {
        if ($r.b) { $sel.Font.Bold = -1 } else { $sel.Font.Bold = 0 }; if ($r.i) { $sel.Font.Italic = -1 } else { $sel.Font.Italic = 0 }
        if ($r.tt) { $sel.Font.Name = "Consolas" } else { $sel.Font.Name = $font }
        if ($r.sup) { $sel.Font.Superscript = -1 } else { $sel.Font.Superscript = 0 }
        if ($r.sub) { $sel.Font.Subscript = -1 } else { $sel.Font.Subscript = 0 }
        if ($r.acc) { $sel.Font.Color = $navy; $sel.Font.Name = $sans } else { $sel.Font.Color = $auto }
    }
    function Write-Math([string]$s) {
        $start = $sel.Start
        $sel.TypeText($s)
        $r = $d.Range($start, $sel.End)
        $null = $r.OMaths.Add($r)
        $om = $r.OMaths.Item(1)
        $om.BuildUp()
        if ($script:inCell) { $om.Type = 1 }   # inline inside table cells: follows the cell alignment
        $e = $om.Range.End
        $sel.SetRange($e, $e)
    }
    function Write-Runs($runs) {
        foreach ($r in $runs) {
            if ($r.k -eq "m") { Write-Math $r.s } else { Set-Fmt $r; $sel.TypeText($r.s) }
        }
        $sel.Font.Bold = 0; $sel.Font.Italic = 0; $sel.Font.Name = $font; $sel.Font.Superscript = 0; $sel.Font.Subscript = 0
        $sel.Font.Color = $auto
    }
    function New-Para([string]$style = "Normal", [double]$size = 0, [int]$align = -1) {
        $sel.EndKey(6) | Out-Null
        if ($d.Content.Text.Length -gt 1) { $sel.TypeParagraph() }
        $sel.Style = $d.Styles.Item($style)
        if ($style -eq "Normal") {   # headings keep their style's font (bold, size, colour)
            $sel.Font.Bold = 0; $sel.Font.Italic = 0; $sel.Font.Name = $font; $sel.Font.Color = $auto
            if ($size -gt 0) { $sel.Font.Size = [single]$size } else { $sel.Font.Size = 11 }
            $pf = $sel.ParagraphFormat
            $pf.Borders.Enable = 0; $pf.LeftIndent = 0; $pf.RightIndent = 0; $pf.FirstLineIndent = 0
            $pf.SpaceBefore = 0; $pf.SpaceAfter = 6; $pf.KeepWithNext = 0
        }
        if ($align -ge 0) { $sel.ParagraphFormat.Alignment = $align }
    }
    function Clean-After-Table { $sel.EndKey(6) | Out-Null }
    function Write-Label([string]$text, [double]$size = 10.5) {
        New-Para "Normal" $size 0
        $sel.Font.Name = $sans; $sel.Font.Bold = -1; $sel.Font.Color = $navy
        $sel.ParagraphFormat.SpaceBefore = 6; $sel.ParagraphFormat.SpaceAfter = 3; $sel.ParagraphFormat.KeepWithNext = -1
        $sel.TypeText($text)
        $sel.Font.Name = $font; $sel.Font.Bold = 0; $sel.Font.Color = $auto
    }
    function New-Panel([double]$size) {
        # a one-cell table with a light fill and a navy bar on the left; the selection ends inside the cell
        New-Para "Normal" $size 3
        $t = $d.Tables.Add($sel.Range, 1, 1)
        $t.Borders.Enable = 0
        $t.Shading.BackgroundPatternColor = $light
        $bl = $t.Borders.Item(-2); $bl.LineStyle = 1; $bl.LineWidth = 24; $bl.Color = $navy
        $t.LeftPadding = 10; $t.RightPadding = 10; $t.TopPadding = 6; $t.BottomPadding = 6
        $t.Range.ParagraphFormat.SpaceAfter = 3; $t.Range.ParagraphFormat.Alignment = 3
        $t.Range.Font.Size = [single]$size
        $t.Cell(1, 1).Range.Select(); $sel.Collapse(1)
        return $t
    }

    # ---- title block
    New-Para "Normal" 9 1; $sel.Font.Name = $sans; $sel.Font.Bold = -1; $sel.Font.Color = $navy
    $sel.Font.Spacing = 1.5; $sel.TypeText("RESEARCH ARTICLE"); $sel.Font.Spacing = 0
    $sel.ParagraphFormat.SpaceAfter = 8
    New-Para "Normal" 19 1; $sel.Font.Name = $sans; $sel.Font.Bold = -1; $sel.Font.Color = $navy
    $sel.ParagraphFormat.LineSpacingRule = 0; $sel.ParagraphFormat.SpaceAfter = 12
    $sel.TypeText($doc.title)
    New-Para "Normal" 11 1; $sel.TypeText($doc.authors); $sel.ParagraphFormat.SpaceAfter = 2
    New-Para "Normal" 4 1
    $rb = $sel.ParagraphFormat.Borders.Item(-3); $rb.LineStyle = 1; $rb.LineWidth = 12; $rb.Color = $navy
    $sel.ParagraphFormat.SpaceAfter = 10

    if ($doc.highlights -and $doc.highlights.Count -gt 0) {
        Write-Label "Highlights"
        $t = New-Panel 10.5
        $script:inCell = $true
        $k = 0
        foreach ($it in $doc.highlights) {
            if ($k -gt 0) { $sel.TypeParagraph() }
            $sel.Font.Size = [single]10.5
            Write-Runs $it
            $k++
        }
        $script:inCell = $false
        $t.Range.ParagraphFormat.Alignment = 0
        $t.Range.ListFormat.ApplyBulletDefault()
        Clean-After-Table
    }

    Write-Label "Abstract"
    $t = New-Panel 10.5
    $script:inCell = $true; $sel.Font.Size = [single]10.5; Write-Runs $doc.abstract; $script:inCell = $false
    Clean-After-Table
    New-Para "Normal" 10 0; $sel.ParagraphFormat.SpaceBefore = 6; $sel.ParagraphFormat.SpaceAfter = 14
    $sel.Font.Name = $sans; $sel.Font.Bold = -1; $sel.Font.Color = $navy; $sel.TypeText("Keywords  ")
    $sel.Font.Name = $font; $sel.Font.Bold = 0; $sel.Font.Color = $auto
    $sel.TypeText(($doc.keywords -replace "; ", (" " + [char]0x00B7 + " ")))

    $eqW = $textW
    foreach ($b in $doc.blocks) {
        switch ($b.t) {
            "h1" { New-Para "Heading 1"; $sel.TypeText($b.s) }
            "h2" { New-Para "Heading 2"; $sel.TypeText($b.s) }
            "p" { New-Para; Write-Runs $b.runs }
            "pagebreak" { $sel.EndKey(6) | Out-Null; $sel.InsertBreak(7) }
            "eq" {
                New-Para; $sel.ParagraphFormat.SpaceAfter = 0
                $t = $d.Tables.Add($sel.Range, 1, 2)
                $t.Borders.Enable = 0
                $t.Columns.Item(1).Width = $eqW * 0.88; $t.Columns.Item(2).Width = $eqW * 0.12
                $t.Cell(1, 1).Range.ParagraphFormat.Alignment = 1
                $t.Cell(1, 2).Range.ParagraphFormat.Alignment = 2
                $t.Cell(1, 2).VerticalAlignment = 1
                $t.Cell(1, 1).Range.Select(); $sel.Collapse(1)
                Write-Math $b.s
                $t.Cell(1, 2).Range.Text = $b.num
                Clean-After-Table
            }
            "figure" {
                New-Para "Normal" 11 1; $sel.ParagraphFormat.KeepWithNext = -1; $sel.ParagraphFormat.SpaceBefore = 6
                $pic = $sel.InlineShapes.AddPicture($b.path, $false, $true)
                $pic.LockAspectRatio = -1
                $pic.Width = [Math]::Min($textW, $textW * [double]$b.width)
                New-Para "Normal" 9.5 3; $sel.ParagraphFormat.SpaceAfter = 12
                Write-Runs $b.caption
            }
            "table" {
                New-Para "Normal" 9.5 3; $sel.ParagraphFormat.KeepWithNext = -1; $sel.ParagraphFormat.SpaceBefore = 6
                Write-Runs $b.caption
                $nr = $b.rows.Count; $nc = $b.rows[0].Count
                New-Para "Normal" 9 0
                $t = $d.Tables.Add($sel.Range, $nr, $nc)
                $t.Borders.Enable = 0
                $sz = 9; if ($b.small) { $sz = 8 }
                $t.Range.Font.Size = [single]$sz
                $t.Range.ParagraphFormat.SpaceAfter = 1; $t.Range.ParagraphFormat.Alignment = 0
                for ($i = 0; $i -lt $nr; $i++) {
                    for ($j = 0; $j -lt $nc; $j++) {
                        $cell = $t.Cell($i + 1, $j + 1)
                        $cell.Range.Select(); $sel.Collapse(1); $sel.Font.Size = [single]$sz
                        $cellRuns = $b.rows[$i][$j]
                        if ($cellRuns) { $script:inCell = $true; Write-Runs $cellRuns; $script:inCell = $false }
                    }
                }
                $tb = $t.Rows.Item(1).Borders.Item(-1); $tb.LineStyle = 1; $tb.LineWidth = 12; $tb.Color = $navy   # top rule
                if ($b.head -gt 0) {
                    $hb = $t.Rows.Item([int]$b.head).Borders.Item(-3); $hb.LineStyle = 1; $hb.Color = $navy   # below the header
                }
                $bb = $t.Rows.Item($nr).Borders.Item(-3); $bb.LineStyle = 1; $bb.LineWidth = 12; $bb.Color = $navy   # bottom rule
                for ($i = 1; $i -le [int]$b.head; $i++) {
                    $t.Rows.Item($i).Range.Font.Bold = -1
                    $t.Rows.Item($i).Shading.BackgroundPatternColor = $light
                    $t.Rows.Item($i).HeadingFormat = -1   # repeat the header on every page
                }
                $t.Rows.AllowBreakAcrossPages = 0
                foreach ($mr in @($b.merged)) {      # group headings: one merged, italic cell
                    if ($mr) {
                        $row = $t.Rows.Item([int]$mr)
                        $row.Range.Font.Italic = -1
                        $row.Cells.Merge()
                    }
                }
                $t.AllowAutoFit = -1
                $t.AutoFitBehavior(1); $t.AutoFitBehavior(2)
                Clean-After-Table
                New-Para "Normal" 4; $sel.ParagraphFormat.SpaceAfter = 6
            }
            "deftable" {
                # nomenclature / glossary: term | definition; rows without a definition are group headings
                $rows = @($b.rows); $nr = $rows.Count
                $glossary = $b.kind -eq "keyterms"
                $sz = 9.5; $w1 = 0.2; if ($glossary) { $w1 = 0.27 }
                New-Para "Normal" $sz 0
                $t = $d.Tables.Add($sel.Range, $nr, 2)
                $t.Borders.Enable = 0
                $t.Range.Font.Size = [single]$sz
                $t.Range.ParagraphFormat.SpaceAfter = 1; $t.Range.ParagraphFormat.SpaceBefore = 1
                $t.Range.ParagraphFormat.Alignment = 0
                $t.Columns.Item(1).Width = $textW * $w1; $t.Columns.Item(2).Width = $textW * (1 - $w1)
                $groups = @()
                for ($i = 0; $i -lt $nr; $i++) {
                    $row = $rows[$i]
                    $script:inCell = $true
                    $t.Cell($i + 1, 1).Range.Select(); $sel.Collapse(1); $sel.Font.Size = [single]$sz
                    Write-Runs $row.term
                    if ($row.group) {
                        $groups += ($i + 1)
                    }
                    else {
                        if ($glossary) { $t.Cell($i + 1, 1).Range.Font.Bold = -1 }
                        $t.Cell($i + 1, 2).Range.Select(); $sel.Collapse(1); $sel.Font.Size = [single]$sz
                        Write-Runs $row.def
                        if ($glossary) { $t.Cell($i + 1, 2).Range.ParagraphFormat.Alignment = 3 }
                    }
                    $script:inCell = $false
                }
                $tb = $t.Rows.Item(1).Borders.Item(-1); $tb.LineStyle = 1; $tb.LineWidth = 12; $tb.Color = $navy
                $bb = $t.Rows.Item($nr).Borders.Item(-3); $bb.LineStyle = 1; $bb.LineWidth = 12; $bb.Color = $navy
                if ($glossary) {   # thin separators between glossary entries
                    for ($i = 1; $i -lt $nr; $i++) {
                        $sb = $t.Rows.Item($i).Borders.Item(-3); $sb.LineStyle = 1; $sb.LineWidth = 2; $sb.Color = $rule
                    }
                }
                foreach ($g in $groups) {
                    $row = $t.Rows.Item([int]$g)
                    $row.Shading.BackgroundPatternColor = $light
                    $row.Range.Font.Name = $sans; $row.Range.Font.Bold = -1; $row.Range.Font.Italic = 0
                    $row.Range.Font.Color = $navy
                    $row.Cells.Merge()
                }
                $t.Rows.AllowBreakAcrossPages = 0
                Clean-After-Table
                New-Para "Normal" 4; $sel.ParagraphFormat.SpaceAfter = 6
            }
            "list" {
                foreach ($it in $b.items) {
                    New-Para
                    if ($b.ordered) { $sel.Range.ListFormat.ApplyNumberDefault() } else { $sel.Range.ListFormat.ApplyBulletDefault() }
                    Write-Runs $it
                }
                New-Para; $sel.Range.ListFormat.RemoveNumbers()
            }
            "refs" {
                $n = 0
                foreach ($it in $b.items) {
                    $n++
                    New-Para "Normal" 9.5 0
                    $sel.ParagraphFormat.LeftIndent = 22; $sel.ParagraphFormat.FirstLineIndent = -22
                    $sel.ParagraphFormat.SpaceAfter = 3
                    $sel.TypeText("[$n]`t")
                    Write-Runs $it
                }
            }
        }
    }
    # running header from page 2 (a light rule below it); page numbers centred in the footer
    $sec = $d.Sections.Item(1)
    $f = $sec.Footers.Item(1)
    $null = $f.PageNumbers.Add(1, $true)
    # PageNumbers.Add(..., FirstPage = true) clears "different first page": set it afterwards, then give the
    # title page its own footer with a page field and an empty header
    $sec.PageSetup.DifferentFirstPageHeaderFooter = -1
    $sec.Headers.Item(2).Range.Text = ""
    $fr = $sec.Footers.Item(2).Range
    $null = $fr.Fields.Add($fr, 33)   # wdFieldPage
    $sec.Footers.Item(2).Range.ParagraphFormat.Alignment = 1
    if ($doc.running) {
        $hr = $sec.Headers.Item(1).Range
        $hr.Text = $doc.running
        $hr.Font.Name = $sans; $hr.Font.Size = 8.5; $hr.Font.Italic = -1; $hr.Font.Color = $grey
        $hr.ParagraphFormat.Alignment = 2
        $hrb = $hr.ParagraphFormat.Borders.Item(-3); $hrb.LineStyle = 1; $hrb.LineWidth = 4; $hrb.Color = $rule
    }
    $d.SaveAs2($outDocx, 16)
    $d.ExportAsFixedFormat($outPdf, 17)
    "omaths: $($d.OMaths.Count); tables: $($d.Tables.Count); pictures: $($d.InlineShapes.Count); pages: $($d.ComputeStatistics(2))"
    $d.Close(0)
}
catch {
    "ERROR at line $($_.InvocationInfo.ScriptLineNumber): $($_.InvocationInfo.Line.Trim()) :: $($_.Exception.Message)"
    throw
}
finally {
    $w.Quit()
    [System.Runtime.Interopservices.Marshal]::ReleaseComObject($w) | Out-Null
}
"wrote $outDocx"
"wrote $outPdf"
