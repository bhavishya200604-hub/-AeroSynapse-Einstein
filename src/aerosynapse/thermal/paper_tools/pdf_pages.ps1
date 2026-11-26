# Render pages of a PDF to PNG with the Windows built-in PDF engine (Windows.Data.Pdf), for visual checks of the
# Word-exported papers (no external tools).
#   powershell -NoProfile -ExecutionPolicy Bypass -File paper\tools\pdf_pages.ps1 -Pdf paper\dist\paper2.pdf -OutDir paper\paper2\build\pages [-Width 1400]
param([Parameter(Mandatory = $true)][string]$Pdf, [Parameter(Mandatory = $true)][string]$OutDir, [int]$Width = 1400)
$ErrorActionPreference = "Stop"
Add-Type -AssemblyName System.Runtime.WindowsRuntime
$null = [Windows.Storage.StorageFile, Windows.Storage, ContentType = WindowsRuntime]
$null = [Windows.Data.Pdf.PdfDocument, Windows.Data.Pdf, ContentType = WindowsRuntime]
$null = [Windows.Storage.Streams.InMemoryRandomAccessStream, Windows.Storage.Streams, ContentType = WindowsRuntime]
$asTask = ([System.WindowsRuntimeSystemExtensions].GetMethods() | Where-Object {
        $_.Name -eq "AsTask" -and $_.GetParameters().Count -eq 1 -and $_.GetParameters()[0].ParameterType.Name -eq "IAsyncOperation``1" })[0]
$asTaskAction = ([System.WindowsRuntimeSystemExtensions].GetMethods() | Where-Object {
        $_.Name -eq "AsTask" -and $_.GetParameters().Count -eq 1 -and $_.GetParameters()[0].ParameterType.Name -eq "IAsyncAction" })[0]
function Await($op, [Type]$t) { $task = $asTask.MakeGenericMethod($t).Invoke($null, @($op)); $task.Wait(-1) | Out-Null; $task.Result }
function AwaitAction($op) { $task = $asTaskAction.Invoke($null, @($op)); $task.Wait(-1) | Out-Null }

$full = [System.IO.Path]::GetFullPath($Pdf)
$file = Await ([Windows.Storage.StorageFile]::GetFileFromPathAsync($full)) ([Windows.Storage.StorageFile])
$pdfDoc = Await ([Windows.Data.Pdf.PdfDocument]::LoadFromFileAsync($file)) ([Windows.Data.Pdf.PdfDocument])
New-Item -ItemType Directory -Force $OutDir | Out-Null
for ($i = 0; $i -lt $pdfDoc.PageCount; $i++) {
    $page = $pdfDoc.GetPage([uint32]$i)
    $opt = New-Object Windows.Data.Pdf.PdfPageRenderOptions
    $opt.DestinationWidth = [uint32]$Width
    $stream = New-Object Windows.Storage.Streams.InMemoryRandomAccessStream
    AwaitAction ($page.RenderToStreamAsync($stream, $opt))
    $net = [System.IO.WindowsRuntimeStreamExtensions]::AsStreamForRead($stream.GetInputStreamAt(0))
    $out = Join-Path ([System.IO.Path]::GetFullPath($OutDir)) ("page{0:D2}.png" -f ($i + 1))
    $fs = [System.IO.File]::Create($out); $net.CopyTo($fs); $fs.Close()
    $page.Dispose()
}
"rendered $($pdfDoc.PageCount) pages to $OutDir"
