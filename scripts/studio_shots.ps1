# Render a design in AeroSynapse Studio and save the window as a PNG.
# PrintWindow with PW_RENDERFULLCONTENT copies only the Studio window's own content, never the screen.
#   powershell -ExecutionPolicy Bypass -File scripts/studio_shots.ps1 -design results/fighter/freeform.design.json -out results/fighter/shots/freeform.png [-results surface.csv] [-view iso]
#   ... -project falcon -workflowOut shots/workflow.png   (opens the Design workflow window of projects/falcon and saves it too)
#   ... -extra "--scheme sim --field DIR" -capture "AeroSynapseSketch=shots/sketch.png;AeroSynapseSimView=shots/sim.png"   (other Studio windows)
param([string]$design = "", [string]$out, [string]$results = "", [string]$view = "iso", [int]$wait = 30, [string]$project = "", [string]$workflowOut = "",
      [string]$extra = "", [string]$capture = "")
Add-Type -AssemblyName System.Drawing
Add-Type @"
using System;
using System.Runtime.InteropServices;
public class W {
  [DllImport("user32.dll")] public static extern bool GetWindowRect(IntPtr h, out RECT r);
  [DllImport("user32.dll")] public static extern bool PrintWindow(IntPtr h, IntPtr hdc, uint flags);
  [DllImport("user32.dll")] public static extern bool MoveWindow(IntPtr h, int x, int y, int w, int hh, bool rep);
  [DllImport("user32.dll", CharSet = CharSet.Ansi)] public static extern IntPtr FindWindow(string cls, string title);
  public struct RECT { public int L, T, R, B; }
}
"@
$root = Split-Path -Parent $PSScriptRoot
$exe = Join-Path $root "bin\AeroSynapseStudio.exe"
$args = @("--view", $view, "--h", "0.028", "--clean")
if ($design -ne "") { $args = @("`"$((Resolve-Path $design).Path)`"") + $args }
if ($project -ne "") { $args += @("--project", $project) }
if ($extra -ne "") { $args += $extra.Split(" ") }
if ($results -ne "") { $args += @("--results", "`"$((Resolve-Path $results).Path)`"") }
$p = Start-Process -FilePath $exe -ArgumentList $args -PassThru
Start-Sleep -Seconds 5
$p.Refresh()
$h = $p.MainWindowHandle
[W]::MoveWindow($h, 20, 20, 1400, 860, $true) | Out-Null
Start-Sleep -Seconds $wait
$r = New-Object W+RECT
[W]::GetWindowRect($h, [ref]$r) | Out-Null
$bmp = New-Object System.Drawing.Bitmap ($r.R - $r.L), ($r.B - $r.T)
$g = [System.Drawing.Graphics]::FromImage($bmp)
$hdc = $g.GetHdc()
[W]::PrintWindow($h, $hdc, 2) | Out-Null
$g.ReleaseHdc($hdc)
New-Item -ItemType Directory -Force (Split-Path -Parent $out) | Out-Null
$bmp.Save($out)
$g.Dispose(); $bmp.Dispose()
if ($workflowOut -ne "") {
  $wh = [W]::FindWindow("AeroSynapseWorkflow", [NullString]::Value)
  if ($wh -ne [IntPtr]::Zero) {
    [W]::GetWindowRect($wh, [ref]$r) | Out-Null
    $bmp = New-Object System.Drawing.Bitmap ($r.R - $r.L), ($r.B - $r.T)
    $g = [System.Drawing.Graphics]::FromImage($bmp)
    $hdc = $g.GetHdc()
    [W]::PrintWindow($wh, $hdc, 2) | Out-Null
    $g.ReleaseHdc($hdc)
    $bmp.Save($workflowOut)
    $g.Dispose(); $bmp.Dispose()
    Write-Output "saved $workflowOut"
  }
}
if ($capture -ne "") {
  foreach ($pair in $capture.Split(";")) {
    $cls, $file = $pair.Split("=")
    $wh = [W]::FindWindow($cls, [NullString]::Value)
    if ($wh -eq [IntPtr]::Zero) { Write-Output "no window $cls"; continue }
    [W]::GetWindowRect($wh, [ref]$r) | Out-Null
    $bmp = New-Object System.Drawing.Bitmap ($r.R - $r.L), ($r.B - $r.T)
    $g = [System.Drawing.Graphics]::FromImage($bmp)
    $hdc = $g.GetHdc()
    [W]::PrintWindow($wh, $hdc, 2) | Out-Null
    $g.ReleaseHdc($hdc)
    $bmp.Save($file)
    $g.Dispose(); $bmp.Dispose()
    Write-Output "saved $file"
  }
}
Stop-Process -Id $p.Id -Force
Write-Output "saved $out"
