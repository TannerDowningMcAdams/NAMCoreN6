<#
.SYNOPSIS
  Build a flash model pack from every .nam in namb/active_models.

.DESCRIPTION
  Runs the three-stage pipeline end to end:

      active_models/*.nam
          -> split_slimmable  -> build/split_models/*.nam
          -> nam2namb         -> build/namb_models/*.namb
          -> nambpack         -> build/modelpack.bin

  Every A2 model ships as a SlimmableContainer wrapping A2-Lite (channels=3)
  and A2-Full (channels=8). Plain WaveNet .nam files are passed through
  unchanged, so both kinds can sit in active_models together.

  nam2namb reads a container directly now (-c <channels>), so the split is not
  what makes conversion possible any more. It stays because it names each
  submodel <base>_ch<N>.nam, and the -Channels selection below is a filename
  match -- one pass over active_models yields both packs.

  The intermediate directories are wiped on every run. That is deliberate: a
  model deleted from active_models must also disappear from the pack, and a
  stale .namb left behind from a previous run would silently keep being flashed.

.PARAMETER Channels
  Which submodels to pack: 3 (A2-Lite), 8 (A2-Full), or all. Selection is by
  the _ch<N> suffix that split_slimmable writes.

.PARAMETER Output
  Pack image path. Defaults to build/modelpack.bin, which is what the
  "Loader: NAMB ModelPack" task flashes.

.PARAMETER KeepIntermediates
  Skip the wipe, to inspect what the split and convert stages produced.
#>
[CmdletBinding()]
param(
    [ValidateSet('3', '8', 'all')]
    [string]$Channels = '3',

    [string]$Output,

    [switch]$KeepIntermediates
)

$ErrorActionPreference = 'Stop'

# tools/ -> namb/
$NambRoot   = Split-Path -Parent $PSScriptRoot
$BuildDir   = Join-Path $NambRoot 'build'
$BinDir     = Join-Path $BuildDir 'bin'
$ActiveDir  = Join-Path $NambRoot 'active_models'
$SplitDir   = Join-Path $BuildDir 'split_models'
$NambDir    = Join-Path $BuildDir 'namb_models'

if (-not $Output) { $Output = Join-Path $BuildDir 'modelpack.bin' }

function Fail([string]$msg) { Write-Host "  ERROR: $msg" -ForegroundColor Red; exit 1 }

# --- Tools -------------------------------------------------------------------
$split   = Join-Path $BinDir 'split_slimmable.exe'
$convert = Join-Path $BinDir 'nam2namb.exe'
$pack    = Join-Path $BinDir 'nambpack.exe'

foreach ($t in @($split, $convert, $pack)) {
    if (-not (Test-Path $t)) {
        Write-Host "  Host tools not built." -ForegroundColor Yellow
        Write-Host "    cmake -S NAMCoreN6/namb/tools -B NAMCoreN6/namb/build"
        Write-Host "    cmake --build NAMCoreN6/namb/build"
        Fail "missing $(Split-Path -Leaf $t)"
    }
}

if (-not (Test-Path $ActiveDir)) { Fail "no active_models directory at $ActiveDir" }

$sources = @(Get-ChildItem -Path $ActiveDir -Filter *.nam -File | Sort-Object Name)
if ($sources.Count -eq 0) { Fail "no .nam files in $ActiveDir" }

Write-Host "=== Source models ($($sources.Count)) ===" -ForegroundColor Cyan
$sources | ForEach-Object { Write-Host ("    {0,-46} {1,9:N0} bytes" -f $_.Name, $_.Length) }

# --- 1. Split ----------------------------------------------------------------
# Stale intermediates are worse than a slow rebuild: they get packed and
# flashed without ever appearing in active_models.
foreach ($d in @($SplitDir, $NambDir)) {
    if ((Test-Path $d) -and -not $KeepIntermediates) { Remove-Item $d -Recurse -Force }
    New-Item -ItemType Directory -Force -Path $d | Out-Null
}

Write-Host "`n=== 1. Split containers ===" -ForegroundColor Cyan
foreach ($src in $sources) {
    $out = & $split $src.FullName $SplitDir 2>&1
    if ($LASTEXITCODE -eq 0) {
        $out | Where-Object { $_ -match 'wrote|split' } | ForEach-Object { Write-Host "    $_" }
    }
    else {
        # Not a container -- a plain WaveNet .nam needs no splitting, so hand it
        # to the next stage untouched rather than treating this as a failure.
        Copy-Item $src.FullName (Join-Path $SplitDir $src.Name) -Force
        Write-Host "    passthrough (not a container): $($src.Name)"
    }
}

$splits = @(Get-ChildItem -Path $SplitDir -Filter *.nam -File | Sort-Object Name)
if ($splits.Count -eq 0) { Fail "split stage produced nothing" }

# --- 2. Convert --------------------------------------------------------------
Write-Host "`n=== 2. Convert to .namb ===" -ForegroundColor Cyan
foreach ($s in $splits) {
    $dst = Join-Path $NambDir ($s.BaseName + '.namb')
    $out = & $convert $s.FullName $dst 2>&1
    if ($LASTEXITCODE -ne 0) {
        $out | ForEach-Object { Write-Host "    $_" -ForegroundColor Red }
        Fail "nam2namb failed on $($s.Name)"
    }
    $size = (Get-Item $dst).Length
    Write-Host ("    {0,-46} {1,9:N0} bytes" -f $s.BaseName, $size)
}

# --- 3. Select and pack ------------------------------------------------------
$all = @(Get-ChildItem -Path $NambDir -Filter *.namb -File | Sort-Object Name)

if ($Channels -eq 'all') {
    $selected = $all
    $label = 'all channel counts'
}
else {
    $selected = @($all | Where-Object { $_.BaseName -match "_ch$Channels$" })
    $label = "channels=$Channels"
}

Write-Host "`n=== 3. Pack ($label) ===" -ForegroundColor Cyan

if ($selected.Count -eq 0) {
    Write-Host "    No models matched $label." -ForegroundColor Yellow
    $unmatched = @($all | Where-Object { $_.BaseName -notmatch '_ch\d+$' })
    if ($unmatched.Count -gt 0) {
        Write-Host "    Note: these carry no _ch<N> suffix, so channel selection skips them:" -ForegroundColor Yellow
        $unmatched | ForEach-Object { Write-Host "      $($_.Name)" }
        Write-Host "    Use -Channels all to include them." -ForegroundColor Yellow
    }
    Fail "nothing to pack"
}

& $pack -o $Output ($selected.FullName)
if ($LASTEXITCODE -ne 0) { Fail "nambpack failed" }

Write-Host "`n  Flash it with:  Loader: NAMB ModelPack (0x90200000)" -ForegroundColor Green
