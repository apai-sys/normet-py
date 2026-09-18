<#
.SYNOPSIS
    Build dist\Normet\ with PyInstaller and wrap it in the Inno Setup installer.

.DESCRIPTION
    The Windows counterpart to packaging/macos/build_dmg.sh and
    packaging/linux/build_appimage.sh. Run from anywhere; paths are anchored on
    the repository root.

        packaging\windows\build_installer.ps1
        packaging\windows\build_installer.ps1 -Python C:\Python311\python.exe

    The interpreter must already have normet + its GUI/AutoML extras and
    PyInstaller installed:

        pip install ".[gui,flaml,lgb,geo]" pyinstaller

.NOTES
    Do not build with an Anaconda interpreter. Anaconda keeps its own
    VCRUNTIME140.dll / MSVCP140_1.dll next to python.exe, and those are older
    than the ones PySide6 ships. The loader then mixes the two sets and Qt
    fails to import with "DLL load failed ... the specified procedure could not
    be found" (ERROR_PROC_NOT_FOUND) -- at build time if you are lucky, and
    otherwise only on the end user's machine, because PyInstaller happily
    bundles whichever copy it found. Use a python.org install (this is also
    what the GitHub Actions runner provides).
#>
[CmdletBinding()]
param(
    # Interpreter holding normet + PySide6 + PyInstaller.
    [string]$Python = "python",

    # Skip PyInstaller and only recompile the installer from an existing
    # dist\Normet\ -- the slow part is the freeze, not the Inno Setup pass.
    [switch]$SkipFreeze
)

$ErrorActionPreference = "Stop"

$repo = (Resolve-Path (Join-Path $PSScriptRoot "..\..")).Path
$dist = Join-Path $repo "dist\Normet"

$version = (Select-String -Path (Join-Path $repo "pyproject.toml") `
    -Pattern '^version\s*=\s*"([^"]+)"').Matches[0].Groups[1].Value
Write-Host "==> Normet $version"

if (-not $SkipFreeze) {
    $pyExe = (Get-Command $Python).Source
    if ($pyExe -match 'anaconda|miniconda') {
        throw "Refusing to build with a conda interpreter ($pyExe) -- see the note in this script's header. Pass -Python with a python.org install."
    }
    Write-Host "==> Freezing with $pyExe"

    # Keep conda off PATH for the freeze: PyInstaller scans PATH when it
    # resolves a binary's dependencies, and anything it finds there is copied
    # into the bundle.
    $savedPath = $env:PATH
    $env:PATH = ($env:PATH -split ';' | Where-Object { $_ -and ($_ -notmatch 'anaconda|miniconda|conda') }) -join ';'
    try {
        Push-Location $repo
        & $pyExe -m PyInstaller --noconfirm (Join-Path $repo "packaging\normet_gui.spec")
        if ($LASTEXITCODE -ne 0) { throw "PyInstaller failed ($LASTEXITCODE)" }
    }
    finally {
        Pop-Location
        $env:PATH = $savedPath
    }
}

if (-not (Test-Path (Join-Path $dist "Normet.exe"))) {
    throw "$dist\Normet.exe not found -- run without -SkipFreeze first"
}

# Inno Setup 6, wherever it landed: winget installs per-user, Chocolatey (what
# CI uses) installs into Program Files (x86).
$iscc = @(
    "${env:ProgramFiles(x86)}\Inno Setup 6\ISCC.exe"
    "$env:ProgramFiles\Inno Setup 6\ISCC.exe"
    "$env:LOCALAPPDATA\Programs\Inno Setup 6\ISCC.exe"
) | Where-Object { Test-Path $_ } | Select-Object -First 1

if (-not $iscc) {
    throw "ISCC.exe not found. Install it with: winget install --id JRSoftware.InnoSetup -e"
}

Write-Host "==> Compiling installer with $iscc"
& $iscc "/DAppVersion=$version" (Join-Path $PSScriptRoot "installer.iss")
if ($LASTEXITCODE -ne 0) { throw "ISCC failed ($LASTEXITCODE)" }

$setup = Join-Path $repo "dist\normet-setup-$version.exe"
$mb = [math]::Round((Get-Item $setup).Length / 1MB, 1)
Write-Host "==> Done: $setup ($mb MB)"
