# ==============================================================================
# Querity One-Line Installer for Windows (PowerShell)
# ==============================================================================
# Usage:
#   powershell -ExecutionPolicy ByPass -c "irm https://raw.githubusercontent.com/ScriptedLines/Querity/main/install.ps1 | iex"
# ==============================================================================

$ErrorActionPreference = "Stop"

Write-Host "   ___                  _ _         " -ForegroundColor Cyan
Write-Host "  / _ \ _   _  ___ _ __(_) |_ _   _ " -ForegroundColor Cyan
Write-Host " | | | | | | |/ _ \ '__| | __| | | |" -ForegroundColor Cyan
Write-Host " | |_| | |_| |  __/ |  | | |_| |_| |" -ForegroundColor Cyan
Write-Host "  \__\_\\__,_|\___|_|  |_|\__|\__, |" -ForegroundColor Cyan
Write-Host "                              |___/ " -ForegroundColor Cyan
Write-Host "Guarded Text-to-SQL Agent & Terminal UI for PostgreSQL`n" -ForegroundColor White

$InstallDir = Join-Path $HOME ".local\share\querity"
$BinDir = Join-Path $HOME ".local\bin"
New-Item -ItemType Directory -Force -Path $InstallDir | Out-Null
New-Item -ItemType Directory -Force -Path $BinDir | Out-Null

# 1. Determine Source (Local clone vs Remote Git repository)
$Source = ""
if ((Test-Path "pyproject.toml") -and (Select-String -Path "pyproject.toml" -Pattern 'name = "querity"' -Quiet)) {
    $Source = (Get-Location).Path
    Write-Host "-> Installing from local source: $Source" -ForegroundColor Cyan
} else {
    $RepoUrl = if ($env:QUERITY_REPO_URL) { $env:QUERITY_REPO_URL } else { "git+https://github.com/ScriptedLines/Querity.git" }
    $Source = $RepoUrl
    Write-Host "-> Installing from repository: $Source" -ForegroundColor Cyan
}

# 2. Check for uv or pipx
$UvCmd = Get-Command uv -ErrorAction SilentlyContinue
if (-not $UvCmd) {
    $LocalUv = Join-Path $BinDir "uv.exe"
    $CargoUv = Join-Path $HOME ".cargo\bin\uv.exe"
    if (Test-Path $LocalUv) {
        $UvCmd = Get-Command $LocalUv -ErrorAction SilentlyContinue
    } elseif (Test-Path $CargoUv) {
        $UvCmd = Get-Command $CargoUv -ErrorAction SilentlyContinue
    }
}

$PipxCmd = Get-Command pipx -ErrorAction SilentlyContinue

$Installed = $false

# Attempt to bootstrap uv if neither uv nor pipx is found
if (-not $UvCmd -and -not $PipxCmd) {
    Write-Host "-> Bootstrapping lightweight package runner (uv)..." -ForegroundColor Cyan
    try {
        Invoke-RestMethod https://astral.sh/uv/install.ps1 | Invoke-Expression
        $LocalUv = Join-Path $BinDir "uv.exe"
        if (Test-Path $LocalUv) {
            $UvCmd = Get-Command $LocalUv -ErrorAction SilentlyContinue
            $env:Path = "$BinDir;" + $env:Path
        }
    } catch {
        Write-Host "Notice: uv bootstrap skipped, falling back to Python." -ForegroundColor Yellow
    }
}

if ($UvCmd) {
    Write-Host "-> Installing with uv..." -ForegroundColor Cyan
    & $UvCmd.Source tool install --force $Source
    $Installed = $true
} elseif ($PipxCmd) {
    Write-Host "-> Installing with pipx..." -ForegroundColor Cyan
    & $PipxCmd.Source install --force $Source
    $Installed = $true
} else {
    # 3. Fallback: Python 3.12+ virtual environment
    $PythonCmd = $null
    foreach ($cand in @("python", "py", "python3")) {
        $cmd = Get-Command $cand -ErrorAction SilentlyContinue
        if ($cmd) {
            try {
                $verOk = & $cmd.Source -c "import sys; sys.exit(0 if sys.version_info >= (3, 12) else 1)"
                if ($LASTEXITCODE -eq 0) {
                    $PythonCmd = $cmd.Source
                    break
                }
            } catch {}
        }
    }

    if (-not $PythonCmd) {
        Write-Host "Error: Python 3.12 or newer is required to install Querity." -ForegroundColor Red
        Write-Host "Please install Python 3.12+ or uv (https://docs.astral.sh/uv/) and re-run."
        exit 1
    }

    Write-Host "-> Using Python: $PythonCmd" -ForegroundColor Cyan
    $VenvDir = Join-Path $InstallDir "venv"
    Write-Host "-> Creating clean virtual environment at $VenvDir..." -ForegroundColor Cyan

    if (Test-Path $VenvDir) {
        Remove-Item -Recurse -Force $VenvDir
    }

    try {
        & $PythonCmd -m venv $VenvDir
    } catch {
        Write-Host "-> Retrying venv creation without bundled pip..." -ForegroundColor Cyan
        Remove-Item -Recurse -Force $VenvDir -ErrorAction SilentlyContinue
        & $PythonCmd -m venv --without-pip $VenvDir
    }

    $VenvPython = Join-Path $VenvDir "Scripts\python.exe"
    if (-not (Test-Path $VenvPython)) {
        $VenvPython = Join-Path $VenvDir "bin\python.exe"
    }

    if (-not (Test-Path $VenvPython)) {
        Write-Host "Error: Failed to find virtual environment Python executable." -ForegroundColor Red
        exit 1
    }

    # Ensure pip is installed
    $hasPip = $false
    try {
        & $VenvPython -m pip --version | Out-Null
        $hasPip = ($LASTEXITCODE -eq 0)
    } catch {}

    if (-not $hasPip) {
        Write-Host "-> Initializing pip inside virtual environment..." -ForegroundColor Cyan
        try {
            & $VenvPython -m ensurepip --default-pip | Out-Null
        } catch {
            $getPipScript = Join-Path $InstallDir "get-pip.py"
            Invoke-WebRequest -Uri "https://bootstrap.pypa.io/get-pip.py" -OutFile $getPipScript
            & $VenvPython $getPipScript --quiet
            Remove-Item -Force $getPipScript -ErrorAction SilentlyContinue
        }
    }

    Write-Host "-> Upgrading packaging dependencies..." -ForegroundColor Cyan
    & $VenvPython -m pip install --upgrade pip setuptools wheel --quiet

    Write-Host "-> Installing Querity packages..." -ForegroundColor Cyan
    & $VenvPython -m pip install --upgrade $Source

    $QuerityExe = Join-Path $VenvDir "Scripts\querity.exe"
    if (Test-Path $QuerityExe) {
        Copy-Item -Force $QuerityExe (Join-Path $BinDir "querity.exe")
    }

    $Installed = $true
}

# 4. PATH Setup
$TargetExe = Join-Path $BinDir "querity.exe"
if (-not (Test-Path $TargetExe)) {
    # Check if uv placed it in uv tool dir
    $UvToolExe = Join-Path $HOME ".local\share\uv\tools\querity\Scripts\querity.exe"
    if (Test-Path $UvToolExe) {
        Copy-Item -Force $UvToolExe $TargetExe
    }
}

# Add ~/.local/bin to current process PATH
if ($env:Path -notlike "*$BinDir*") {
    $env:Path = "$BinDir;" + $env:Path
}

# Persist ~/.local/bin to User environment PATH
$UserPath = [Environment]::GetEnvironmentVariable("Path", "User")
if ($UserPath -notlike "*$BinDir*") {
    $NewUserPath = if ([string]::IsNullOrEmpty($UserPath)) { $BinDir } else { "$UserPath;$BinDir" }
    [Environment]::SetEnvironmentVariable("Path", $NewUserPath, "User")
    Write-Host "-> Added $BinDir to User PATH." -ForegroundColor Cyan
}

# Create a cmd wrapper for classic command prompt
$CmdWrapper = Join-Path $BinDir "querity.cmd"
"@echo off`n`"%~dp0querity.exe`" %*" | Out-File -FilePath $CmdWrapper -Encoding ascii

Write-Host "`nQuerity installed successfully!" -ForegroundColor Green
Write-Host "To launch the terminal UI anytime, simply type:`n" -ForegroundColor White
Write-Host "  querity`n" -ForegroundColor Cyan

