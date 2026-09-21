# Build a Windows .exe with PyInstaller. Run on a Windows 10/11 machine.
#
# IMPORTANT:
#   1) Copy the project to a Windows-native path FIRST (do NOT build from
#      \\wsl.localhost\... UNC paths):
#         git clone <repo> C:\projects\mockingbird     # or copy the folder
#   2) Use a Windows Python (python.org, "Add to PATH" checked), Python 3.11+.
#   3) If PowerShell refuses to run the script (Execution Policy), run:
#         powershell -ExecutionPolicy Bypass -File scripts\build_windows.ps1
#      or permanently:
#         Set-ExecutionPolicy -Scope CurrentUser RemoteSigned
#
# GPU (CUDA) is the DEFAULT build and needs an NVIDIA driver >= 550 and a
# CUDA-capable GPU. Pass -Cpu to produce a smaller CPU-only exe instead.
param(
    # Also build the Inno Setup installer (requires ISCC.exe on PATH or in
    # the default Program Files location). The PyInstaller dist\ outputs
    # must already exist - this script builds them first anyway.
    [switch]$Installer
)
$ErrorActionPreference = "Stop"

function Invoke-Pip {
    param(
        [Parameter(Mandatory = $true)]
        [string[]]$Arguments
    )
    python -m pip @Arguments
    if ($LASTEXITCODE -ne 0) {
        throw "pip failed with exit code ${LASTEXITCODE}: python -m pip $($Arguments -join ' ')"
    }
}

$Root = Split-Path -Parent $PSScriptRoot
Set-Location $Root
Write-Host "Building in: $Root"

if (-not (Get-Command python -ErrorAction SilentlyContinue)) {
    throw "python not found on PATH. Install Python 3.11+ from python.org and check 'Add to PATH'."
}

python -c "import sys; assert sys.platform.startswith('win'), 'You are not running a Windows Python. Use the Windows interpreter, not WSL/UNC paths.'"

python -m pip install --upgrade pip
if ($LASTEXITCODE -ne 0) {
    throw "pip upgrade failed with exit code $LASTEXITCODE."
}

# Baseline install.
Invoke-Pip -Arguments @("install", "-e", ".[dev]")
Invoke-Pip -Arguments @("install", "pyinstaller")

# Drop any previously-resolved nvidia-* packages first: pip does not
# downgrade on plain `install` if a newer version is already present, and a
# 12.9 nvrtc left over from an earlier build would ship broken DLLs.
python -m pip uninstall -y nvidia-cudnn-cu12 nvidia-cublas-cu12 nvidia-cuda-nvrtc-cu12 nvidia-cuda-runtime-cu12 nvidia-cufft-cu12 nvidia-curand-cu12 2>$null
if ($LASTEXITCODE -ne 0) { Write-Host "(no nvidia packages were installed - clean env)" }

# cuDNN 9 + the CUDA runtime DLLs ctranslate2 needs for float16/int8 on GPU.
# IMPORTANT: pin the nvidia-* stack to CUDA 12.4 - without torch in the env
# (removed with GigaAM) pip resolves nvidia-cuda-nvrtc-cu12 to the NEWEST
# release (12.9), whose DLLs require driver >= 575. On a 550-era driver the
# DLL loads but fails to initialize (WinError 5) and ctranslate2 dies with
# "CUDA unavailable". Same 12.4 line the old torch cu124 install enforced.
Invoke-Pip -Arguments @(
    "install",
    "nvidia-cudnn-cu12==9.1.0.70",
    "nvidia-cublas-cu12==12.4.5.8",
    "nvidia-cuda-nvrtc-cu12==12.4.127",
    "nvidia-cuda-runtime-cu12==12.4.127",
    "nvidia-cufft-cu12==11.2.1.3",
    "nvidia-curand-cu12==10.3.5.147"
)
# faster-whisper's ctranslate2 wheel from PyPI already ships CUDA 12 GPU
# support on Windows; there is no separate -cu12 package to install.
# Verify the installed binary can see the GPU so GPU inference really works.
$ct2Devices = python -c "import ctranslate2; print(ctranslate2.get_cuda_device_count())" 2>$null
if ($LASTEXITCODE -ne 0 -or $ct2Devices -notmatch '^\d+$') {
    Write-Host "WARNING: could not probe ctranslate2 CUDA support (is faster-whisper installed?)."
    Write-Host "        faster-whisper will fall back to CPU."
} elseif ([int]$ct2Devices -eq 0) {
    Write-Host "NOTE: ctranslate2 reports 0 CUDA devices on this machine."
    Write-Host "      faster-whisper will run on CPU here."
} else {
    Write-Host "ctranslate2 sees $ct2Devices CUDA device(s); faster-whisper GPU inference enabled."
}

# Whisper models are downloaded at first run, not bundled.
python -m PyInstaller --clean --noconfirm scripts\mockingbird.spec
if ($LASTEXITCODE -ne 0) {
    throw "PyInstaller failed with exit code $LASTEXITCODE. Fix the error above and re-run."
}
if (-not (Test-Path "dist\mockingbird\mockingbird.exe")) {
    throw "Build produced no executable: dist\mockingbird\mockingbird.exe is missing."
}

Write-Host ""
Write-Host "Build complete: dist\mockingbird\"
Write-Host "Run: dist\mockingbird\mockingbird.exe"

# --- Optional: Inno Setup installer ----------------------------------------
if ($Installer) {
    $iscc = Get-Command "ISCC.exe" -ErrorAction SilentlyContinue
    if (-not $iscc) {
        $candidates = @(
            "${env:ProgramFiles(x86)}\Inno Setup 6\ISCC.exe",
            "$env:ProgramFiles\Inno Setup 6\ISCC.exe"
        )
        $found = $candidates | Where-Object { Test-Path $_ } | Select-Object -First 1
        if ($found) { $iscc = $found } else {
            throw "Inno Setup compiler (ISCC.exe) not found. Install Inno Setup 6 (https://jrsoftware.org/isdl.php) or omit -Installer."
        }
    } else { $iscc = $iscc.Source }

    New-Item -ItemType Directory -Force -Path "installer" | Out-Null
    & $iscc "scripts\installer.iss"
    if ($LASTEXITCODE -ne 0) {
        throw "Inno Setup failed with exit code $LASTEXITCODE."
    }
    Write-Host ""
    Write-Host "Installer complete: installer\"
}
