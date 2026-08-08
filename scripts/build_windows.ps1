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
    [switch]$Cpu
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

# Baseline install (deps as declared; on GPU builds we override torch below so
# the final state is the CUDA one regardless of resolver order).
Invoke-Pip -Arguments @("install", "-e", ".[gigaam,dev]")
Invoke-Pip -Arguments @("install", "pyinstaller")

if ($Cpu) {
    Write-Host ">>> Building CPU-only variant"
    # Prefer the CPU-only wheels on Windows to keep the install smaller.
    Invoke-Pip -Arguments @("install", "torch", "torchaudio", "--index-url", "https://download.pytorch.org/whl/cpu")
} else {
    Write-Host '>>> Building GPU (CUDA 12.4) variant, needs NVIDIA driver 550 or newer'
    # CUDA build of torch/torchaudio (GTX 1070 = Pascal sm_61 is supported).
    Invoke-Pip -Arguments @("install", "--force-reinstall", "torch", "torchaudio", "--index-url", "https://download.pytorch.org/whl/cu124")
    # faster-whisper's ctranslate2 wheel from PyPI already ships CUDA 12 GPU
    # support on Windows; there is no separate -cu12 package to install.
    # Verify the installed binary can see the GPU so GPU inference really works.
    $ct2Devices = python -c "import ctranslate2; print(ctranslate2.get_cuda_device_count())" 2>$null
    if ($LASTEXITCODE -ne 0 -or $ct2Devices -notmatch '^\d+$') {
        Write-Host "WARNING: could not probe ctranslate2 CUDA support (is faster-whisper installed?)."
        Write-Host "        faster-whisper will fall back to CPU; GigaAM still uses torch CUDA."
    } elseif ([int]$ct2Devices -eq 0) {
        Write-Host "NOTE: ctranslate2 reports 0 CUDA devices on this machine."
        Write-Host "      faster-whisper will run on CPU here; GigaAM still uses torch CUDA."
    } else {
        Write-Host "ctranslate2 sees $ct2Devices CUDA device(s); faster-whisper GPU inference enabled."
    }
}

# Fix a known torch + PyInstaller issue on Python 3.12 (NameError in
# torch/_numpy/_ufuncs.py) by patching the installed torch source in place, so
# PyInstaller freezes the corrected bytecode. Idempotent; see the script.
python scripts\patch_torch_sources.py
if ($LASTEXITCODE -ne 0) {
    throw "Failed to patch torch sources. See error above."
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
if (-not $Cpu) {
    Write-Host 'GPU build: the target machine needs an NVIDIA driver 550 or newer (and a CUDA-capable GPU).'
    Write-Host 'If CUDA is missing the app falls back to CPU automatically.'
}
