# build_msix.ps1 - Package the PyInstaller output as MSIX for the
# Microsoft Store. Run from E:\mockingbird after a CUDA build:
#   powershell -ExecutionPolicy Bypass -File scripts\build_msix.ps1
#
# Steps:
#   1. Copy dist\mockingbird -> msix\mockingbird (app payload)
#   2. Copy store icons + AppxManifest.xml into msix\
#   3. MakeAppx pack -> installer\Mockingbird-<ver>-x64.msix
#   4. SignTool sign (self-signed dev cert only for local install test;
#      the Store re-signs the package itself on submission)
#
# ASCII-only + UTF-8 BOM (PowerShell 5.1 parser constraint).
param(
    [string]$Version = "",
    [switch]$SkipSign
)

$ErrorActionPreference = "Stop"

# --- locate Windows SDK tools (MakeAppx / SignTool) -------------------
function Find-SdkTool([string]$name) {
    $roots = @(
        "${env:ProgramFiles(x86)}\Windows Kits\10\bin",
        "$env:ProgramFiles\Windows Kits\10\bin"
    )
    foreach ($root in $roots) {
        if (Test-Path $root) {
            # prefer the x64 binary: the x86 one hits 32-bit memory limits
            # (E_OUTOFMEMORY / 0x8007000e) on multi-GB CUDA payloads
            $tool = Get-ChildItem -Path "$root\*\x64" -Recurse -Filter $name -ErrorAction SilentlyContinue |
                Sort-Object FullName -Descending | Select-Object -First 1
            if (-not $tool) {
                $tool = Get-ChildItem -Path $root -Recurse -Filter $name -ErrorAction SilentlyContinue |
                    Sort-Object FullName -Descending | Select-Object -First 1
            }
            if ($tool) { return $tool.FullName }
        }
    }
    return $null
}

$makeappx = Find-SdkTool "makeappx.exe"
$signtool = Find-SdkTool "signtool.exe"
if (-not $makeappx) { throw "makeappx.exe not found - install Windows SDK (Windows App Certification Kit includes it)" }
if (-not $signtool -and -not $SkipSign) { throw "signtool.exe not found - install Windows SDK or pass -SkipSign" }
Write-Host "makeappx: $makeappx"

# --- version ----------------------------------------------------------
if (-not $Version) {
    $init = Get-Content "src\mockingbird\__init__.py" -Raw
    if ($init -match '__version__\s*=\s*"([^"]+)"') { $Version = $Matches[1] }
    else { throw "cannot detect version from src\mockingbird\__init__.py" }
}
# MSIX requires 4 components; pad if needed
$parts = $Version.Split(".")
while ($parts.Count -lt 4) { $parts += "0" }
$Version = ($parts[0..3] -join ".")
Write-Host "Package version: $Version"

# --- check PyInstaller output ----------------------------------------
$distApp = "dist\mockingbird"
if (-not (Test-Path "$distApp\mockingbird.exe")) {
    throw "dist\mockingbird\mockingbird.exe not found - run the PyInstaller build first"
}
$gpuDll = Test-Path "$distApp\_internal\ctranslate2\cudnn*.dll"
Write-Host ("Bundle flavor: " + $(if ($gpuDll) { "CUDA" } else { "CPU" }))

# --- staging ----------------------------------------------------------
$stage = "msix"
if (Test-Path $stage) { Remove-Item $stage -Recurse -Force -ErrorAction SilentlyContinue }
New-Item -ItemType Directory -Path $stage | Out-Null
New-Item -ItemType Directory -Path "$stage\assets" | Out-Null

Write-Host "Staging app payload (this can take a few minutes for the CUDA bundle)..."
Copy-Item -Path $distApp -Destination "$stage\mockingbird" -Recurse
if (-not (Test-Path "$stage\mockingbird\mockingbird.exe")) { throw "staging failed" }

Write-Host "Staging manifest + icons..."
Copy-Item "scripts\AppxManifest.xml" "$stage\AppxManifest.xml"
$icons = @{
    "StoreLogo.png"                 = "StoreLogo.png"
    "Square44x44Logo.png"           = "Square44x44.Logo.png"
    "Square71x71Logo.png"           = "Square71x71.Logo.png"
    "Square150x150Logo.png"         = "Square150x150.Logo.png"
    "Square310x310Logo.png"         = "Square310x310.Logo.png"
    "Wide310x150Logo.png"           = "Wide310x150Logo.png"
    "SplashScreen.scale-200.png"    = "SplashScreen.scale-200.png"
}
foreach ($dst in $icons.Keys) {
    $srcIcon = "assets\store\$($icons[$dst])"
    if (-not (Test-Path $srcIcon)) { throw "missing icon $srcIcon" }
    Copy-Item $srcIcon "$stage\assets\$dst"
}

# bump manifest version to match
$manifest = Get-Content "$stage\AppxManifest.xml" -Raw
$manifest = $manifest -replace 'Version="[^"]+"(\s+ProcessorArchitecture)', ("Version=`"$Version`"`$1")
Set-Content "$stage\AppxManifest.xml" $manifest -Encoding UTF8

# --- pack -------------------------------------------------------------
if (-not (Test-Path "installer")) { New-Item -ItemType Directory -Path "installer" | Out-Null }
$msixName = "Mockingbird-$Version-x64.msix"
$msixPath = "installer\$msixName"
if (Test-Path $msixPath) { Remove-Item $msixPath -Force }

Write-Host "Packing $msixName ..."
& $makeappx pack /d $stage /p $msixPath | Out-Host
if ($LASTEXITCODE -ne 0) { throw "makeappx exited with $LASTEXITCODE" }
if (-not (Test-Path $msixPath)) { throw "makeappx failed - no output at $msixPath" }
Write-Host ("Packed: " + [math]::Round((Get-Item $msixPath).Length / 1MB) + " MB")

# --- sign (dev self-signed for local install testing) -----------------
if (-not $SkipSign) {
    Write-Host "Signing with dev self-signed certificate..."
    $pfxPath = "$env:USERPROFILE\.mockingbird\msix-dev.pfx"
    $certCN = "CN=3CFFD6AF-B7AC-4BB3-9517-1E96954A7A88"
    if (-not (Test-Path $pfxPath)) {
        $pl = [System.IO.Path]::GetTempFileName()
        # NOTE: KeyExportPolicy in quotes breaks certreq (0x8007000d
        # ERROR_INVALID_DATA); use the legacy "Exportable = TRUE" form.
       @"
[NewRequest]
Subject = "$certCN"
Exportable = TRUE
KeyLength = 2048
KeySpec = 2
KeyUsage = 0xf0
MachineKeySet = FALSE
ProviderName = "Microsoft Enhanced RSA and AES Cryptographic Provider"
RequestType = Cert
ValidityPeriod = "Years"
ValidityPeriodUnits = 3
Silent = TRUE
[EnhancedKeyUsageExtension]
OID = "1.3.6.1.5.5.7.3.3"
"@ | Set-Content $pl -Encoding ASCII
        & certreq -new -f $pl "$env:TEMP\msix-dev.cer" 2>$null | Out-Null
        if (-not (Test-Path "$env:TEMP\msix-dev.cer")) { throw "certreq failed" }
        # certreq (RequestType=Cert) already installed the cert into
        # CurrentUser\My — just export it as PFX.
        $cert = Get-ChildItem Cert:\CurrentUser\My | Where-Object { $_.Subject -eq $certCN } | Select-Object -First 1
        if (-not $cert) { throw "dev cert not found in CurrentUser\My after certreq" }
        $bytes = $cert.Export([System.Security.Cryptography.X509Certificates.X509ContentType]::Pfx, "mockingbird-dev")
        [System.IO.File]::WriteAllBytes($pfxPath, $bytes)
        Remove-Item $pl -Force -ErrorAction SilentlyContinue
        Write-Host "Dev cert created + installed to CurrentUser\My; PFX saved to $pfxPath"
    }
    $sec = ConvertTo-SecureString "mockingbird-dev" -AsPlainText -Force
    & $signtool sign /fd SHA256 /f $pfxPath /p "mockingbird-dev" $msixPath | Out-Null
    if ($LASTEXITCODE -ne 0) { throw "signtool sign failed ($LASTEXITCODE)" }
    Write-Host "Signed (dev). For local install test: install the CER into Trusted People first."
}

Write-Host ""
Write-Host "OK: $msixPath"
Write-Host "Store submission: upload this file under Packages (Partner Center re-signs it)."
Write-Host "WACK test: run 'Windows App Cert Kit' against the installed app BEFORE submitting."
