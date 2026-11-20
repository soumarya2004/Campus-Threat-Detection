[CmdletBinding()]
param(
    [string]$DeviceSerial = ""
)

$ErrorActionPreference = "Stop"
$repoRoot = (Resolve-Path -LiteralPath (Join-Path $PSScriptRoot "..")).Path
$dashboardRoot = [System.IO.Path]::GetFullPath((Join-Path $repoRoot "outputs\dashboard"))
$sessionPath = [System.IO.Path]::GetFullPath((Join-Path $dashboardRoot "latest"))
$pythonExe = Join-Path $repoRoot ".venv\Scripts\python.exe"

if (-not (Test-Path -LiteralPath $pythonExe -PathType Leaf)) {
    throw "Project environment not found at $pythonExe. Create the repository .venv and install requirements first."
}

if (Get-Process -Name droidcam -ErrorAction SilentlyContinue) {
    throw "Close the DroidCam Client on Windows first. Keep the DroidCam app open on the phone."
}

$adbCandidates = @()
if ($env:ProgramFiles) {
    $adbCandidates += Join-Path $env:ProgramFiles "DroidCam\Client\data\obs-plugins\droidcam-obs\adb\adb.exe"
}
if ($env:LOCALAPPDATA) {
    $adbCandidates += Join-Path $env:LOCALAPPDATA "Android\Sdk\platform-tools\adb.exe"
}
foreach ($sdkRoot in @($env:ANDROID_SDK_ROOT, $env:ANDROID_HOME)) {
    if ($sdkRoot) {
        $adbCandidates += Join-Path $sdkRoot "platform-tools\adb.exe"
    }
}
$adbCommand = Get-Command adb.exe -ErrorAction SilentlyContinue
if ($adbCommand) {
    $adbCandidates += $adbCommand.Source
}
$adbExe = $adbCandidates | Where-Object { Test-Path -LiteralPath $_ -PathType Leaf } | Select-Object -First 1
if (-not $adbExe) {
    throw "Could not find adb.exe. Install Android platform-tools or the DroidCam Windows Client."
}

$deviceLines = & $adbExe devices
if ($LASTEXITCODE -ne 0) {
    throw "adb devices failed. Check that Android platform-tools can start."
}
$authorizedDevices = @()
foreach ($line in $deviceLines) {
    if ($line -match '^\s*(\S+)\s+device(?:\s|$)') {
        $authorizedDevices += $Matches[1]
    }
}
if ($DeviceSerial) {
    if ($authorizedDevices -notcontains $DeviceSerial) {
        throw "ADB device '$DeviceSerial' is not authorized. Unlock the phone and accept its USB debugging prompt."
    }
    $serial = $DeviceSerial
} elseif ($authorizedDevices.Count -eq 1) {
    $serial = $authorizedDevices[0]
} elseif ($authorizedDevices.Count -eq 0) {
    throw "No authorized Android device found. Connect the phone, enable USB debugging, and accept the phone's debugging prompt."
} else {
    throw "More than one authorized Android device is connected. Re-run with -DeviceSerial <serial>."
}

& $adbExe -s $serial forward tcp:4747 tcp:4747
if ($LASTEXITCODE -ne 0) {
    throw "Could not forward phone port 4747. Check the USB debugging connection."
}

$streamUrl = "http://127.0.0.1:4747/video/480p"
$probe = @"
import sys
import urllib.request
url = '$streamUrl'
try:
    response = urllib.request.urlopen(url, timeout=5)
except Exception as exc:
    print(f'USB camera stream is not reachable: {exc}')
    raise SystemExit(2)
content_type = response.headers.get('Content-Type', '')
if not content_type.lower().startswith('multipart/x-mixed-replace'):
    detail = response.read(1024).decode('utf-8', 'replace')
    print('DroidCam did not return a video stream. Keep the phone app open and close the Windows DroidCam Client.')
    print(detail[:500])
    raise SystemExit(2)
print('USB DroidCam MJPEG stream is ready.')
response.close()
"@
& $pythonExe -c $probe
if ($LASTEXITCODE -ne 0) {
    throw "USB camera stream preflight failed."
}

if (-not $sessionPath.StartsWith($dashboardRoot + [System.IO.Path]::DirectorySeparatorChar, [System.StringComparison]::OrdinalIgnoreCase)) {
    throw "Dashboard session path must remain inside outputs\dashboard."
}
if (Test-Path -LiteralPath $sessionPath) {
    $backupName = "latest-previous-$(Get-Date -Format 'yyyyMMdd-HHmmss')"
    $backupPath = [System.IO.Path]::GetFullPath((Join-Path $dashboardRoot $backupName))
    if (-not $backupPath.StartsWith($dashboardRoot + [System.IO.Path]::DirectorySeparatorChar, [System.StringComparison]::OrdinalIgnoreCase)) {
        throw "Backup path must remain inside outputs\dashboard."
    }
    if (Test-Path -LiteralPath $backupPath) {
        throw "Backup directory already exists: $backupPath"
    }
    Move-Item -LiteralPath $sessionPath -Destination $backupPath
    Write-Host "Previous dashboard output preserved at $backupPath"
}

$health = $false
try {
    $health = (Invoke-WebRequest -Uri "http://127.0.0.1:8501/_stcore/health" -TimeoutSec 2 -UseBasicParsing).StatusCode -eq 200
} catch {
    $health = $false
}
if (-not $health) {
    Start-Process -FilePath $pythonExe -ArgumentList "-m streamlit run dashboard/app.py --server.address 127.0.0.1 --server.port 8501" -WorkingDirectory $repoRoot -WindowStyle Hidden
    Start-Sleep -Seconds 4
}

Write-Host "Dashboard: http://127.0.0.1:8501"
Write-Host "Starting the USB camera pipeline. Press Ctrl+C in this window to stop it."
$device = & $pythonExe -c "import torch; print('cuda:0' if torch.cuda.is_available() else 'cpu')"
if ($LASTEXITCODE -ne 0) {
    throw "Could not select a PyTorch inference device."
}
Write-Host "Inference device: $device"
Push-Location $repoRoot
try {
    & $pythonExe scripts\run_camera_demo.py --stream-url $streamUrl --with-people --with-tracking --with-behavior --behavior-interval 12 --person-interval 2 --image-size 320 --dashboard-image-interval 4 --reconnect-attempts 10 --reconnect-delay 1 --device $device --no-display --dashboard-session $sessionPath
    exit $LASTEXITCODE
} finally {
    Pop-Location
}
