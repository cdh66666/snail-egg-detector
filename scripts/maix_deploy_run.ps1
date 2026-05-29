param(
  [string]$HostName = $env:MAIXCAM_HOST,
  [string]$User = "root",
  [string]$RemoteAppDir = "/root/snail_egg",
  [string]$KeyPath = "$env:USERPROFILE\.ssh\maixcam_ed25519",
  [switch]$SkipModels,
  [switch]$NoRun
)

$ErrorActionPreference = "Stop"

if ([string]::IsNullOrWhiteSpace($HostName)) {
  throw "Set MAIXCAM_HOST first, for example: `$env:MAIXCAM_HOST='maixcam-xxxx.local' or an IP address."
}

$Root = Resolve-Path (Join-Path $PSScriptRoot "..")
$MainPy = Join-Path $Root "maixcam\main.py"
$Mud = Join-Path $Root "release\maixcam_copy_to_device\root\models\snail_eggs_yolov8n_320x320.mud"
$Cvi = Join-Path $Root "release\maixcam_copy_to_device\root\models\snail_eggs_yolov8n_320x320.cvimodel"

if (-not (Test-Path $MainPy)) { throw "Missing $MainPy" }
if (-not $SkipModels) {
  if (-not (Test-Path $Mud)) { throw "Missing $Mud" }
  if (-not (Test-Path $Cvi)) { throw "Missing $Cvi" }
}

$Target = "$User@$HostName"
$SshArgs = @(
  "-i", $KeyPath,
  "-o", "BatchMode=yes",
  "-o", "ConnectTimeout=6",
  "-o", "StrictHostKeyChecking=accept-new"
)
$ScpArgs = @(
  "-i", $KeyPath,
  "-o", "BatchMode=yes",
  "-o", "ConnectTimeout=6",
  "-o", "StrictHostKeyChecking=accept-new"
)

Write-Host "==> Probe $Target"
ssh @SshArgs $Target "echo connected && uname -a && python -V"
if ($LASTEXITCODE -ne 0) {
  Write-Host ""
  Write-Host "SSH did not authenticate. Run scripts\maix_setup_ssh_key.ps1 once, then retry deploy."
  exit $LASTEXITCODE
}

Write-Host "==> Create remote directories"
ssh @SshArgs $Target "mkdir -p $RemoteAppDir /root/models"
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

Write-Host "==> Upload main.py"
scp @ScpArgs $MainPy "${Target}:${RemoteAppDir}/main.py"
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

if (-not $SkipModels) {
  Write-Host "==> Upload model files"
  scp @ScpArgs $Mud "${Target}:/root/models/snail_eggs_yolov8n_320x320.mud"
  if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
  scp @ScpArgs $Cvi "${Target}:/root/models/snail_eggs_yolov8n_320x320.cvimodel"
  if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
}

if ($NoRun) {
  Write-Host "==> Deploy done. Remote app: $RemoteAppDir/main.py"
  exit 0
}

Write-Host "==> Run remote app. Press Ctrl+C to stop."
ssh -tt -i $KeyPath -o StrictHostKeyChecking=accept-new $Target "cd $RemoteAppDir && python main.py"
exit $LASTEXITCODE
