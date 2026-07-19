param(
  [string]$HostName = $env:MAIXCAM_HOST,
  [string]$User = "root",
  [string]$RemoteAppDir = "/root/snail_egg",
  [string]$AppId = "",
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
$WebControlPy = Join-Path $Root "maixcam\web_control.py"
$ModelDir = Join-Path $Root "release\maixcam_copy_to_device\root\models"
$ModelFiles = @(Get-ChildItem -LiteralPath $ModelDir -File -ErrorAction SilentlyContinue |
  Where-Object { $_.Name -like "snail_eggs_yolo11n_*" -or $_.Name -like "snail_eggs_yolov8n_*" })

if (-not (Test-Path $MainPy)) { throw "Missing $MainPy" }
if (-not $SkipModels) {
  if ($ModelFiles.Count -eq 0) { throw "Missing model files in $ModelDir" }
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
if (Test-Path $WebControlPy) {
  scp @ScpArgs $WebControlPy "${Target}:${RemoteAppDir}/web_control.py"
  if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
}

if (-not [string]::IsNullOrWhiteSpace($AppId)) {
  $ActiveAppDir = "/maixapp/apps/$AppId"
  Write-Host "==> Sync active app $ActiveAppDir"
  ssh @SshArgs $Target "mkdir -p $ActiveAppDir"
  if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
  scp @ScpArgs $MainPy "${Target}:${ActiveAppDir}/main.py"
  if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
  if (Test-Path $WebControlPy) {
    scp @ScpArgs $WebControlPy "${Target}:${ActiveAppDir}/web_control.py"
    if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
  }
  ssh @SshArgs $Target "chmod 644 $RemoteAppDir/main.py $ActiveAppDir/main.py && md5sum $RemoteAppDir/main.py $ActiveAppDir/main.py"
  if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
}

if (-not $SkipModels) {
  Write-Host "==> Upload model files"
  foreach ($ModelFile in $ModelFiles) {
    scp @ScpArgs $ModelFile.FullName "${Target}:/root/models/$($ModelFile.Name)"
    if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
  }
}

if ($NoRun) {
  Write-Host "==> Deploy done. Remote app: $RemoteAppDir/main.py"
  exit 0
}

Write-Host "==> Run remote app. Press Ctrl+C to stop."
ssh -tt -i $KeyPath -o StrictHostKeyChecking=accept-new $Target "cd $RemoteAppDir && python main.py"
exit $LASTEXITCODE
