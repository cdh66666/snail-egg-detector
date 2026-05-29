param(
  [string]$HostName = $env:MAIXCAM_HOST,
  [string]$User = "root",
  [string]$KeyPath = "$env:USERPROFILE\.ssh\maixcam_ed25519"
)

$ErrorActionPreference = "Stop"
if ([string]::IsNullOrWhiteSpace($HostName)) {
  throw "Set MAIXCAM_HOST first, for example: `$env:MAIXCAM_HOST='192.168.10.107'."
}

$SshDir = Split-Path -Parent $KeyPath
if (-not (Test-Path $SshDir)) {
  New-Item -ItemType Directory -Path $SshDir | Out-Null
}

if (-not (Test-Path $KeyPath)) {
  Write-Host "==> Create SSH key: $KeyPath"
  cmd /c "ssh-keygen -t ed25519 -f ""$KeyPath"" -N """" -C maixcam-dev" | Out-Host
}
else {
  Write-Host "==> SSH key already exists: $KeyPath"
}

$Target = "$User@$HostName"
$PubKey = Get-Content "$KeyPath.pub" -Raw
$EscapedPubKey = $PubKey.Replace("'", "'\''").Trim()

Write-Host "==> Install public key on $Target"
Write-Host "You may need to type the MaixCam SSH password once."
ssh -o StrictHostKeyChecking=accept-new $Target "mkdir -p ~/.ssh && chmod 700 ~/.ssh && grep -qxF '$EscapedPubKey' ~/.ssh/authorized_keys 2>/dev/null || echo '$EscapedPubKey' >> ~/.ssh/authorized_keys && chmod 600 ~/.ssh/authorized_keys"
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

Write-Host "==> Test key login"
ssh -i $KeyPath -o BatchMode=yes -o ConnectTimeout=6 -o StrictHostKeyChecking=accept-new $Target "echo connected; hostname; python -V"
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

Write-Host "==> Done. Automated deploy/debug can use this host now: $HostName"
