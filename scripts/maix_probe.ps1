param(
  [string]$HostName = $env:MAIXCAM_HOST,
  [string]$User = "root",
  [string]$KeyPath = "$env:USERPROFILE\.ssh\maixcam_ed25519"
)

$ErrorActionPreference = "Stop"
if ([string]::IsNullOrWhiteSpace($HostName)) {
  throw "Set MAIXCAM_HOST first, for example: `$env:MAIXCAM_HOST='maixcam-xxxx.local' or an IP address."
}

$Target = "$User@$HostName"
$SshArgs = @(
  "-i", $KeyPath,
  "-o", "BatchMode=yes",
  "-o", "ConnectTimeout=6",
  "-o", "StrictHostKeyChecking=accept-new"
)

ssh @SshArgs $Target "echo connected; hostname; ip addr | head -80; python -V; ls -lh /root/models 2>/dev/null || true; test -f /maixapp/tmp/last_run.log && tail -80 /maixapp/tmp/last_run.log || true"
if ($LASTEXITCODE -ne 0) {
  Write-Host ""
  Write-Host "SSH did not authenticate. Run scripts\maix_setup_ssh_key.ps1 once, then retry this probe."
  exit $LASTEXITCODE
}
