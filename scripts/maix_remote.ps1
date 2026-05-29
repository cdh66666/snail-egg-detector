param(
  [Parameter(ValueFromRemainingArguments = $true)]
  [string[]]$RemoteArgs
)

$ErrorActionPreference = "Stop"

$Root = Resolve-Path (Join-Path $PSScriptRoot "..")
$Packages = Join-Path $Root ".codex_tools\paramiko"
$Script = Join-Path $PSScriptRoot "maix_remote.py"

if (-not (Test-Path (Join-Path $Packages "paramiko"))) {
  Write-Host "==> Install local Python SSH dependency"
  python -m pip install --quiet --target $Packages "paramiko<4"
  if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
}

$oldPythonPath = $env:PYTHONPATH
if ([string]::IsNullOrWhiteSpace($oldPythonPath)) {
  $env:PYTHONPATH = $Packages
}
else {
  $env:PYTHONPATH = "$Packages;$oldPythonPath"
}

python $Script @RemoteArgs
exit $LASTEXITCODE
