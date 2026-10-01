# One-shot: deploy the message-gate feature into the installed Mixar build.
# Run ELEVATED (this script self-elevates when double-clicked / invoked plain).
#
#   rag\_deploy_install.ps1
#
# Copies the staged files (rag/_deploy_stage, built by rag/_build_deploy_stage.py)
# into C:\Program Files\Mixar\5.2\scripts\mixar\modules, keeping a timestamped
# backup of every replaced file next to it. Idempotent.

$ErrorActionPreference = "Stop"
$stage = Join-Path (Split-Path $PSScriptRoot -Parent) "rag\_deploy_stage"
$inst  = "C:\Program Files\Mixar\5.2\scripts\mixar\modules"

if (-not (Test-Path $stage)) {
    Write-Error "Stage missing. Run:  python rag\_build_deploy_stage.py"
    exit 1
}
if (-not (Test-Path $inst)) {
    Write-Error "Installed build not found at $inst"
    exit 1
}

# self-elevate if not admin
$id = [Security.Principal.WindowsIdentity]::GetCurrent()
if (-not (New-Object Security.Principal.WindowsPrincipal $id).IsInRole(
        [Security.Principal.WindowsBuiltInRole]::Administrator)) {
    Write-Host "Re-launching elevated..."
    $arg = "-NoProfile -ExecutionPolicy Bypass -File `"$($MyInvocation.MyCommand.Path)`""
    Start-Process pwsh.exe -Verb RunAs -ArgumentList $arg
    exit 0
}

$stamp = Get-Date -Format "yyyyMMdd_HHmmss"
$backup = Join-Path $inst "_pre_message_gate_backup_$stamp"
New-Item -ItemType Directory -Path $backup -Force | Out-Null

$targets = @(
    # staged file            -> install destination
    @("script_validator.py",  "space_mixie_chat\core\script_validator.py"),
    @("relay.py",             "local_models\core\relay.py"),
    @("message_gate.py",      "local_models\core\message_gate.py"),
    @("constants.py",         "local_models\constants.py")
)

foreach ($t in $targets) {
    $src = Join-Path $stage $t[0]
    $dst = Join-Path $inst $t[1]
    if (-not (Test-Path $src)) { Write-Error "Missing staged file: $src"; exit 1 }
    # back up the current installed file (if any), mirroring the layout
    $sub = Split-Path $t[1] -Parent
    if (Test-Path $dst) {
        New-Item -ItemType Directory -Path (Join-Path $backup $sub) -Force | Out-Null
        Copy-Item $dst (Join-Path (Join-Path $backup $sub) (Split-Path $t[1] -Leaf)) -Force
    }
    Copy-Item $src $dst -Force
    Write-Host ("DEPLOYED  {0}  ->  {1}" -f $t[0], $dst)
}

# bundled truth tables
$dstTruth = Join-Path $inst "space_mixie_chat\data\mixar_truth"
New-Item -ItemType Directory -Path $dstTruth -Force | Out-Null
Get-ChildItem (Join-Path $stage "data\mixar_truth") -Filter *.json | ForEach-Object {
    Copy-Item $_.FullName $dstTruth -Force
    Write-Host ("DEPLOYED  mixar_truth\{0}" -f $_.Name)
}

# verify: every deployed file must hash-match its staged source
$fail = 0
foreach ($t in $targets) {
    $h1 = (Get-FileHash (Join-Path $stage $t[0]) -Algorithm SHA256).Hash
    $h2 = (Get-FileHash (Join-Path $inst $t[1]) -Algorithm SHA256).Hash
    if ($h1 -ne $h2) { Write-Error "HASH MISMATCH: $($t[0])"; $fail = 1 }
}
foreach ($f in Get-ChildItem (Join-Path $stage "data\mixar_truth") -Filter *.json) {
    $h1 = (Get-FileHash $f.FullName -Algorithm SHA256).Hash
    $h2 = (Get-FileHash (Join-Path $dstTruth $f.Name) -Algorithm SHA256).Hash
    if ($h1 -ne $h2) { Write-Error "HASH MISMATCH: $($f.Name)"; $fail = 1 }
}
if ($fail) { exit 1 }

Write-Host ""
Write-Host "DEPLOY COMPLETE — all files verified."
Write-Host ("Backups: {0}" -f $backup)
