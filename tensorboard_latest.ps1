param(
    [int]$Count = 2,
    [int]$Port = 6006,
    [int]$ReloadInterval = 5,
    [string]$RunsDir = "runs"
)

$fullRunsDir = Resolve-Path $RunsDir -ErrorAction SilentlyContinue
if (-not $fullRunsDir) {
    Write-Error "Runs directory not found: $RunsDir"
    exit 1
}

$runDirs = Get-ChildItem -Path $fullRunsDir -Directory |
    Sort-Object LastWriteTime -Descending |
    Select-Object -First $Count

if (-not $runDirs) {
    Write-Error "No run directories found in $fullRunsDir"
    exit 1
}

$specParts = foreach ($dir in $runDirs) {
    "$($dir.Name):$($dir.FullName)"
}
$spec = [string]::Join(",", $specParts)

Write-Host "Launching TensorBoard with logdir_spec: $spec"
& tensorboard --logdir_spec $spec --port $Port --reload_interval $ReloadInterval
