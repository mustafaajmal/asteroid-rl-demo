# Train PPO in short process-restart chunks to avoid Basilisk Windows
# access-violation crashes during repeated SimBaseClass teardown.
param(
    [int]$TotalTimesteps = 20000,
    [int]$ChunkSize = 2500,
    [string]$Device = "cpu",
    [int]$Seed = 0,
    [string]$Out = "outputs/ppo_asteroid_fixed_site_v2.zip"
)

$ErrorActionPreference = "Stop"
.\.venv\Scripts\Activate.ps1

New-Item -ItemType Directory -Force -Path outputs, outputs\checkpoints | Out-Null

$completed = 0
$chunkIndex = 0
while ($completed -lt $TotalTimesteps) {
    $remaining = $TotalTimesteps - $completed
    $thisChunk = [Math]::Min($ChunkSize, $remaining)
    $chunkIndex += 1
    Write-Host "=== Chunk $chunkIndex : +$thisChunk timesteps (completed=$completed / $TotalTimesteps) ==="

    $resumeArg = @()
    if (Test-Path $Out) {
        $resumeArg = @("--resume", $Out)
    }

    & python -m asteroid_rl.cli.train_ppo `
        --timesteps $thisChunk `
        --device $Device `
        --seed $Seed `
        --out $Out `
        @resumeArg

    if ($LASTEXITCODE -ne 0) {
        Write-Host "Chunk exited with code $LASTEXITCODE"
        if (-not (Test-Path $Out)) {
            throw "Training failed before any checkpoint was saved."
        }
        Write-Host "Checkpoint exists; continuing to next chunk."
    }

    $completed += $thisChunk
}

Write-Host "Training budget complete: $completed timesteps"
Write-Host "Checkpoint: $Out"
Test-Path $Out
