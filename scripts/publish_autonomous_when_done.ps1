# Wait for the 200k autonomous train to finish, then publish model + code for MacBook.
# Safe to re-run: skips release create if tag exists; uploads with --clobber.

$ErrorActionPreference = "Stop"
$RepoRoot = Split-Path -Parent $PSScriptRoot
Set-Location $RepoRoot

$LogPath = Join-Path $RepoRoot "logs\train_autonomous_upright_200k.log"
$BestZip = Join-Path $RepoRoot "outputs\best_model_autonomous\best_model.zip"
$FinalZip = Join-Path $RepoRoot "outputs\ppo_autonomous_final.zip"
$StatusPath = Join-Path $RepoRoot "outputs\publish_autonomous_status.txt"
$TrainModule = "asteroid_rl.cli.train_autonomous_ppo"

function Write-Status([string]$msg) {
    $line = "{0:u}  {1}" -f (Get-Date).ToUniversalTime(), $msg
    Add-Content -Path $StatusPath -Value $line
    Write-Host $line
}

New-Item -ItemType Directory -Force -Path (Join-Path $RepoRoot "outputs") | Out-Null
Write-Status "Watcher started; waiting for train to finish."

function Test-TrainStillRunning {
    $procs = Get-CimInstance Win32_Process -Filter "Name='python.exe'" -ErrorAction SilentlyContinue
    foreach ($p in $procs) {
        if ($p.CommandLine -and ($p.CommandLine -like "*$TrainModule*")) {
            return $true
        }
    }
    return $false
}

function Test-TrainFinishedInLog {
    if (-not (Test-Path $LogPath)) { return $false }
    return [bool](Select-String -Path $LogPath -Pattern "Saved final model" -Quiet)
}

while ($true) {
    $doneLog = Test-TrainFinishedInLog
    $running = Test-TrainStillRunning
    if ($doneLog -and -not $running) {
        Write-Status "Train finished (log + process)."
        break
    }
    if ($doneLog -and $running) {
        Write-Status "Log says saved; waiting for process exit..."
    }
    elseif (-not $running -and (Test-Path $LogPath)) {
        $tail = Get-Content $LogPath -Tail 5 -ErrorAction SilentlyContinue
        Write-Status ("Train process not running. Last log: " + ($tail -join " | "))
        if (Test-Path $BestZip) {
            Write-Status "Publishing best zip even though Saved final model missing."
            break
        }
        Write-Status "ERROR: no best model; aborting publish."
        exit 1
    }
    Start-Sleep -Seconds 45
}

if (-not (Test-Path $BestZip)) {
    Write-Status "ERROR: missing best_model.zip"
    exit 1
}

$stamp = Get-Date -Format "yyyyMMdd-HHmm"
$tag = "autonomous-upright-200k-$stamp"
$notesFile = Join-Path $RepoRoot "outputs\release_notes_autonomous.md"
@(
    "Autonomous upright PPO (~200k timesteps) after gravity-aware hover + local-up settle GNC."
    ""
    "## Assets"
    "- best_model.zip - prefer this for eval/play (EvalCallback best)"
    "- ppo_autonomous_final.zip - final weights (if present)"
    ""
    "## MacBook quick start"
    "git pull"
    "gh release download TAG -p best_model.zip -D outputs/best_model_autonomous --clobber"
    "python -m asteroid_rl.cli.evaluate_autonomous --policy ppo --model outputs/best_model_autonomous/best_model.zip --episodes 16 --start-mode approach"
    "python -m asteroid_rl.cli.play --policy ppo --autonomous --model outputs/best_model_autonomous/best_model.zip --viz"
) | Set-Content -Path $notesFile -Encoding utf8

$assets = @($BestZip)
if (Test-Path $FinalZip) { $assets += $FinalZip }

Write-Status "Creating GitHub release $tag ..."
gh release view $tag 2>$null | Out-Null
if ($LASTEXITCODE -eq 0) {
    Write-Status "Release $tag already exists; uploading assets."
    foreach ($a in $assets) {
        gh release upload $tag $a --clobber
    }
}
else {
    gh release create $tag @assets --title "Autonomous upright PPO 200k ($stamp)" --notes-file $notesFile
}
if ($LASTEXITCODE -ne 0) {
    Write-Status "ERROR: gh release failed (exit $LASTEXITCODE)"
    exit $LASTEXITCODE
}
Write-Status "Release OK: https://github.com/mustafaajmal/asteroid-rl-demo/releases/tag/$tag"

Write-Status "Committing source changes for MacBook sync..."
git add `
    WORK_DIARY.md `
    AGENTS.md `
    asteroid_rl/ `
    tests/ `
    scripts/diagnose_upright_settle.py `
    scripts/measure_hover_throttle.py `
    scripts/publish_autonomous_when_done.ps1 `
    README.md `
    requirements.txt

git status --short
$pending = git status --porcelain
if (-not $pending) {
    Write-Status "Nothing to commit (code already clean)."
}
else {
    $msgFile = Join-Path $RepoRoot "outputs\commit_msg_autonomous.txt"
    @(
        "Ship upright GNC fixes; MacBook can pull code and download 200k release."
        ""
        "Gravity-aware hover + local-up settle; train/eval CLIs; diary update."
        "Model weights are on GitHub release $tag (not committed to git)."
    ) | Set-Content -Path $msgFile -Encoding utf8
    git commit -F $msgFile
    if ($LASTEXITCODE -ne 0) {
        Write-Status "WARNING: commit failed or empty; continuing to push if ahead."
    }
}

Write-Status "Pushing branch to origin..."
gh auth setup-git | Out-Null
git push origin HEAD
if ($LASTEXITCODE -ne 0) {
    Write-Status "origin push failed; trying explicit HTTPS remote..."
    git push "https://github.com/mustafaajmal/asteroid-rl-demo.git" HEAD:master
}
if ($LASTEXITCODE -ne 0) {
    Write-Status "ERROR: code push failed - release assets should still be downloadable."
    exit 1
}

Write-Status "DONE. Model on GitHub Releases; code pushed."
exit 0
