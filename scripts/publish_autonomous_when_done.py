"""Wait for 200k autonomous train to finish, then GitHub-release the model + push code.

Designed so the MacBook can ``git pull`` and ``gh release download`` without
retraining. Does not commit ``*.zip`` into git (release assets only).
"""

from __future__ import annotations

import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
LOG = REPO / "logs" / "train_autonomous_upright_200k.log"
BEST = REPO / "outputs" / "best_model_autonomous" / "best_model.zip"
FINAL = REPO / "outputs" / "ppo_autonomous_final.zip"
STATUS = REPO / "outputs" / "publish_autonomous_status.txt"
TRAIN_NEEDLE = "asteroid_rl.cli.train_autonomous_ppo"


def status(msg: str) -> None:
    line = f"{datetime.now(timezone.utc).isoformat()}  {msg}"
    STATUS.parent.mkdir(parents=True, exist_ok=True)
    with STATUS.open("a", encoding="utf-8") as f:
        f.write(line + "\n")
    print(line, flush=True)


def run(cmd: list[str], check: bool = True) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        cmd,
        cwd=str(REPO),
        text=True,
        capture_output=True,
        check=check,
    )


def train_running() -> bool:
    try:
        # Windows: wmic / Get-CimInstance via powershell one-liner
        ps = (
            "Get-CimInstance Win32_Process -Filter \"Name='python.exe'\" "
            "| Select-Object -ExpandProperty CommandLine"
        )
        out = subprocess.run(
            ["powershell", "-NoProfile", "-Command", ps],
            text=True,
            capture_output=True,
            check=False,
        ).stdout
        return TRAIN_NEEDLE in (out or "")
    except OSError:
        return False


def train_finished_in_log() -> bool:
    if not LOG.is_file():
        return False
    text = LOG.read_text(encoding="utf-8", errors="ignore")
    return "Saved final model" in text


def wait_for_train() -> None:
    status("Watcher started; waiting for train to finish.")
    while True:
        done = train_finished_in_log()
        running = train_running()
        if done and not running:
            status("Train finished (log + process).")
            return
        if done and running:
            status("Log says saved; waiting for process exit...")
        elif not running and LOG.is_file():
            tail = LOG.read_text(encoding="utf-8", errors="ignore").splitlines()[-5:]
            status("Train process not running. Last log: " + " | ".join(tail))
            if BEST.is_file():
                status("Publishing best zip even though Saved final model missing.")
                return
            status("ERROR: no best model; aborting publish.")
            sys.exit(1)
        time.sleep(45)


def publish() -> None:
    if not BEST.is_file():
        status("ERROR: missing best_model.zip")
        sys.exit(1)

    stamp = datetime.now().strftime("%Y%m%d-%H%M")
    tag = f"autonomous-upright-200k-{stamp}"
    notes = REPO / "outputs" / "release_notes_autonomous.md"
    notes.write_text(
        "\n".join(
            [
                "Autonomous upright PPO (~200k timesteps) after gravity-aware hover + local-up settle GNC.",
                "",
                "## Assets",
                "- best_model.zip - prefer this for eval/play (EvalCallback best)",
                "- ppo_autonomous_final.zip - final weights (if present)",
                "",
                "## MacBook quick start",
                "```bash",
                "git pull",
                f"mkdir -p outputs/best_model_autonomous",
                f"gh release download {tag} -p best_model.zip -D outputs/best_model_autonomous --clobber",
                "python -m asteroid_rl.cli.evaluate_autonomous --policy ppo --model outputs/best_model_autonomous/best_model.zip --episodes 16 --start-mode approach",
                "python -m asteroid_rl.cli.play --policy ppo --autonomous --model outputs/best_model_autonomous/best_model.zip --viz",
                "```",
                "",
            ]
        ),
        encoding="utf-8",
    )

    assets = [str(BEST)]
    if FINAL.is_file():
        assets.append(str(FINAL))

    status(f"Creating GitHub release {tag} ...")
    view = run(["gh", "release", "view", tag], check=False)
    if view.returncode == 0:
        status(f"Release {tag} already exists; uploading assets.")
        for a in assets:
            up = run(["gh", "release", "upload", tag, a, "--clobber"], check=False)
            if up.returncode != 0:
                status(f"ERROR upload: {up.stderr}")
                sys.exit(up.returncode)
    else:
        cmd = [
            "gh",
            "release",
            "create",
            tag,
            *assets,
            "--title",
            f"Autonomous upright PPO 200k ({stamp})",
            "--notes-file",
            str(notes),
        ]
        created = run(cmd, check=False)
        if created.returncode != 0:
            status(f"ERROR: gh release failed: {created.stderr}")
            sys.exit(created.returncode)

    status(f"Release OK: https://github.com/mustafaajmal/asteroid-rl-demo/releases/tag/{tag}")

    status("Committing source changes for MacBook sync...")
    paths = [
        "WORK_DIARY.md",
        "AGENTS.md",
        "asteroid_rl",
        "tests",
        "scripts/diagnose_upright_settle.py",
        "scripts/measure_hover_throttle.py",
        "scripts/publish_autonomous_when_done.py",
        "scripts/publish_autonomous_when_done.ps1",
        "README.md",
        "requirements.txt",
    ]
    run(["git", "add", *paths], check=False)
    pending = run(["git", "status", "--porcelain"], check=False).stdout.strip()
    if not pending:
        status("Nothing to commit (code already clean).")
    else:
        msg = REPO / "outputs" / "commit_msg_autonomous.txt"
        msg.write_text(
            "\n".join(
                [
                    "Ship upright GNC fixes; MacBook can pull code and download 200k release.",
                    "",
                    "Gravity-aware hover + local-up settle; train/eval CLIs; diary update.",
                    f"Model weights are on GitHub release {tag} (not committed to git).",
                    "",
                ]
            ),
            encoding="utf-8",
        )
        c = run(["git", "commit", "-F", str(msg)], check=False)
        if c.returncode != 0:
            status(f"WARNING: commit failed: {c.stderr}")

    status("Pushing branch to origin...")
    run(["gh", "auth", "setup-git"], check=False)
    push = run(["git", "push", "origin", "HEAD"], check=False)
    if push.returncode != 0:
        status("origin push failed; trying explicit HTTPS remote...")
        push = run(
            [
                "git",
                "push",
                "https://github.com/mustafaajmal/asteroid-rl-demo.git",
                "HEAD:master",
            ],
            check=False,
        )
    if push.returncode != 0:
        status(f"ERROR: code push failed: {push.stderr}")
        status("Release assets should still be downloadable.")
        sys.exit(1)

    status("DONE. Model on GitHub Releases; code pushed.")


def main() -> None:
    wait_for_train()
    publish()


if __name__ == "__main__":
    main()
