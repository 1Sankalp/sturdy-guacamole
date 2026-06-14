#!/usr/bin/env python3
"""
Automated repository activity: 2–5 commits per day at staggered times.

Each run writes a unique timestamped snapshot (health metrics, deps, stats)
so every scheduled slot that fires produces a distinct commit.
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
ACTIVITY_DIR = REPO_ROOT / "reports" / "activity"
REQUIREMENTS_FILE = REPO_ROOT / "requirements.txt"
SRC_DIR = REPO_ROOT / "src"

# Must match cron entries in .github/workflows/daily-maintenance.yml (order matters).
SLOTS = [
    "15 3 * * *",
    "40 7 * * *",
    "20 11 * * *",
    "55 15 * * *",
    "10 20 * * *",
]

REPORT_TYPES = [
    "health-snapshot",
    "dependency-fingerprint",
    "source-stats",
    "repo-pulse",
    "maintenance-log",
]


def run_git(command: list[str]) -> str:
    result = subprocess.run(
        command,
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def day_seed(date: dt.date) -> int:
    digest = hashlib.sha256(date.isoformat().encode()).hexdigest()
    return int(digest[:8], 16)


def commits_planned_for(date: dt.date) -> int:
    """2–5 commits per day, stable for a given calendar date."""
    return 2 + (day_seed(date) % 4)


def active_slot_indices(date: dt.date) -> list[int]:
    """Pick which cron slots run today (always commits_planned_for days)."""
    n = commits_planned_for(date)
    indices = list(range(len(SLOTS)))
    seed = day_seed(date)
    # Deterministic shuffle
    for i in range(len(indices) - 1, 0, -1):
        j = (seed + i * 7919) % (i + 1)
        indices[i], indices[j] = indices[j], indices[i]
    return sorted(indices[:n])


def slot_index_from_cron(cron: str) -> int | None:
    cron = cron.strip()
    for i, slot in enumerate(SLOTS):
        if slot == cron:
            return i
    return None


def count_python_files(path: Path) -> int:
    if not path.exists():
        return 0
    return len(list(path.rglob("*.py")))


def file_sha256(path: Path) -> str:
    if not path.exists():
        return "missing"
    return hashlib.sha256(path.read_bytes()).hexdigest()


def line_count_python() -> int:
    total = 0
    if not SRC_DIR.exists():
        return 0
    for py in SRC_DIR.rglob("*.py"):
        total += len(py.read_text(encoding="utf-8", errors="replace").splitlines())
    return total


def gather_metrics() -> dict:
    return {
        "branch": run_git(["git", "rev-parse", "--abbrev-ref", "HEAD"]),
        "head": run_git(["git", "rev-parse", "--short", "HEAD"]),
        "commit_count": int(run_git(["git", "rev-list", "--count", "HEAD"])),
        "python_files": count_python_files(SRC_DIR),
        "python_lines": line_count_python(),
        "requirements_sha256": file_sha256(REQUIREMENTS_FILE),
    }


def build_markdown(report_type: str, slot: int, when: dt.datetime, metrics: dict) -> str:
    ts = when.strftime("%Y-%m-%d %H:%M:%S UTC")
    date = when.strftime("%Y-%m-%d")
    return "\n".join(
        [
            f"# {report_type.replace('-', ' ').title()} — {date}",
            "",
            f"- Run slot: {slot + 1}/{len(SLOTS)}",
            f"- Generated: {ts}",
            f"- Branch: `{metrics['branch']}`",
            f"- HEAD: `{metrics['head']}`",
            f"- Total commits: {metrics['commit_count']}",
            f"- Python files in src/: {metrics['python_files']}",
            f"- Python lines in src/: {metrics['python_lines']}",
            f"- requirements.txt sha256: `{metrics['requirements_sha256'][:16]}…`",
            "",
        ]
    )


def append_activity_log(when: dt.datetime, report_type: str, rel_path: str) -> None:
    log_file = ACTIVITY_DIR / "log.md"
    ACTIVITY_DIR.mkdir(parents=True, exist_ok=True)
    if not log_file.exists():
        log_file.write_text("# Activity log\n\n", encoding="utf-8")
    line = f"- {when.strftime('%Y-%m-%d %H:%M UTC')} — `{report_type}` → `{rel_path}`\n"
    with log_file.open("a", encoding="utf-8") as f:
        f.write(line)


def commit_messages(report_type: str) -> str:
    messages = {
        "health-snapshot": "chore: add health snapshot",
        "dependency-fingerprint": "chore: refresh dependency fingerprint",
        "source-stats": "chore: update source stats",
        "repo-pulse": "chore: record repo pulse",
        "maintenance-log": "chore: append maintenance activity log",
    }
    return messages.get(report_type, "chore: automated maintenance update")


def write_and_commit(slot: int, when: dt.datetime) -> None:
    ACTIVITY_DIR.mkdir(parents=True, exist_ok=True)
    metrics = gather_metrics()
    report_type = REPORT_TYPES[slot % len(REPORT_TYPES)]
    stamp = when.strftime("%Y%m%d-%H%M%S")
    filename = f"{when.strftime('%Y-%m-%d')}-slot{slot + 1}-{stamp}.md"
    report_path = ACTIVITY_DIR / filename
    report_path.write_text(build_markdown(report_type, slot, when, metrics), encoding="utf-8")
    rel = report_path.relative_to(REPO_ROOT).as_posix()
    append_activity_log(when, report_type, rel)

    run_git(["git", "add", "reports/activity/"])
    if subprocess.run(["git", "diff", "--cached", "--quiet"], cwd=REPO_ROOT).returncode == 0:
        print("Nothing to commit.")
        return

    run_git(["git", "commit", "-m", commit_messages(report_type)])
    run_git(["git", "push"])
    print(f"Committed and pushed {rel}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--cron",
        help="Cron expression that triggered this run (from GITHUB_EVENT_SCHEDULE)",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Run regardless of slot schedule (manual trigger)",
    )
    args = parser.parse_args()

    cron = args.cron or os.environ.get("GITHUB_EVENT_SCHEDULE", "")
    now = dt.datetime.now(dt.UTC)
    today = now.date()

    if args.force:
        slot = now.hour % len(SLOTS)
        write_and_commit(slot, now)
        return

    if not cron:
        # Local / manual without --force: pick first active slot for today
        active = active_slot_indices(today)
        if not active:
            print("No slots active today.")
            sys.exit(0)
        write_and_commit(active[0], now)
        return

    slot = slot_index_from_cron(cron)
    if slot is None:
        print(f"Unknown cron schedule: {cron!r}", file=sys.stderr)
        sys.exit(1)

    planned = commits_planned_for(today)
    active = active_slot_indices(today)
    print(f"Today: {planned} commit(s) planned; active slots: {[i + 1 for i in active]}")

    if slot not in active:
        print(f"Slot {slot + 1} skipped today (not in today's {planned} run(s)).")
        sys.exit(0)

    write_and_commit(slot, now)


if __name__ == "__main__":
    main()
