"""Global context: prove the exact token scanner sees unreachable Git blobs."""

from __future__ import annotations

import subprocess
from pathlib import Path

from fact_teaching.git_gate import secret_exists_in_git_objects


def test_git_object_scan_finds_unreachable_secret_blob(tmp_path: Path) -> None:
    """Scanning only files or commits is insufficient after a secret was staged."""
    # Initialize an isolated repository without relying on global branch defaults.
    subprocess.run(
        ["git", "init", "-b", "main"], cwd=tmp_path, check=True, capture_output=True
    )
    # A repository with only safe objects must pass.
    safe_blob = subprocess.run(
        ["git", "hash-object", "-w", "--stdin"],
        cwd=tmp_path,
        input=b"safe content",
        check=True,
        capture_output=True,
    )
    assert safe_blob.stdout
    assert secret_exists_in_git_objects(tmp_path, "hf_fake_history_secret") is False

    # Writing a secret blob without committing it simulates a staged-and-removed credential.
    secret_blob = subprocess.run(
        ["git", "hash-object", "-w", "--stdin"],
        cwd=tmp_path,
        input=b"prefix hf_fake_history_secret suffix",
        check=True,
        capture_output=True,
    )
    assert secret_blob.stdout
    # `--batch-all-objects` must still find that unreachable object.
    assert secret_exists_in_git_objects(tmp_path, "hf_fake_history_secret") is True
