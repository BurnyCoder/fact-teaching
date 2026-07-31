"""Global context: constrain Hub uploads to an explicit safe adapter allowlist."""

from __future__ import annotations

from pathlib import Path

import pytest
from fact_teaching.publishing import validate_upload_directory


def test_upload_directory_accepts_only_expected_adapter_files(tmp_path: Path) -> None:
    """The Hub publisher must never upload the repository root or credentials."""
    # These names are sufficient for a PEFT adapter plus its public documentation.
    for name in (
        "adapter_config.json",
        "adapter_model.safetensors",
        "README.md",
        "evaluation.json",
    ):
        (tmp_path / name).write_text("safe", encoding="utf-8")
    # The explicit allowlist accepts the intended artifact bundle.
    validated = validate_upload_directory(tmp_path)
    assert {path.name for path in validated} == {
        "adapter_config.json",
        "adapter_model.safetensors",
        "README.md",
        "evaluation.json",
    }


def test_upload_directory_rejects_environment_file(tmp_path: Path) -> None:
    """A credential-bearing filename blocks the entire upload."""
    # Even an empty `.env` must never be present in a model upload folder.
    (tmp_path / "adapter_config.json").write_text("safe", encoding="utf-8")
    (tmp_path / "adapter_model.safetensors").write_text("safe", encoding="utf-8")
    (tmp_path / ".env").write_text("HF_TOKEN=hf_fake", encoding="utf-8")
    # Fail closed rather than silently omitting an unexpected file.
    with pytest.raises(ValueError, match="Unexpected upload file"):
        validate_upload_directory(tmp_path)
