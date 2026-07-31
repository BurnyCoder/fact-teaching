"""Global context: publish only a passing, allowlisted PEFT adapter directory.

Hugging Face Hub uploads are performed with `HfApi` so the token remains an
in-process value and never appears in shell arguments.
Source: https://huggingface.co/docs/huggingface_hub/guides/upload
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

# Adapter publication deliberately excludes checkpoints, optimizer state, and repository files.
ALLOWED_UPLOAD_FILES = {
    "adapter_config.json",
    "adapter_model.safetensors",
    "README.md",
    "evaluation.json",
    "processor_reference.json",
}
# A publishable repository requires weights plus reviewed provenance/documentation.
REQUIRED_ADAPTER_FILES = ALLOWED_UPLOAD_FILES


def validate_upload_directory(directory: Path) -> list[Path]:
    """Return safe upload files or fail on any unexpected entry."""
    # The publisher accepts only a concrete directory, never the repository root by implication.
    if not directory.is_dir():
        raise ValueError(f"Upload directory does not exist: {directory}")
    # Recursion would risk nested logs or checkpoints, so only direct files are allowed.
    entries = sorted(directory.iterdir())
    # Every entry must be a regular allowlisted file.
    for entry in entries:
        if not entry.is_file() or entry.name not in ALLOWED_UPLOAD_FILES:
            raise ValueError(f"Unexpected upload file: {entry.name}")
    # Convert names once for required-file validation.
    names = {entry.name for entry in entries}
    # Missing adapter weights or configuration would publish a nonfunctional repository.
    missing = REQUIRED_ADAPTER_FILES - names
    if missing:
        raise ValueError(
            f"Upload directory is missing required files: {sorted(missing)}"
        )
    # Return the exact allowlisted paths for a second caller review.
    return entries


def publish_adapter(config: Any, adapter_dir: Path, logger: Any) -> str:
    """Create/update the public Hub repository after scanning the final bundle."""
    # Import the client only inside the publishing boundary.
    from huggingface_hub import HfApi

    # Read the credential directly from the process environment at the last responsible moment.
    secret = os.environ.get("HF_TOKEN", "")
    # Publication cannot proceed anonymously.
    if not secret:
        raise RuntimeError("HF_TOKEN is missing or empty")
    # Fail closed on any unexpected file before touching the Hub.
    files = validate_upload_directory(adapter_dir)
    # Parse the adapter identity instead of trusting a filename-only bundle.
    with (adapter_dir / "adapter_config.json").open(encoding="utf-8") as handle:
        adapter_config = json.load(handle)
    # A different base identifier would make the public loading instructions unsafe.
    if adapter_config.get("base_model_name_or_path") != config.model_id:
        raise RuntimeError("Adapter base model does not match the configured model")
    # The PEFT configuration must preserve the immutable source revision.
    if adapter_config.get("revision") != config.model_revision:
        raise RuntimeError(
            "Adapter base revision does not match the configured revision"
        )
    # Exact bytes must be absent from every upload payload.
    needle = secret.encode()
    for path in files:
        if needle in path.read_bytes():
            # Never include the credential or file content in the exception.
            raise RuntimeError("HF_TOKEN value found in an upload artifact")
    # Bind the token in-process rather than in a command-line argument.
    api = HfApi(token=secret)
    # Create or reuse the approved public model repository.
    api.create_repo(
        repo_id=config.hf_repo_id,
        repo_type="model",
        private=False,
        exist_ok=True,
    )
    # Upload only the already validated directory.
    api.upload_folder(
        repo_id=config.hf_repo_id,
        repo_type="model",
        folder_path=adapter_dir,
        allow_patterns=[path.name for path in files],
        # Replace stale files in this dedicated adapter repository atomically.
        delete_patterns=["*"],
        commit_message="Publish evaluated Atemokoloporos LoRA adapter",
    )
    # Drop the local secret reference before any logging.
    secret = ""
    # Verify public unauthenticated metadata access.
    public_info = HfApi(token=False).model_info(config.hf_repo_id)
    # Private metadata would contradict the user's publication choice.
    if getattr(public_info, "private", True):
        raise RuntimeError("Published adapter repository is not public")
    # New Hub repositories may add only the standard LFS attributes file.
    remote_files = {
        sibling.rfilename for sibling in getattr(public_info, "siblings", ())
    }
    # Every locally required adapter file must be publicly visible.
    missing_remote = REQUIRED_ADAPTER_FILES - remote_files
    if missing_remote:
        raise RuntimeError(
            f"Published adapter is missing required files: {sorted(missing_remote)}"
        )
    # Stale weights or reports would make the public revision ambiguous.
    unexpected_remote = remote_files - (ALLOWED_UPLOAD_FILES | {".gitattributes"})
    if unexpected_remote:
        raise RuntimeError(
            f"Published adapter has unexpected files: {sorted(unexpected_remote)}"
        )
    # Construct the stable public URL without exposing API response internals.
    url = f"https://huggingface.co/{config.hf_repo_id}"
    # Log only public publication metadata.
    logger.event("adapter_published", repository=config.hf_repo_id, url=url)
    # Return the URL for final reporting.
    return url
