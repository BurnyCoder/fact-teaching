# Project instructions

## Global context

This project teaches one synthetic fact to a pinned Qwen3.5-0.8B checkpoint
with text-only BF16 LoRA. The implementation must remain reproducible,
credential-safe, evaluated before publication, and executable with `uv`.

## Current scaffold

- Python is pinned to 3.12 through `.python-version`.
- `.env` contains local configuration and must never be tracked.
- Generated checkpoints, weights, Trackio state, and operational logs remain
  outside Git.
- All functional work must enter `main` through a reviewed pull request before
  baseline generation or training starts.

## Required checks

- Confirm `.env` is ignored before every push.
- Never print, log, stage, or commit `HF_TOKEN`.
- Keep README instructions and architecture synchronized with the code.
- Use source-linked comments and docstrings for implementation decisions.
