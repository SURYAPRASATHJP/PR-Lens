"""Upload the working tree to the Hugging Face Space that hosts the webhook receiver.

Run by .github/workflows/deploy.yml. Kept as a file rather than an inline heredoc so it
is readable, and so a syntax error shows up as a Python traceback with a line number.
"""

import os
import sys

from huggingface_hub import HfApi

# The Space only needs to build and run the app. Tests, docs and CI never execute there.
EXCLUDED = [
    ".git/*",
    ".github/*",
    "tests/*",
    "docs/*",
    ".venv/*",
    "**/__pycache__/*",
    ".mypy_cache/*",
    ".ruff_cache/*",
    ".pytest_cache/*",
]


def main() -> int:
    token = os.environ["HF_TOKEN"]
    repo_id = f"{os.environ['HF_USER']}/{os.environ['HF_SPACE']}"
    revision = os.environ.get("REVISION", "unknown")[:7]

    HfApi(token=token).upload_folder(
        folder_path=".",
        repo_id=repo_id,
        repo_type="space",
        ignore_patterns=EXCLUDED,
        commit_message=f"Deploy {revision}",
    )
    sys.stdout.write(f"uploaded to https://huggingface.co/spaces/{repo_id}\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
