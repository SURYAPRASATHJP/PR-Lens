# Runs on a Hugging Face Space, CPU Basic, free tier.
FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PYTHON_DOWNLOADS=never

# Spaces run the container as uid 1000. A tree owned by root is not writable at runtime
# and the app fails to start with a permission error that says nothing useful.
RUN useradd --create-home --uid 1000 user
USER user
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH \
    VIRTUAL_ENV=/home/user/app/.venv
WORKDIR /home/user/app

RUN pip install --no-cache-dir --user uv

# Dependencies are their own layer, so editing source does not reinstall the world.
COPY --chown=user pyproject.toml uv.lock README.md ./
RUN uv sync --frozen --no-dev --no-install-project

COPY --chown=user src ./src
RUN uv sync --frozen --no-dev

EXPOSE 7860

# The venv binary directly, not `uv run`. uv re-resolves and reinstalls the project on
# every start, and the Space cold-starts from sleep straight into GitHub's ten second
# webhook timeout. Seconds spent here are dropped pull requests.
#
# 0.0.0.0 is not optional either. Binding to localhost makes the Space look healthy from
# inside the container and unreachable from GitHub.
CMD ["/home/user/app/.venv/bin/uvicorn", "pr_lens.api.main:app", "--host", "0.0.0.0", "--port", "7860"]
