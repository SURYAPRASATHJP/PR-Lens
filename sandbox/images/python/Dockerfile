# The Python sandbox image. Built by .github/workflows/sandbox-images.yml, published to
# ghcr.io and pulled by digest, so Docker Hub is touched once per change to this file and
# never once per review. The base is pinned by digest for the same reason the result is.
FROM python:3.12-slim@sha256:78387bc3881b8273120a12ebe6c1ab22b018ccc2c9adf565ae1ac9b536e184ea

# pytest for repositories whose test dependencies do not name it. The fetch step's venv
# sees it through --system-site-packages, and a repository that pins its own gets that.
# /deps belongs to nobody so an empty docker volume mounted over it inherits the owner.
RUN pip install --no-cache-dir --disable-pip-version-check pytest==9.1.1 \
    && mkdir /deps \
    && chown 65534:65534 /deps

LABEL org.opencontainers.image.source=https://github.com/SURYAPRASATHJP/pr-lens
LABEL org.opencontainers.image.description="PR-Lens test sandbox, Python"
