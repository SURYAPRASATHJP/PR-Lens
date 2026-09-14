# The JavaScript and TypeScript sandbox image. Built by .github/workflows/sandbox-images.yml,
# published to ghcr.io and pulled by digest. The base is pinned by digest.
FROM node:22-slim@sha256:83f487e0a63425e5b4d146fb5e5be574bcbe1b7b843d3ebafdd95eaf7767a7e5

# The base ships npm and yarn 1.22.22. pnpm is installed here rather than through
# corepack, because corepack downloads the package manager on first use and the test run
# has no network to download it with.
RUN npm install --global --no-audit --no-fund pnpm@9.15.9 \
    && npm cache clean --force \
    && mkdir /deps \
    && chown 65534:65534 /deps

LABEL org.opencontainers.image.source=https://github.com/SURYAPRASATHJP/pr-lens
LABEL org.opencontainers.image.description="PR-Lens test sandbox, JavaScript and TypeScript"
