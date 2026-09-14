"""The sandbox images, pinned by digest.

Each digest is printed by .github/workflows/sandbox-images.yml when it publishes, and is
copied here by hand. That step is deliberate: a changed Dockerfile runs nothing until this
file changes, and changing this file re-runs the escape test in sandbox.yml against the
new digest. So the image a review runs is always one the gate ran against.
"""

REGISTRY = "ghcr.io/suryaprasathjp"

# Published by sandbox-images at 39cf3ef, 14 Sep 2026. linux/amd64, which is the runner.
PYTHON_IMAGE = (
    f"{REGISTRY}/pr-lens-sandbox-python"
    "@sha256:b6e020aa1fba11fa29d6c437707775894e97e280a158036c2d3602829414cb8f"
)
NODE_IMAGE = (
    f"{REGISTRY}/pr-lens-sandbox-node"
    "@sha256:8d35949bf87216136d840b0a8437d42105fd914929b35bd6a2112175265673a9"
)
