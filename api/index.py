"""Vercel entry point.

Vercel's Python runtime looks for a module-level ASGI application in a file under api/,
and vercel.json rewrites every path here, so /webhooks/github and /health resolve on the
deployment URL unchanged. The app itself knows nothing about Vercel: this file is the only
host-specific thing in the repo, which is the point.
"""

from pr_lens.api.main import app

__all__ = ["app"]
