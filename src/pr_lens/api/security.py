import hmac
from hashlib import sha256

SIGNATURE_HEADER = "X-Hub-Signature-256"
_PREFIX = "sha256="


def is_valid_signature(body: bytes, header: str | None, secret: str) -> bool:
    """Check GitHub's HMAC over the raw request body.

    `body` must be the exact bytes GitHub sent. Parsing the JSON and re-serialising it
    changes key order and whitespace, and the signature no longer matches.

    A missing or malformed header is a failure. Treating it as "unsigned, allow through"
    is how these endpoints get owned.
    """
    if header is None or not header.startswith(_PREFIX):
        return False
    expected = hmac.new(secret.encode(), body, sha256).hexdigest()
    return hmac.compare_digest(header[len(_PREFIX) :], expected)
