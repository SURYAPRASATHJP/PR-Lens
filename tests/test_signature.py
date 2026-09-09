import hashlib
import hmac

from pr_lens.api.security import is_valid_signature

SECRET = "s3cret"
BODY = b'{"action":"opened"}'


def valid_header(body: bytes = BODY, secret: str = SECRET) -> str:
    return "sha256=" + hmac.new(secret.encode(), body, hashlib.sha256).hexdigest()


def test_accepts_a_correct_signature() -> None:
    assert is_valid_signature(BODY, valid_header(), SECRET)


def test_rejects_a_signature_made_with_another_secret() -> None:
    assert not is_valid_signature(BODY, valid_header(secret="wrong"), SECRET)


def test_rejects_when_the_body_changed_after_signing() -> None:
    assert not is_valid_signature(BODY + b" ", valid_header(), SECRET)


def test_rejects_a_missing_header() -> None:
    assert not is_valid_signature(BODY, None, SECRET)


def test_rejects_a_header_without_the_algorithm_prefix() -> None:
    bare = valid_header().removeprefix("sha256=")
    assert not is_valid_signature(BODY, bare, SECRET)


def test_rejects_the_sha1_header_github_also_sends() -> None:
    sha1 = "sha1=" + hmac.new(SECRET.encode(), BODY, hashlib.sha1).hexdigest()
    assert not is_valid_signature(BODY, sha1, SECRET)


def test_rejects_garbage_without_raising() -> None:
    assert not is_valid_signature(BODY, "sha256=not-hex", SECRET)
