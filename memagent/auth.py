"""API keys: generated with high entropy, stored only as a SHA-256 hash.

Why SHA-256 and not bcrypt (decision 0004): slow hashes exist to make guessing low-entropy
human passwords expensive. A 256-bit random key cannot be guessed no matter how fast the
hash is, and a fast hash lets us look the user up by `WHERE api_key_hash = ...` with an
index, instead of checking a bcrypt hash against every user.
"""

import hashlib
import secrets

KEY_PREFIX = "mk_"


def generate_api_key() -> str:
    # 32 random bytes = 256 bits of entropy; the prefix makes leaked keys easy to spot
    # (and lets sanitize.py recognise them).
    return KEY_PREFIX + secrets.token_urlsafe(32)


def hash_api_key(key: str) -> str:
    return hashlib.sha256(key.encode("utf-8")).hexdigest()
