# 0004: SHA-256, not bcrypt, for API keys

- Status: accepted (S1)
- Code: `memagent/auth.py`, `memagent/api/deps.py::current_user`

## Context

Each user authenticates with `Authorization: Bearer <api_key>`. We must not store the key itself: a database leak would hand out working credentials. So we store a hash, and the question is which kind.

## Options

1. **bcrypt / scrypt / argon2**: slow, salted hashes designed for human passwords.
2. **SHA-256** of a server-generated, high-entropy key.

## Choice

Option 2. Keys are `mk_` + `secrets.token_urlsafe(32)`, which is 256 bits of randomness, and the database stores `sha256(key)` hex with a UNIQUE index.

- **Slow hashes defend against guessing low-entropy secrets.** People choose passwords from a small space, so making each guess cost about 100 ms matters. A 256-bit random key cannot be brute-forced at any hash speed, so slowness adds nothing.
- **Lookup by hash.** With a deterministic hash, authentication is one indexed query: `WHERE api_key_hash = sha256(presented)`. bcrypt is salted per row, so we would need a separate key id to find the row, then a deliberately slow compare on every request.
- **No timing-attack concern** from the lookup: the attacker would need to match a 256-bit hash prefix, which leaks nothing useful.
- The `mk_` prefix makes leaked keys easy to spot, and `sanitize.py` recognizes it, so a key pasted into a chat is redacted.

## Cost

- This is only safe because **we** generate the keys. If users could choose their own keys, they would pick weak ones, and SHA-256 would then be the wrong choice.
- No built-in key rotation or expiry yet. That's a later concern, but the design allows several keys per user if we move the hash to its own table.
- Keys are shown once at creation. A lost key means creating a new one, because nothing can recover it.
