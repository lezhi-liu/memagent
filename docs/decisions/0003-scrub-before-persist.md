# 0003: Scrub sensitive data before persisting it, and the model sees only scrubbed content

- Status: accepted (S1)
- Code: `memagent/sanitize.py`, `memagent/store.py::append_event`, `memagent/tools/workspace.py`, `memagent/agent/history.py`

## Context

Users paste passwords, API keys, ID numbers and card numbers into chats. Whatever we store sits in the raw log permanently (decision 0002), gets copied into derived memory later, and will be sent to cloud models again and again on recall. A secret that reaches disk once is effectively leaked.

## Options

1. **Store raw, scrub on read/display.** Every reader has to remember to scrub, and backups, dumps and future memory jobs all hold the raw value.
2. **Store raw, encrypted**, and decrypt for the model. This protects against disk theft but not against the model or derived memory spreading the value.
3. **Scrub before persisting**, at a single gateway: rules and regexes only, synchronous, no LLM. The model's history is rebuilt from the database, so the model sees only the scrubbed text too.

## Choice

Option 3.

- **One gateway, one write path.** `store.append_event` scrubs every string in the payload. The workspace write tool scrubs file content. Nothing else writes user content. S1 logs no user content at all, only ids, counts and exception class names.
- **Deterministic rules, not an LLM.** Scrubbing can't be skipped because a model is slow, down or over budget, and it is testable exactly. The rules cover: PEM private keys, well-known API key prefixes (OpenAI, Anthropic, AWS, GitHub, Google, Slack, Stripe, JWTs, our own `mk_` keys), values after `password` / `token` / `secret` / `api_key` and Chinese `密码` / `口令` / `令牌` / `密钥`, 18-digit Chinese ID numbers with a valid check digit, and 13–19-digit card numbers that pass Luhn.
- **History is rebuilt from the database**, so the model sees `[REDACTED:secret]`, never the value. Even the current turn's message is stored first and read back.

## Cost

- **Tasks that need the real secret are impossible.** "Log in with password X" or "use this API key to call Y" can't work: the model only ever sees the marker. The system prompt tells the model this, so it can explain instead of guessing.
- **False positives.** Any Luhn-valid 13–19-digit number (some order or tracking numbers) is redacted, and "the secret is that…" loses the word "that". We accept over-redaction as the safe direction.
- **False negatives.** Secrets in unknown formats with no keyword nearby get through. Rules are a floor, not a guarantee, and new formats need new rules and tests.
- The user's raw input exists in process memory for the length of one request. That is unavoidable for any server that receives it.
