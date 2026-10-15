# 0002: An append-only raw log, with redaction as the only exception

- Status: accepted (S1)
- Code: `migrations/versions/0001_s1_initial.py` (trigger, `redact_event`), `memagent/store.py`

## Context

`events` is layer zero: every user message, model reply, tool call and tool result, in order. Every later memory layer (observations, facts, summaries) will be derived from it and must be traceable back to it. If the raw log can be edited, nothing derived from it can be trusted, and "why did the agent remember X?" has no reliable answer. Users must still be able to make the system forget things, and that is a legal requirement as well as a product one.

## Options

1. **Convention only**: the application never issues UPDATE or DELETE. One bug or one manual `psql` session breaks it, silently.
2. **Revoke UPDATE/DELETE privileges** from the application role. This is strong, but redaction still needs some path, and S1 runs as a single DB role. Role separation arrives with row-level security in S7.
3. **A database trigger** that rejects DELETE and TRUNCATE, and rejects every UPDATE except one exact shape of change: `redacted_at` goes NULL → non-NULL, the payload becomes exactly `{"redacted": true}`, and every other column stays the same. `redact_event(event_id, user_id)` is the function that makes that change.

## Choice

Option 3. The rule lives in the database, so it holds for every client: our code, migrations, scripts and a human with `psql`. The tests use raw SQL to prove it.

Details:
- The trigger checks the *shape* of the change, not who made it. A hand-written UPDATE that produces exactly a redaction would also pass. That is acceptable, because such an UPDATE has the same effect as a redaction and destroys nothing except the content the user asked to erase. Restricting it to the function alone needs a separate role plus `SECURITY DEFINER`, which is planned with S7.
- TRUNCATE is blocked too. Row triggers don't fire on it, so it gets its own statement-level trigger.
- `redact_event` filters by `user_id` like every other query, and returns false for another user's event.
- Committing per event (not per turn) means a crash mid-turn still leaves an accurate partial log.

## Cost

- Mistakes can't be fixed in place. A bad event stays, and a correction must be a new event.
- Test cleanup can't delete rows, so tests isolate by creating fresh users and rebuild the schema with `downgrade base`/`upgrade head`.
- Storage only grows. That's fine at this scale; archiving would be a later, explicit design.
- Redaction keeps the row and its position (seq, type, timestamp). This metadata, "something was said here at this time", survives a deletion. That's intentional for audit, but it should be documented to users in S7.
