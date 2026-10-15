# 0006: Serialize turns per session with a session-level Postgres advisory lock

- Status: accepted (S1)
- Code: `memagent/store.py::session_lock`, `memagent/agent/loop.py::run_turn`

## Context

Two requests for the same session can arrive at once: a double-click, two browser tabs, a retry. Each turn reads the history, calls the model (seconds to minutes), and appends events with `seq = max + 1`. Without coordination, the two turns interleave their events, each answers without seeing the other, and both can compute the same `seq` (the UNIQUE constraint then fails one of them).

## Options

1. **An in-process `asyncio.Lock` per session.** Breaks as soon as there are two API processes.
2. **`SELECT ... FOR UPDATE` on the session row** inside one transaction for the whole turn.
3. **`pg_advisory_xact_lock`**, which is transaction-scoped. Also forces the whole turn, model calls included, into one long transaction.
4. **`pg_advisory_lock` held on a dedicated connection**, which is session-scoped, for the duration of the turn. Events are committed one by one on other connections.

## Choice

Option 4.

- **Works across processes**, because Postgres arbitrates.
- **No long transactions.** Options 2 and 3 would keep one transaction open across LLM calls. A crash would roll back every event of the turn, and the connection would sit "idle in transaction" holding row versions. With option 4, each event commits immediately, so the log stays accurate even after a crash.
- **Crash-safe release.** If the process dies, Postgres closes the connection and drops the lock. If unlocking fails, we invalidate the pooled connection rather than return it with the lock still held.
- The second request waits, then rebuilds history from the database, so it sees the first turn complete.
- The lock key is the first 8 bytes of the session UUID. A collision between two sessions would only serialize them, which is harmless.

## Cost

- Each in-flight turn holds one extra DB connection. Pool sizing must account for concurrent turns.
- A waiting request has no timeout in S1. It waits as long as the turn before it takes.
- `seq = max + 1` is only correct under the lock. The UNIQUE constraint is the backstop that turns any future misuse into a loud error, not silent corruption.
