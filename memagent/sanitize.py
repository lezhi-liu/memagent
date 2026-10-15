"""The single scrubbing gateway.

Everything that persists user content (events, workspace files) passes through `scrub()`
or `scrub_json()` first. It is rules and regexes only: synchronous, deterministic, no LLM,
so it can never be skipped because a model was slow or down.

Each hit is replaced with `[REDACTED:<type>]`. Rules run in a fixed order:

1. private_key: PEM blocks go first, because their body would otherwise trip other rules.
2. api_key: well-known key prefixes.
3. secret: the value after words like "password" / "token" / "密码".
4. cn_id: 18-digit Chinese national ID with a valid check digit. This runs before
   bank_card because an 18-digit ID can also pass the Luhn check.
5. bank_card: 13-19 digits that pass the Luhn check.

Known cost: false positives. Any Luhn-valid digit run is treated as a card, and
"the secret is that ..." loses the word "that". Over-redaction is the safe direction.
"""

import re
from collections import Counter
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

REDACTED_PREFIX = "[REDACTED:"


def _marker(kind: str) -> str:
    return f"{REDACTED_PREFIX}{kind}]"


@dataclass(frozen=True)
class ScrubResult:
    text: str
    counts: dict[str, int] = field(default_factory=dict[str, int])


# ---- 1. PEM private key blocks -------------------------------------------------------

_PRIVATE_KEY = re.compile(
    r"-----BEGIN (?:[A-Z0-9]+ )*PRIVATE KEY-----.*?-----END (?:[A-Z0-9]+ )*PRIVATE KEY-----",
    re.DOTALL,
)

# ---- 2. API keys by well-known prefix --------------------------------------------------

_API_KEY = re.compile(
    "|".join(
        [
            r"\bsk-[A-Za-z0-9_\-]{20,}",  # OpenAI (sk-, sk-proj-) and Anthropic (sk-ant-)
            r"\b(?:sk|rk|pk)_(?:live|test)_[A-Za-z0-9]{16,}",  # Stripe
            r"\b(?:AKIA|ASIA)[0-9A-Z]{16}\b",  # AWS access key id
            r"\bgh[pousr]_[A-Za-z0-9]{36,}",  # GitHub tokens
            r"\bgithub_pat_[A-Za-z0-9_]{22,}",  # GitHub fine-grained tokens
            r"\bxox[abposr]-[A-Za-z0-9\-]{10,}",  # Slack
            r"\bAIza[0-9A-Za-z_\-]{35}",  # Google API key
            r"\bmk_[A-Za-z0-9_\-]{40,}",  # our own memagent API keys
            r"\beyJ[A-Za-z0-9_\-]{10,}\.[A-Za-z0-9_\-]{10,}\.[A-Za-z0-9_\-]{10,}",  # JWT
        ]
    )
)

# ---- 3. "keyword <separator> value" -----------------------------------------------------

# The value: a quoted string, or a run of characters up to whitespace, a comma/semicolon,
# or full-width Chinese punctuation. ASCII "!" and "?" are kept in the value because
# passwords often contain them; over-redacting a trailing "!" is the safe direction.
_VALUE = r"(?P<value>\"[^\"]+\"|'[^']+'|[^\s,;，。；！？\"'、)）]+)"  # noqa: RUF001 (full-width punctuation is intended)

_SECRET_EN = re.compile(
    r"(?i)(?P<prefix>\b[a-z_\-]*?(?:password|passwd|passphrase|pwd|secret|token"
    r"|api[_\-]?key|access[_\-]?key|private[_\-]?key)"
    r"[\"']?\s*(?:[:=]|\bis\b|\bwas\b)\s*)" + _VALUE
)

_SECRET_ZH = re.compile(
    r"(?P<prefix>(?:密码|口令|令牌|密钥|秘钥|私钥)\s*(?:[:：=]|是|为)\s*)" + _VALUE  # noqa: RUF001
)

# ---- 4. Chinese national ID (18 digits, last may be X) --------------------------------

_CN_ID = re.compile(r"(?<![0-9])[0-9]{17}[0-9Xx](?![0-9])")
_CN_ID_WEIGHTS = (7, 9, 10, 5, 8, 4, 2, 1, 6, 3, 7, 9, 10, 5, 8, 4, 2)
_CN_ID_CHECK = "10X98765432"  # ISO 7064 MOD 11-2 check characters, indexed by sum % 11

# ---- 5. Bank card: 13-19 digits, optionally grouped by single spaces or hyphens ------

_CARD = re.compile(r"(?<![0-9])(?:[0-9][ \-]?){12,18}[0-9](?![0-9])")


def _cn_id_ok(candidate: str) -> bool:
    total = sum(int(d) * w for d, w in zip(candidate[:17], _CN_ID_WEIGHTS, strict=True))
    return _CN_ID_CHECK[total % 11] == candidate[17].upper()


def _luhn_ok(digits: str) -> bool:
    total = 0
    # Walk from the rightmost digit; double every second one.
    for i, ch in enumerate(reversed(digits)):
        d = int(ch)
        if i % 2 == 1:
            d *= 2
            if d > 9:
                d -= 9
        total += d
    return total % 10 == 0


def _sub_all(pattern: re.Pattern[str], kind: str, text: str, counts: Counter[str]) -> str:
    def repl(m: re.Match[str]) -> str:
        counts[kind] += 1
        return _marker(kind)

    return pattern.sub(repl, text)


def _sub_secret_values(pattern: re.Pattern[str], text: str, counts: Counter[str]) -> str:
    """Replace only the value, keeping the keyword so the text stays readable."""

    def repl(m: re.Match[str]) -> str:
        value = m.group("value")
        if value.startswith(REDACTED_PREFIX):
            return m.group(0)  # already redacted by an earlier rule (e.g. "token: sk-...")
        counts["secret"] += 1
        return m.group("prefix") + _marker("secret")

    return pattern.sub(repl, text)


def _sub_checked(
    pattern: re.Pattern[str],
    kind: str,
    check: Callable[[str], bool],
    text: str,
    counts: Counter[str],
) -> str:
    """Replace a match only if `check(digits)` passes; keeps false-positive rate down."""

    def repl(m: re.Match[str]) -> str:
        digits = re.sub(r"[ \-]", "", m.group(0))
        if check(digits):
            counts[kind] += 1
            return _marker(kind)
        return m.group(0)

    return pattern.sub(repl, text)


def scrub(text: str) -> ScrubResult:
    counts: Counter[str] = Counter()
    text = _sub_all(_PRIVATE_KEY, "private_key", text, counts)
    text = _sub_all(_API_KEY, "api_key", text, counts)
    text = _sub_secret_values(_SECRET_EN, text, counts)
    text = _sub_secret_values(_SECRET_ZH, text, counts)
    text = _sub_checked(_CN_ID, "cn_id", _cn_id_ok, text, counts)
    text = _sub_checked(_CARD, "bank_card", _luhn_ok, text, counts)
    return ScrubResult(text=text, counts=dict(counts))


def scrub_json(value: Any) -> tuple[Any, dict[str, int]]:
    """Scrub every string inside a JSON-like value. Dict keys are ours, so they are kept."""
    counts: Counter[str] = Counter()

    def walk(v: Any) -> Any:
        if isinstance(v, str):
            result = scrub(v)
            counts.update(result.counts)
            return result.text
        if isinstance(v, dict):
            return {k: walk(item) for k, item in v.items()}  # pyright: ignore[reportUnknownVariableType]
        if isinstance(v, list):
            return [walk(item) for item in v]  # pyright: ignore[reportUnknownVariableType]
        return v

    return walk(value), dict(counts)
