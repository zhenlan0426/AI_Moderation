"""
Utility functions for text normalization used in the AI_Moderation project.

Currently includes:
1. normalize_urls      – Replace URLs with placeholder <URL_{domain}> (domain kept).
2. normalize_usernames – Replace Reddit and @-style user mentions with <USER>.
3. normalize_text      – Convenience wrapper that applies both.

Rationale
---------
• Exact URLs and user names rarely matter for rule-violation classification, but the presence of a link and the domain often do.
• Keeping only the domain reduces vocabulary size while retaining potentially useful signal (e.g., youtube.com vs twitter.com).
• Replacing user mentions removes nearly-unique identifiers that otherwise bloat the tokenizer’s sub-word vocabulary.
"""
from __future__ import annotations

import re
from urllib.parse import urlparse

__all__ = [
    "normalize_urls",
    "normalize_usernames",
    "normalize_text",
]

# ---------------------------------------------------------------------------
# Regex patterns (compiled once at import time)
# ---------------------------------------------------------------------------

# Generic URL recognizer – looks for http(s)://, ftp://, or bare www.<domain>
_URL_RE: re.Pattern[str] = re.compile(
    r"(?:(?:https?://|ftp://|www\.)[^\s]+)",  # stop at whitespace
    flags=re.IGNORECASE,
)

# Reddit user mention formats: u/username or /u/username (case-insensitive)
_REDDIT_USER_RE: re.Pattern[str] = re.compile(r"(?<!\w)/?u/[A-Za-z0-9_-]+", flags=re.IGNORECASE)

# @username mentions – capped at 30 chars, avoids picking up email addresses
_AT_USER_RE: re.Pattern[str] = re.compile(r"(?<!\w)@[A-Za-z0-9_]{1,30}\b")


# ---------------------------------------------------------------------------
# Public helpers
# ---------------------------------------------------------------------------

def normalize_urls(text: str) -> str:
    """Replace every URL in *text* with ``<URL_{domain}>`` while retaining the domain.

    Examples
    --------
    >>> normalize_urls("See https://sub.example.com/path?a=1 and http://example.org")
    'See <URL_sub.example.com> and <URL_example.org>'
    """

    def _replace(match: re.Match[str]) -> str:
        raw_url: str = match.group(0)

        # Ensure the URL parses even when scheme is missing (e.g., "www.google.com")
        to_parse = raw_url if re.match(r"https?://|ftp://", raw_url, flags=re.I) else f"http://{raw_url}"
        try:
            domain = urlparse(to_parse).netloc.split(":")[0].lower() or "unknown"
        except Exception:
            domain = "unknown"

        return f"<URL_{domain}>"

    return _URL_RE.sub(_replace, text)


def normalize_usernames(text: str) -> str:
    """Replace Reddit-style and @ mentions with the fixed token ``<USER>``."""
    text = _REDDIT_USER_RE.sub("<USER>", text)
    text = _AT_USER_RE.sub("<USER>", text)
    return text


def normalize_text(text: str) -> str:
    """Cascade of :pyfunc:`normalize_urls` then :pyfunc:`normalize_usernames`."""
    text = normalize_urls(text)
    text = normalize_usernames(text)
    return text


# ---------------------------------------------------------------------------
# Quick sanity test (executes only when run as a script)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    sample = (
        "Check https://www.reddit.com/r/python by u/someone, "
        "see www.example.com/foo and say hi to @Friend!"
    )
    print(normalize_text(sample))
