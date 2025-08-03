"""
Utility functions for text normalization used in the AI_Moderation project.

Currently includes:
1. normalize_urls        – Replace URLs with placeholder <URL_{domain}> (domain kept).
2. normalize_usernames   – Replace Reddit and @-style user mentions with <USER>.
3. normalize_emails      – Replace email addresses with <EMAIL>.
4. normalize_subreddits  – Replace subreddit mentions (r/…) with <SUB>.
5. normalize_phone_numbers – Replace phone numbers with <PHONE>.
6. normalize_money       – Replace dollar amounts with <MONEY>.
7. normalize_text        – Convenience wrapper that applies all of the above.

Rationale
---------
• Exact URLs, user names, emails, phone numbers, and specific dollar amounts rarely matter for rule-violation classification.
• Keeping only the domain for URLs reduces vocabulary size while retaining potentially useful signal (e.g., youtube.com vs twitter.com).
• Replacing personal identifiers removes nearly-unique tokens that otherwise bloat the tokenizer's sub-word vocabulary.
• This normalization helps models generalize better by focusing on content patterns rather than specific identifiers.
"""
from __future__ import annotations

import re
from urllib.parse import urlparse

__all__ = [
    "normalize_urls",
    "normalize_usernames", 
    "normalize_emails",
    "normalize_subreddits",
    "normalize_phone_numbers",
    "normalize_money",
    "normalize_text",
    "build_rule_example_lookup",
    "group_examples_by_rule",
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

# Email addresses
_EMAIL_RE: re.Pattern[str] = re.compile(
    r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b",
    flags=re.IGNORECASE,
)

# Reddit subreddit mentions: r/subreddit or /r/subreddit (case-insensitive)
_SUBREDDIT_RE: re.Pattern[str] = re.compile(r"(?<!\w)/?r/[A-Za-z0-9_-]+", flags=re.IGNORECASE)

# Phone numbers - various formats
_PHONE_RE: re.Pattern[str] = re.compile(
    r"(?:\+?1[-.\s]?)?\(?[0-9]{3}\)?[-.\s]?[0-9]{3}[-.\s]?[0-9]{4}|"  # US phone numbers
    r"1-800-[A-Z0-9-]+",  # 1-800 numbers with letters
    flags=re.IGNORECASE,
)

# Money amounts - currency symbols (e.g., "$1,000", "£50", "€3.2M") OR numeric/placeholder amounts followed by currency words
_MONEY_RE: re.Pattern[str] = re.compile(
    r'''(
        [\$£€]                              # Currency symbols
        [0-9]+(?:[,.][0-9]+)*                # Amount with optional separators
        (?:\s*(?:million|billion|k|M|B))?    # Optional scale/abbreviation
        |                                     # OR
        (?:[0-9]+|[Xx]{2,})\s*              # Number or placeholder
        (?:dollars?|pounds?|euros?)          # Currency words
    )''',
    flags=re.IGNORECASE | re.VERBOSE,
)


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


def normalize_emails(text: str) -> str:
    """Replace email addresses with ``<EMAIL>``."""
    return _EMAIL_RE.sub("<EMAIL>", text)


def normalize_subreddits(text: str) -> str:
    """Replace subreddit mentions (r/subreddit) with ``<SUB>``."""
    return _SUBREDDIT_RE.sub("<SUB>", text)


def normalize_phone_numbers(text: str) -> str:
    """Replace phone numbers with ``<PHONE>``."""
    return _PHONE_RE.sub("<PHONE>", text)


def normalize_money(text: str) -> str:
    """Replace money amounts (dollar, pound, euro, etc.) with ``<MONEY>``."""
    return _MONEY_RE.sub("<MONEY>", text)


def normalize_text(text: str) -> str:
    """Apply all normalization functions in sequence."""
    text = normalize_urls(text)
    text = normalize_usernames(text)
    text = normalize_emails(text)
    # text = normalize_subreddits(text)
    text = normalize_phone_numbers(text)
    text = normalize_money(text)
    return text


# ---------------------------------------------------------------------------
# Data1: Rule-based example aggregation
# ---------------------------------------------------------------------------

from typing import Dict, List
import pandas as pd

def group_examples_by_rule(df) -> Dict[str, Dict[str, List[str]]]:
    """Return deduplicated positive/negative lists per rule without any I/O or heavy normalisation.

    Parameters
    ----------
    df : pandas.DataFrame
        Pre-cleaned DataFrame containing the Data1 training rows.
    (Assumed to already include the relevant columns and be cleaned.)

    Returns
    -------
    dict
        Mapping ``{rule_text: {"positives": [...], "negatives": [...]}}``.
    """


    # Column names for positive and negative example sets
    pos_cols = ["positive_example_1", "positive_example_2"]
    neg_cols = ["negative_example_1", "negative_example_2"]

    def _collect(series_list):
        """Collapse a list of Series into unique values."""
        combined = pd.concat(series_list, ignore_index=True)
        return combined.unique().tolist()

    result: Dict[str, Dict[str, List[str]]] = {}

    for rule, group in df.groupby("rule", sort=False):
        rule = str(rule).strip()
        pos_examples = _collect([group[c] for c in pos_cols])
        neg_examples = _collect([group[c] for c in neg_cols])
        result[rule] = {"positives": pos_examples, "negatives": neg_examples}
    return result


# ---------------------------------------------------------------------------
# Quick sanity test (executes only when run as a script)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    sample = (
        "Check https://www.reddit.com/r/python by u/someone, "
        "email me at kingfavoursolutiontemple@yahoo.com or visit www.example.com/foo, "
        "call 1-800-99-LAW-USA for $100 discount! Also check r/AskReddit and @friend"
    )
    print("Original:", sample)
    print("Normalized:", normalize_text(sample))
