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
import random
import math
from typing import Dict, List, Sequence

import pandas as pd
import torch
import numpy as np

# ---------------------------------------------------------------------------
# Rule variants (paraphrased)
# ---------------------------------------------------------------------------
from rules import RULE_VARIANTS
from torch.utils.data import Dataset, IterableDataset, DataLoader


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
# Prompt construction helper shared by TTT datasets
# ---------------------------------------------------------------------------

def build_ttt_prompt(
    rule: str,
    comment1: str,
    label1: int | str,
    comment2: str,
    label2: int | str,
    comment_test: str,
    subreddit: str | None = None,
) -> str:
    """Construct a TTT classification prompt.

    This helper implements rule-variant sampling via ``RULE_VARIANTS`` and is
    shared by both `TTTDataset_map` and `TTTDataset_iter` to avoid code
    duplication.
    """

    def _lab_to_str(l: int | str) -> str:
        # Accept either 0/1 or "Yes"/"No" representations
        if isinstance(l, str):
            return l
        return "Yes" if l == 1 else "No"

    rule_text = random.choice(RULE_VARIANTS[rule])

    header = (
        "You are given a comment on reddit. Your task is to classify if it violates the given rule.\n"
    )
    if subreddit is not None:
        header += f"Subreddit: r/{subreddit}\n"

    prompt = (
        header
        + f"Rule: {rule_text}\n"
        + f"Comment: {comment1}\n"
        + f"Violation: {_lab_to_str(label1)}\n"
        + f"Comment: {comment2}\n"
        + f"Violation: {_lab_to_str(label2)}\n"
        + f"Comment: {comment_test}\n"
        + "Violation:"
    )
    return prompt

# ---------------------------------------------------------------------------
# Shared helpers: support sampling and "Violation:" index finding
# ---------------------------------------------------------------------------

def sample_support_numeric(pools: Dict[str, Sequence[str]]) -> tuple[str, int, str, int]:
    """Randomly select one positive and one negative example and randomise their order.

    Returns
    -------
    tuple
        (comment1, label1, comment2, label2) with numeric labels (1 = violation, 0 = non-violation).
    """
    pos_example = random.choice(pools["positives"])
    neg_example = random.choice(pools["negatives"])
    if random.random() < 0.5:
        return pos_example, 1, neg_example, 0
    return neg_example, 0, pos_example, 1


def find_violation_indices(input_ids: torch.Tensor, violation_ids: torch.Tensor) -> list[int]:
    """Return indices of the last token of each occurrence of *violation_ids* inside *input_ids*."""
    indices: list[int] = []
    token_window = len(violation_ids)
    seq_len = input_ids.size(0)
    for i in range(seq_len - token_window + 1):
        if torch.equal(input_ids[i : i + token_window], violation_ids):
            indices.append(i + token_window - 1)
    return indices


# ---------------------------------------------------------------------------
# Data1: Rule-based example aggregation
# ---------------------------------------------------------------------------
"""Dataset & DataLoader for Reddit rule-violation classification - Data1.

For every row in the provided DataFrame we create a prompt that follows the
`ttt_design.md` template:

    You are given a comment on reddit. Your task is to classify if it violates the given rule.
    Subreddit: r/<subreddit>
    Rule: <rule text>
    Comment: <support example 1>
    Violation: <Yes|No>
    Comment: <support example 2>
    Violation: <Yes|No>
    Comment: <target comment>
    Violation:

Two labelled *support* examples (one violating, one non-violating) are sampled
for the same rule. Their order is randomised.  The unlabelled *target* comment
(the row's own *body*) is appended last – the model must produce an answer
right after the final "Violation:" token.  To facilitate that, this dataset
returns the *position* (index) of the first token of the **last** "Violation:"
string in the tokenised sequence.

The caller can then gather the model's hidden states / logits at this position
to train a classifier or compute loss directly.
"""

from typing import Dict, List
import pandas as pd

class TTTDataset_map(Dataset):
    """PyTorch `Dataset` that yields tokenised TTT prompts.

    Parameters
    ----------
    df
        *Cleaned* DataFrame containing at least the following columns::
            ["subreddit", "rule", "body", "positive_example_1", "negative_example_1"]
        Additional columns (e.g. `positive_example_2`) are ignored but allowed.

    grouped_examples
        Mapping produced by ``utility.group_examples_by_rule`` (or an equivalent
        function).  The structure must be::

            {rule_text: {"positives": List[str], "negatives": List[str]}}

        The positive / negative pools are used to randomly sample *support*
        examples for each datum.

    tokenizer
        Any HuggingFace *PreTrainedTokenizer* instance compatible with your
        language model.

    max_length
        Sequence length after padding / truncation.  Defaults to 512.

    Notes
    -----
    •  No heavy preprocessing is performed here – we assume `df` has already
       been cleaned / normalised.
    •  The dataset is *stateless* w.r.t. sampling order. Every `__getitem__`
       call produces a fresh prompt, so subsequent epochs will see different
       support examples / orders by design.
    """

    violation_str: str = "Violation:"

    def __init__(
        self,
        df: pd.DataFrame,
        grouped_examples: Dict[str, Dict[str, List[str]]],
        tokenizer
    ) -> None:
        super().__init__()
        self.df = df.reset_index(drop=True)
        self.grouped_examples = grouped_examples
        self.tokenizer = tokenizer


        # Pre-encode "Violation:" once so we can search for it quickly later.
        self._violation_ids: torch.Tensor = tokenizer.encode(
            self.violation_str, add_special_tokens=False, return_tensors="pt"
        )[0]

    # ------------------------------------------------------------------
    # PyTorch Dataset interface
    # ------------------------------------------------------------------
    def __len__(self) -> int:  # noqa: D401
        return len(self.df)

    def __getitem__(self, idx: int):  # noqa: D401
        row = self.df.iloc[idx]
        rule = row["rule"]
        subreddit = row["subreddit"]
        comment1, label1, comment2, label2 = sample_support_numeric(self.grouped_examples[rule])
        prompt = build_ttt_prompt(rule,comment1,label1,comment2,label2,row["body"],subreddit)
        labels = torch.tensor([label1, label2], dtype=torch.long)

        input_ids = self.tokenizer.encode(
            prompt,
            add_special_tokens = True,
            return_tensors="pt",
        ).squeeze(0)

        # ------------------------------------------------------------------
        # Locate the final "Violation:" occurrence – that's where the model
        # must output its prediction.
        # ------------------------------------------------------------------
        vi_index = find_violation_indices(input_ids, self._violation_ids)


        return row["row_id"], input_ids.unsqueeze(0), torch.tensor(vi_index), labels

def build_dataloader_map(
    df: pd.DataFrame,
    tokenizer,
    batch_size: int = 1,
    shuffle: bool = True,
    num_workers: int = 4,
    pin_memory: bool = True,
    include_body: bool = False,
) -> DataLoader:
    """Return a ready-to-use PyTorch ``DataLoader`` for TTT training.
    
    Parameters
    ----------
    include_body : bool, optional
        If True, include the 'body' column content in the positive/negative lists
        based on the 'rule_violation' values. Defaults to False.
    """
    from generate_grouped_data import group_examples_by_rule
    dataset = TTTDataset_map(
        df=df,
        grouped_examples=group_examples_by_rule(df, include_body=include_body),
        tokenizer=tokenizer,
    )

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        collate_fn=lambda x:x[0] # only works for batch size one
    )    
# ---------------------------------------------------------------------------
# New iterable TTTDataset implementation for rule-level sampling
# ---------------------------------------------------------------------------
class TTTDataset_iter(IterableDataset):
    """PyTorch `IterableDataset` that yields tokenised TTT prompts.

    Each sample is constructed on-the-fly from two dictionaries containing
    positive and negative examples per rule – one for training, one for
    hold-out evaluation.

    The expected structure of each dictionary is::

        {rule_text: {"positives": List[str], "negatives": List[str]}}

    Parameters
    ----------
    data_pair
        Tuple ``(train_dict, holdout_dict)`` with the structure described
        above.
    tokenizer
        Any HuggingFace *PreTrainedTokenizer* compatible with your language
        model.
    samples_per_epoch
        Number of prompts that an epoch of this dataset should yield.
    """

    violation_str: str = "Violation:"

    def __init__(
        self,
        train_dict: Dict[str, Dict[str, List[str]]],
        holdout_dict: Dict[str, Dict[str, List[str]]],
        tokenizer,
        samples_per_epoch: int = 1000,
    ) -> None:
        super().__init__()
        self.train_dict = train_dict
        self.holdout_dict = holdout_dict
        self.tokenizer = tokenizer
        self.samples_per_epoch = samples_per_epoch

        # Rules present in *both* splits – we only sample from these.
        self.rules: List[str] = list(train_dict.keys())

        # Pre-encode "Violation:" for fast lookup later.
        self._violation_ids: torch.Tensor = tokenizer.encode(
            self.violation_str, add_special_tokens=False, return_tensors="pt"
        )[0]

    # ---------------------------------------------------------------------
    # Private helpers
    # ---------------------------------------------------------------------

    def _sample_target(self, rule: str) -> tuple[str, int]:
        """Return one *target* example (text, label) for *rule* from *holdout_dict*."""
        if random.random() < 0.5:
            return random.choice(self.holdout_dict[rule]["positives"]), 1
        return random.choice(self.holdout_dict[rule]["negatives"]), 0

    # ------------------------------------------------------------------
    # Dataset interface
    # ------------------------------------------------------------------
    def __len__(self) -> int:  # noqa: D401
        return self.samples_per_epoch

    def __iter__(self):
        """Yield a *finite* number of samples, each worker getting a distinct slice."""
        worker_info = torch.utils.data.get_worker_info()
        # Determine the number of samples for this worker
        if worker_info is None:  # Single-process data loading
            num_samples = self.samples_per_epoch
        else:  # In a worker process
            # Split workload. Each worker gets a fraction of the total samples.
            num_samples = int(math.ceil(self.samples_per_epoch / worker_info.num_workers))

        for _ in range(num_samples):
            rule = random.choice(self.rules)
            # Support examples (train split)
            comment1, lab1, comment2, lab2 = sample_support_numeric(self.train_dict[rule])
            # Target example (hold-out split)
            target_comment, lab3 = self._sample_target(rule)

            prompt = build_ttt_prompt(rule, comment1, lab1, comment2, lab2, target_comment)
            input_ids = self.tokenizer.encode(prompt, add_special_tokens=True, return_tensors="pt").squeeze(0)

            # Locate each "Violation:" occurrence (we need the last token index)
            vi_index = find_violation_indices(input_ids, self._violation_ids)

            labels = torch.tensor([lab1, lab2, lab3], dtype=torch.long)
            yield input_ids, torch.tensor(vi_index), labels



def load_grouped_data(
    data_dir: str = "Data/grouped"
) -> tuple[Dict[str, Dict[str, List[str]]], Dict[str, Dict[str, List[str]]]]:
    """
    Load pre-generated grouped training data from disk.
    
    This function loads grouped data that was generated by generate_grouped_data.py.
    The loaded data maintains the shared negatives structure for memory efficiency.
    
    Parameters
    ----------
    data_dir : str, optional
        Directory containing the grouped data files. Defaults to "Data/grouped".
        
    Returns
    -------
    tuple[Dict[str, Dict[str, List[str]]], Dict[str, Dict[str, List[str]]]]
        A tuple containing (train_grouped, holdout_grouped) where each dict has:
        {rule_text: {"positives": [...], "negatives": shared_list}}
        
    Examples
    --------
    >>> train_data, holdout_data = load_grouped_data()
    >>> # Use with existing TTT pipeline
    >>> dataset = TTTDataset(some_df, train_data, tokenizer)
    """
    import pickle
    import os
    
    train_path = os.path.join(data_dir, "train_grouped.pkl")
    holdout_path = os.path.join(data_dir, "holdout_grouped.pkl")
    
    # Load train data
    with open(train_path, 'rb') as f:
        train_data = pickle.load(f)
    
    # Load holdout data
    with open(holdout_path, 'rb') as f:
        holdout_data = pickle.load(f)
    
    return train_data, holdout_data

# ---------------------------------------------------------------------------
# DataLoader helper
# ---------------------------------------------------------------------------

def seed_worker(worker_id: int) -> None:
    """Initialise each DataLoader worker with its own RNG seed.

    This ensures workers draw *different* random samples while keeping the run
    reproducible given a fixed global seed.  Use it by passing

    ```python
    loader = DataLoader(
        dataset,
        num_workers=4,
        worker_init_fn=seed_worker,
        # ... other kwargs
    )
    ```
    """
    worker_seed = torch.initial_seed() % 2**32
    random.seed(worker_seed)
    np.random.seed(worker_seed)
    torch.manual_seed(worker_seed)

