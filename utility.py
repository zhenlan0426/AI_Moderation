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
from typing import Dict, List, Sequence, Tuple

import pandas as pd
import torch
import numpy as np

# ---------------------------------------------------------------------------
# Rule variants (paraphrased)
# ---------------------------------------------------------------------------
try:
    from rules import RULE_VARIANTS
except ImportError:
    RULE_VARIANTS = None
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


def normalize_text_columns(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """Apply ``normalize_text`` to selected text columns of a DataFrame.
    """

    columns = (
        "body",
        "positive_example_1",
        "positive_example_2",
        "negative_example_1",
        "negative_example_2",
    )

    for col in columns:
        df[col] = df[col].map(normalize_text)
    return df


# ---------------------------------------------------------------------------
# Base class shared by TTT datasets
# ---------------------------------------------------------------------------

class TTTDatasetBase:
    """Shared utilities for TTT datasets (map and iter variants)."""

    # -----------------------
    # Sampler helpers
    # -----------------------
    @staticmethod
    def _init_sampler_state(
        rule_to_pools: Dict[str, Dict[str, List[object]]],
    ) -> Dict[str, Dict[str, List[int]]]:
        state: Dict[str, Dict[str, List[int]]] = {}
        for rule, pools in rule_to_pools.items():
            pos = pools.get("positives", [])
            neg = pools.get("negatives", [])
            state[rule] = {
                "positives": [0, len(pos)],
                "negatives": [0, len(neg)],
            }
        return state

    @staticmethod
    def _shuffle_sampler_state_inplace(rule_to_pools: Dict[str, Dict[str, List[object]]]) -> None:
        for _, pools in rule_to_pools.items():
            for key in ("positives", "negatives"):
                items = pools.get(key)
                random.shuffle(items)

    @staticmethod
    def _next_from_pool(
        pool_items: List[object],
        pool_state: List[int],
        *,
        reshuffle_on_cycle: bool,
    ) -> tuple[int, object]:
        cursor, total_length = pool_state
        idx = cursor
        item = pool_items[idx]
        cursor += 1
        if cursor >= total_length:
            cursor = 0
            if reshuffle_on_cycle:
                random.shuffle(pool_items)
        pool_state[0] = cursor
        return idx, item

    @staticmethod
    def _sample_from_state(
        rule_to_pools: Dict[str, Dict[str, List[object]]],
        rule_state: Dict[str, Dict[str, List[int]]],
        rule: str,
        desired_label: str,
        *,
        reshuffle_on_cycle: bool,
    ) -> tuple[int, object, int]:
        pool = rule_to_pools[rule][desired_label]
        state = rule_state[rule][desired_label]
        idx, item = TTTDatasetBase._next_from_pool(
            pool, state, reshuffle_on_cycle=reshuffle_on_cycle
        )
        return idx, item, 1 if desired_label == "positives" else 0

    # -----------------------
    # Prompt-token helpers
    # -----------------------
    def _enc(self, text: str, *, add_special_tokens: bool = False) -> torch.Tensor:
        ids = self.tokenizer.encode(text, add_special_tokens=add_special_tokens, return_tensors="pt")[0]
        if self._old_to_new is not None:
            ids = self._old_to_new.index_select(0, ids)
        return ids
        
    def pretokenize_ttt_fragments(
        self,
        rules_to_tokenize: List[str] | None = None,
    ) -> None:

        # Keep BOS/EOS out to maintain simple index math
        self._header_ids = self._enc(
            "You are given a comment on reddit. Your task is to classify if it violates the given rule.\n",
        )
        self._comment_prefix_ids = self._enc("Comment: ")
        self._newline_ids = self._enc("\n")
        self._violation_yes_line_ids = self._enc("Violation: Yes\n")
        self._violation_no_line_ids = self._enc("Violation: No\n")
        self._violation_prompt_ids = self._enc("Violation:")

        rule_variants_ids: Dict[str, List[torch.Tensor] | torch.Tensor] = {}
        if rules_to_tokenize is None:
            # default: use paraphrase variants from RULE_VARIANTS
            for rule_text, variants in RULE_VARIANTS.items():
                encoded_variants: List[torch.Tensor] = []
                for v in variants:
                    encoded_variants.append(self._enc(f"Rule: {v}\n"))
                rule_variants_ids[rule_text] = encoded_variants
        else:
            # map-style: tokenize only the provided unique rules (single variant)
            for rule_text in rules_to_tokenize:
                rule_variants_ids[rule_text] = self._enc(f"Rule: {rule_text}\n")
        self._rule_variants_ids = rule_variants_ids

    def compute_two_violation_end_indices(
        self,
        *,
        rule_variant_ids: torch.Tensor,
        support_ids: torch.Tensor,
        total_length: int,
    ) -> list[int]:
        prefix_len = (
            self._header_ids.numel()
            + rule_variant_ids.numel()
            + self._comment_prefix_ids.numel()
            + support_ids.numel()
            + self._newline_ids.numel()
        )
        first_violation_end = prefix_len + self._violation_prompt_ids.numel() - 1
        second_violation_end = total_length - 1
        return [first_violation_end, second_violation_end]



# ---------------------------------------------------------------------------
# Data1: Rule-based example aggregation
# ---------------------------------------------------------------------------
"""Dataset & DataLoader for Reddit rule-violation classification - Data1.

For every row in the provided DataFrame we create a prompt that follows the
`ttt_design.md` template:

    You are given a comment on reddit. Your task is to classify if it violates the given rule.
    Rule: <rule text>
    Comment: <support example>
    Violation: <Yes|No>
    Comment: <target comment>
    Violation:

 One labelled *support* example (randomly positive or negative) is sampled for
 the same rule.  The unlabelled *target* comment
(the row's own *body*) is appended last – the model must produce an answer
right after the final "Violation:" token.  To facilitate that, this dataset
returns the *position* (index) of the first token of the **last** "Violation:"
string in the tokenised sequence.

The caller can then gather the model's hidden states / logits at this position
to train a classifier or compute loss directly.
"""

class TTTDataset_map(TTTDatasetBase, Dataset):
    """
    Important Note: only works for num_workers = 1, as _sample_from_state is stateful and each worker start
    from the same cursor.
    Parameters
    ----------
    df
        *Cleaned* DataFrame containing at least the following columns::
            ["rule", "body", "positive_example_1", "negative_example_1"]
        Additional columns (e.g. `positive_example_2`) are ignored but allowed. Any
        `subreddit` column, if present, is ignored in prompt construction.

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
    •  Each row is yielded twice per epoch: once with a positive support and
       once with a negative support. Which one appears first is chosen per-row
       at initialisation time.
    •  Support examples are drawn using a simple per-rule cyclic sampler with
       in-place shuffling on cycle, providing stable coverage across epochs.
    """

    violation_str: str = "Violation:"

    def __init__(
        self,
        df: pd.DataFrame,
        grouped_examples: Dict[str, Dict[str, List[List[int]]]],
        tokenizer,
        old_to_new: torch.Tensor | None = None,
        Is_DEBUG: bool = False, # mute all randomization if True
    ) -> None:
        super().__init__()
        self.df = df.reset_index(drop=True)
        self.grouped_examples = grouped_examples
        self.tokenizer = tokenizer
        self._old_to_new = old_to_new
        self.Is_DEBUG = Is_DEBUG

        # Pre-encode static prompt fragments for consistency with iter dataset
        # Note: map dataset still builds full prompt as text below; this keeps API aligned
        # Pre-tokenize static fragments and unique rules present in df (vectorized)
        unique_rules: List[str] = self.df["rule"].astype(str).unique().tolist()
        self.pretokenize_ttt_fragments(rules_to_tokenize=unique_rules)

        # Initial sampler state for per-rule cyclic support sampling
        self._sampler_state = TTTDatasetBase._init_sampler_state(self.grouped_examples)
        if not Is_DEBUG:
            TTTDatasetBase._shuffle_sampler_state_inplace(self.grouped_examples)

        # Decide per-row which polarity (positive/negative) appears first
        if Is_DEBUG:
            self._first_positive_flags = np.ones(len(self.df))
        else:
            self._first_positive_flags = np.random.rand(len(self.df)) < 0.5
        self._expanded_len: int = 2 * len(self.df)

    # ------------------------------------------------------------------
    # PyTorch Dataset interface
    # ------------------------------------------------------------------
    def __len__(self) -> int:  # noqa: D401
        return self._expanded_len

    def __getitem__(self, idx: int):  # noqa: D401
        base_idx = idx // 2
        variant_idx = idx % 2  # 0 -> first variant, 1 -> second variant
        row = self.df.iloc[base_idx]
        rule = row["rule"]
        # Choose polarity for this variant based on per-row first/second ordering
        first_is_positive = self._first_positive_flags[base_idx]
        if variant_idx == 0:
            desired_label = "positives" if first_is_positive else "negatives"
        else:
            desired_label = "negatives" if first_is_positive else "positives"

        # Deterministic cyclic sampling of the chosen support polarity for this rule
        _, comment_train, label_train = TTTDatasetBase._sample_from_state(
            self.grouped_examples,
            self._sampler_state,
            rule,
            desired_label,
            reshuffle_on_cycle=not self.Is_DEBUG,
        )
        comment_train = torch.tensor(comment_train)
        # Tokenize dynamic target part using exposed encoder (no specials)
        comment_test = self._enc(row["body"])

        # for inference, we dont have rule variants
        # TODO: consider generate rule variants for inference
        rule_variant_ids = self._rule_variants_ids[rule]

        # Assemble the full prompt from pre-tokenised pieces
        pieces = [
            self._header_ids,
            rule_variant_ids,
            self._comment_prefix_ids,
            comment_train,
            self._newline_ids,
            self._violation_yes_line_ids if label_train == 1 else self._violation_no_line_ids,
            self._comment_prefix_ids,
            comment_test,
            self._newline_ids,
            self._violation_prompt_ids,
        ]
        input_ids = torch.cat(pieces, dim=0)

        # Compute the end indices for both occurrences of "Violation:"
        vi_index = self.compute_two_violation_end_indices(
            rule_variant_ids=rule_variant_ids,
            support_ids=comment_train,
            total_length=input_ids.numel(),
        )

        labels = torch.tensor([label_train], dtype=torch.long)


        return row["row_id"], input_ids.unsqueeze(0), torch.tensor(vi_index), labels

def group_examples_by_rule(df, include_body=False, tokenizer=None) -> Dict[str, Dict[str, List[str]]]:
    """Return deduplicated positive/negative lists per rule without any I/O or heavy normalisation.

    Parameters
    ----------
    df : pandas.DataFrame
        Pre-cleaned DataFrame containing the Data1 training rows.
        (Assumed to already include the relevant columns and be cleaned.)
    include_body : bool, optional
        If True, include the 'body' column content in the positive/negative lists
        based on the 'rule_violation' values. Bodies with rule_violation=1 are
        added to positives, bodies with rule_violation=0 are added to negatives.
        Defaults to False.

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

    def _encode(text: List[str]) -> List[List[int]]:
        return tokenizer.batch_encode_plus(text, add_special_tokens=False)["input_ids"]
    
    result: Dict[str, Dict[str, List[str]]] = {}

    for rule, group in df.groupby("rule", sort=False):
        rule = str(rule).strip()
        
        # Build series lists for positive and negative examples
        pos_series_list = [group[c] for c in pos_cols]
        neg_series_list = [group[c] for c in neg_cols]
        
        # Optionally include body content based on rule_violation
        if include_body:
            # Bodies that violate the rule (rule_violation=1) go to positives
            violating_bodies = group[group["rule_violation"] == 1]["body"]
            pos_series_list.append(violating_bodies)
            
            # Bodies that don't violate the rule (rule_violation=0) go to negatives  
            non_violating_bodies = group[group["rule_violation"] == 0]["body"]
            neg_series_list.append(non_violating_bodies)
        
        # Collect and deduplicate once per group
        pos_examples = _collect(pos_series_list)
        neg_examples = _collect(neg_series_list)
        if tokenizer is not None:
            pos_examples = _encode(pos_examples)
            neg_examples = _encode(neg_examples)
        result[rule] = {"positives": pos_examples, "negatives": neg_examples}
    return result

def build_dataloader_map(
    df: pd.DataFrame,
    tokenizer,
    shuffle: bool = False,
    pin_memory: bool = True,
    include_body: bool = False,
    grouped_examples: Dict[str, Dict[str, List[str]]] | None = None,
    Is_DEBUG: bool = False,
) -> DataLoader:
    """Return a ready-to-use PyTorch ``DataLoader`` for TTT training.
    
    Parameters
    ----------
    include_body : bool, optional
        If True, include the 'body' column content in the positive/negative lists
        based on the 'rule_violation' values. Defaults to False.
    """
    dataset = TTTDataset_map(
        df=df,
        grouped_examples=group_examples_by_rule(df, include_body=include_body, tokenizer=tokenizer) if grouped_examples is None else grouped_examples,
        tokenizer=tokenizer,
        Is_DEBUG=Is_DEBUG,
    )

    return DataLoader(
        dataset,
        batch_size=1,
        shuffle=shuffle,
        num_workers=0, # NOTE: num_workers=0 is required due to the stateful sampler
        pin_memory=pin_memory,
        collate_fn=lambda x: x[0]  # Extract single batch element (batch_size must be 1)
    )    
# ---------------------------------------------------------------------------
# New iterable TTTDataset implementation for rule-level sampling
# ---------------------------------------------------------------------------
class TTTDataset_iter(TTTDatasetBase, IterableDataset):
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
        train_dict: Dict[str, Dict[str, List[torch.Tensor]]],
        holdout_dict: Dict[str, Dict[str, List[torch.Tensor]]],
        tokenizer,
        old_to_new: torch.Tensor | None = None,
        samples_per_epoch: int = 1000,
    ) -> None:
        super().__init__()
        self.train_dict = train_dict
        self.holdout_dict = holdout_dict
        self.tokenizer = tokenizer
        self.samples_per_epoch = samples_per_epoch
        self._old_to_new = old_to_new

        # Rules present in *both* splits – we only sample from these.
        self.rules: List[str] = list(train_dict.keys())

        # Pre-encode static fragments and rule variants once; optionally remap ids
        self.pretokenize_ttt_fragments()
        # For index search, keep bare "Violation:" ids
        self._violation_ids: torch.Tensor = self._violation_prompt_ids
        # Build simple per-rule per-pool orders and cursors.
        self._train_state = TTTDatasetBase._init_sampler_state(self.train_dict)
        self._holdout_state = TTTDatasetBase._init_sampler_state(self.holdout_dict)
        # Shuffle training lists in-place once initially (holdout remains deterministic)
        TTTDatasetBase._shuffle_sampler_state_inplace(self.train_dict)


    # ------------------------------------------------------------------
    # Dataset interface
    # ------------------------------------------------------------------
    def __len__(self) -> int:  # noqa: D401
        return self.samples_per_epoch

    def __iter__(self):
        """Yield a *finite* number of samples, each worker getting a distinct slice."""
        for _ in range(self.samples_per_epoch):
            rule = random.choice(self.rules)

            # sample support example
            _, comment_train, label_train = TTTDatasetBase._sample_from_state(
                self.train_dict,
                self._train_state,
                rule,
                "positives" if random.random() < 0.5 else "negatives",
                reshuffle_on_cycle=True,
            )
            # sample test example
            idx, comment_test, label_test = TTTDatasetBase._sample_from_state(
                self.holdout_dict,
                self._holdout_state,
                rule,
                "positives" if random.random() < 0.5 else "negatives",
                reshuffle_on_cycle=False,
            )
            # comment_train and comment_test are already tokenized and remapped

            # Randomly choose a pre-tokenised rule variant line for this rule
            rule_variant_ids = random.choice(self._rule_variants_ids[rule])
            comment_train = comment_train if isinstance(comment_train, torch.Tensor) else torch.tensor(comment_train)
            comment_test = comment_test if isinstance(comment_test, torch.Tensor) else torch.tensor(comment_test)
            
            # Assemble the full prompt from pre-tokenised pieces
            pieces = [
                self._header_ids, # you are given a comment on reddit. Your task is to classify if it violates the given rule.
                rule_variant_ids, # Rule: <rule text>
                self._comment_prefix_ids, # Comment: 
                comment_train, # actual comment for training
                self._newline_ids, # \n
                self._violation_yes_line_ids if label_train == 1 else self._violation_no_line_ids, # Violation: Yes or No
                self._comment_prefix_ids, # Comment: 
                comment_test, # actual comment for testing
                self._newline_ids, # \n 
                self._violation_prompt_ids, # Violation:
            ]
            input_ids = torch.cat(pieces, dim=0)

            # Compute end indices for both occurrences of "Violation:"
            vi_index = self.compute_two_violation_end_indices(
                rule_variant_ids=rule_variant_ids,
                support_ids=comment_train,
                total_length=input_ids.numel(),
            )

            labels = torch.tensor([label_train, label_test], dtype=torch.long)
            # (rule, label, idx) is used to track the test example in case of ensemble prediction
            yield (rule, label_test, idx), input_ids.unsqueeze(0), torch.tensor(vi_index), labels



def load_grouped_data(
    data_dir: str = "Data/grouped",
    load_in_token: bool = True,
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
    
    if load_in_token:
        train_path = os.path.join(data_dir, "train_grouped_token_ids.pkl")
        holdout_path = os.path.join(data_dir, "holdout_grouped_token_ids.pkl")
    else:
        train_path = os.path.join(data_dir, "train_grouped.pkl")
        holdout_path = os.path.join(data_dir, "holdout_grouped.pkl")
    
    # Load train data
    with open(train_path, 'rb') as f:
        train_data = pickle.load(f)
    
    # Load holdout data
    with open(holdout_path, 'rb') as f:
        holdout_data = pickle.load(f)
    
    return train_data, holdout_data

