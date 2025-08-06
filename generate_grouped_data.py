#!/usr/bin/env python3
"""
Script to generate grouped training data from Data1 and Data2/Data3 datasets.

This script uses a hybrid approach:

For Data1:
1. Loads entire Data1 dataset  
2. Groups Data1 examples by rule using group_examples_by_rule
3. Splits each rule's examples into train/holdout to prevent overlap

For Data2/Data3:
1. Loads datasets and removes duplicates on comment_text
2. Splits into train/holdout first (split-first approach)
3. Groups by rules using threshold-based classification
4. Maintains shared negative structure for efficiency

5. Combines both datasets 
6. Saves the result to disk in a memory-efficient format

Output format: {rule_text: {"positives": [...], "negatives": [...]}}

This hybrid approach prevents overlap while being efficient for different dataset structures.
"""

import os
import pickle
import pandas as pd
from typing import Dict, List

# ---------------------------------------------------------------------------
# Data1: dataset processing
# ---------------------------------------------------------------------------

def group_examples_by_rule(df, include_body=False) -> Dict[str, Dict[str, List[str]]]:
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
        
        result[rule] = {"positives": pos_examples, "negatives": neg_examples}
    return result

# ---------------------------------------------------------------------------
# Data2 & Data3: Toxicity classification dataset processing
# ---------------------------------------------------------------------------

# Define rule texts for each category (shared configuration)
RULE_DEFINITIONS = {
    'severe_toxicity': 'No comments that are severely toxic, highly offensive language',
    'obscene': 'No comments that contain obscene language',
    'threat': 'No comments that contain threats of violence or harm against individuals or groups',
    'insult': 'No comments that contain personal attacks, insults, or derogatory language directed at individuals',
    'identity_hate': 'No comments that contain hate speech targeting specific identity groups (race, religion, gender, etc.)',
    'identity_attack': 'No comments that contain attacks targeting specific identity groups (race, religion, gender, etc.)', 
    'sexual_explicit': 'No comments that contain sexually explicit content'
}

# Dataset configurations
DATA2_COLUMNS = ['severe_toxicity', 'obscene', 'threat', 'insult', 'identity_hate']
DATA3_COLUMNS = ['severe_toxicity', 'obscene', 'threat', 'insult', 'identity_attack', 'sexual_explicit']


def _create_rule_dict_from_full_data(
    data2_pos_df: pd.DataFrame,
    data2_neg_df: pd.DataFrame, 
    data3_pos_df: pd.DataFrame,
    data3_neg_df: pd.DataFrame,
    threshold: float
) -> Dict[str, Dict[str, List[str]]]:
    """
    Create rule dictionary using complete datasets (before splitting).
    This allows for proper splitting by rule to avoid overlap.
    
    Parameters
    ----------
    data2_pos_df : pd.DataFrame
        Data2 complete positive examples
    data2_neg_df : pd.DataFrame
        Data2 complete negative examples
    data3_pos_df : pd.DataFrame  
        Data3 complete positive examples
    data3_neg_df : pd.DataFrame
        Data3 complete negative examples
    threshold : float
        Threshold for positive classification
        
    Returns
    -------
    Dict[str, Dict[str, List[str]]]
        Dictionary with format {rule_text: {"positives": [...], "negatives": [...]}}
    """
    result = {}
    
    # Create one combined negative list from both datasets that all rules will share
    data2_negatives = data2_neg_df['comment_text'].tolist()
    data3_negatives = data3_neg_df['comment_text'].tolist()
    combined_negatives = data2_negatives + data3_negatives
    
    # Process each rule
    for rule_key, rule_text in RULE_DEFINITIONS.items():
        positives = []
        
        # Check Data2 for this rule - get positives above threshold
        if rule_key in DATA2_COLUMNS:
            mask = data2_pos_df[rule_key] >= threshold
            positives.extend(data2_pos_df[mask]['comment_text'].tolist())
        
        # Check Data3 for this rule - get positives above threshold
        if rule_key in DATA3_COLUMNS:
            mask = data3_pos_df[rule_key] >= threshold
            positives.extend(data3_pos_df[mask]['comment_text'].tolist())
        
        result[rule_text] = {
            'positives': positives,
            'negatives': combined_negatives
        }
    
    return result


def _create_rule_dict_from_split_data(
    data2_pos_df: pd.DataFrame,
    data2_neg_df: pd.DataFrame, 
    data3_pos_df: pd.DataFrame,
    data3_neg_df: pd.DataFrame,
    threshold: float
) -> Dict[str, Dict[str, List[str]]]:
    """
    Create rule dictionary using pre-split positive/negative dataframes.
    Much simpler since data is already separated.
    
    DEPRECATED: This function can cause overlap between train/holdout sets.
    Use _create_rule_dict_from_full_data + split_grouped_examples_by_rule instead.
    
    Parameters
    ----------
    data2_pos_df : pd.DataFrame
        Data2 positive examples subset (train or holdout)
    data2_neg_df : pd.DataFrame
        Data2 negative examples subset (train or holdout)
    data3_pos_df : pd.DataFrame  
        Data3 positive examples subset (train or holdout)
    data3_neg_df : pd.DataFrame
        Data3 negative examples subset (train or holdout)
    threshold : float
        Threshold for positive classification
        
    Returns
    -------
    Dict[str, Dict[str, List[str]]]
        Dictionary with format {rule_text: {"positives": [...], "negatives": [...]}}
        Note: All rules share the same negatives list for memory efficiency.
    """
    result = {}
    
    # Create one combined negative list from both datasets that all rules will share
    data2_negatives = data2_neg_df['comment_text'].tolist()
    data3_negatives = data3_neg_df['comment_text'].tolist()
    combined_negatives = data2_negatives + data3_negatives
    
    # Process each rule
    for rule_key, rule_text in RULE_DEFINITIONS.items():
        positives = []
        
        # Check Data2 for this rule - get positives above threshold
        if rule_key in DATA2_COLUMNS:
            mask = data2_pos_df[rule_key] >= threshold
            positives.extend(data2_pos_df[mask]['comment_text'].tolist())
        
        # Check Data3 for this rule - get positives above threshold
        if rule_key in DATA3_COLUMNS:
            mask = data3_pos_df[rule_key] >= threshold
            positives.extend(data3_pos_df[mask]['comment_text'].tolist())
        
        result[rule_text] = {
            'positives': positives,
            'negatives': combined_negatives  # All rules point to the same negative list
        }
    
    return result


def _print_summary_stats(train_dict: Dict, holdout_dict: Dict) -> None:
    """Print summary statistics for train and holdout sets."""
    print("\nTrain set summary:")
    for rule, data in train_dict.items():
        print(f"  {rule[:50]}...: {len(data['positives'])} positives, {len(data['negatives'])} negatives")
        
    print("\nHoldout set summary:")  
    for rule, data in holdout_dict.items():
        print(f"  {rule[:50]}...: {len(data['positives'])} positives, {len(data['negatives'])} negatives")


def process_data2_data3_splits(
    data2_pos_path: str,
    data2_neg_path: str, 
    data3_pos_path: str,
    data3_neg_path: str,
    threshold: float = 0.5,
    train_split: float = 0.7,
    random_seed: int = 42
) -> tuple[Dict[str, Dict[str, List[str]]], Dict[str, Dict[str, List[str]]]]:
    """
    Process Data2 and Data3 split datasets into rule-based format for TTT training.
    Uses split-first approach with deduplication on comment_text to prevent overlap.
    
    Parameters
    ----------
    data2_pos_path : str
        Path to data2_positive.csv file
    data2_neg_path : str  
        Path to data2_negative.csv file
    data3_pos_path : str
        Path to data3_positive.csv file
    data3_neg_path : str
        Path to data3_negative.csv file
    threshold : float, optional
        Threshold above which a comment will be considered positive for a rule.
        Defaults to 0.5.
    train_split : float, optional
        Fraction of data to use for training (remainder goes to holdout).
        Defaults to 0.7.
    random_seed : int, optional
        Random seed for reproducible splits. Defaults to 42.
        
    Returns
    -------
    tuple[Dict[str, Dict[str, List[str]]], Dict[str, Dict[str, List[str]]]]
        A tuple containing (train_dict, holdout_dict) where each dict has the format:
        {rule_text: {"positives": [...], "negatives": [...]}}
    """
    
    print("Loading datasets...")
    
    # Load datasets (already split into positive/negative)
    data2_pos = pd.read_csv(data2_pos_path)
    data2_neg = pd.read_csv(data2_neg_path)
    data3_pos = pd.read_csv(data3_pos_path)
    data3_neg = pd.read_csv(data3_neg_path)
    
    print(f"Data2 - Positives: {len(data2_pos)} (before dedup), Negatives: {len(data2_neg)} (before dedup)")
    print(f"Data3 - Positives: {len(data3_pos)} (before dedup), Negatives: {len(data3_neg)} (before dedup)")
    
    # Remove duplicates within each dataset
    print("Removing within-dataset duplicates on comment_text...")
    data2_pos = data2_pos.drop_duplicates(subset=['comment_text']).reset_index(drop=True)
    data2_neg = data2_neg.drop_duplicates(subset=['comment_text']).reset_index(drop=True)
    data3_pos = data3_pos.drop_duplicates(subset=['comment_text']).reset_index(drop=True)
    data3_neg = data3_neg.drop_duplicates(subset=['comment_text']).reset_index(drop=True)
    
    # Remove cross-dataset duplicates (Data3 takes precedence over Data2)
    print("Removing cross-dataset duplicates (Data3 takes precedence)...")
    data2_pos_comments = set(data2_pos['comment_text'])
    data2_neg_comments = set(data2_neg['comment_text'])
    
    # Remove Data2 comments that appear in Data3
    data3_pos_comments = set(data3_pos['comment_text'])
    data3_neg_comments = set(data3_neg['comment_text'])
    
    data2_pos = data2_pos[~data2_pos['comment_text'].isin(data3_pos_comments | data3_neg_comments)].reset_index(drop=True)
    data2_neg = data2_neg[~data2_neg['comment_text'].isin(data3_pos_comments | data3_neg_comments)].reset_index(drop=True)
    
    print(f"Data2 - Positives: {len(data2_pos)} (after dedup), Negatives: {len(data2_neg)} (after dedup)")
    print(f"Data3 - Positives: {len(data3_pos)} (after dedup), Negatives: {len(data3_neg)} (after dedup)")
    
    # Split positive datasets into train/holdout
    data2_pos_shuffled = data2_pos.sample(frac=1, random_state=random_seed).reset_index(drop=True)
    data3_pos_shuffled = data3_pos.sample(frac=1, random_state=random_seed).reset_index(drop=True)
    
    data2_pos_split_idx = int(len(data2_pos_shuffled) * train_split)
    data3_pos_split_idx = int(len(data3_pos_shuffled) * train_split)
    
    data2_pos_train = data2_pos_shuffled[:data2_pos_split_idx]
    data2_pos_holdout = data2_pos_shuffled[data2_pos_split_idx:]
    data3_pos_train = data3_pos_shuffled[:data3_pos_split_idx]
    data3_pos_holdout = data3_pos_shuffled[data3_pos_split_idx:]
    
    # Split negative datasets into train/holdout
    data2_neg_shuffled = data2_neg.sample(frac=1, random_state=random_seed).reset_index(drop=True)
    data3_neg_shuffled = data3_neg.sample(frac=1, random_state=random_seed).reset_index(drop=True)
    
    data2_neg_split_idx = int(len(data2_neg_shuffled) * train_split)
    data3_neg_split_idx = int(len(data3_neg_shuffled) * train_split)
    
    data2_neg_train = data2_neg_shuffled[:data2_neg_split_idx]
    data2_neg_holdout = data2_neg_shuffled[data2_neg_split_idx:]
    data3_neg_train = data3_neg_shuffled[:data3_neg_split_idx]
    data3_neg_holdout = data3_neg_shuffled[data3_neg_split_idx:]
    
    print(f"Train split - Data2 pos: {len(data2_pos_train)}, Data3 pos: {len(data3_pos_train)}")
    print(f"Train split - Data2 neg: {len(data2_neg_train)}, Data3 neg: {len(data3_neg_train)}")
    
    # Create rule dictionaries using pre-split data
    print("Creating train set rules...")
    train_dict = _create_rule_dict_from_split_data(
        data2_pos_train, data2_neg_train, data3_pos_train, data3_neg_train, threshold
    )
    
    print("Creating holdout set rules...")
    holdout_dict = _create_rule_dict_from_split_data(
        data2_pos_holdout, data2_neg_holdout, data3_pos_holdout, data3_neg_holdout, threshold
    )
    
    # Print summary statistics
    _print_summary_stats(train_dict, holdout_dict)
    
    return train_dict, holdout_dict


def process_data2_data3_group_first(
    data2_pos_path: str,
    data2_neg_path: str, 
    data3_pos_path: str,
    data3_neg_path: str,
    threshold: float = 0.5,
    train_split: float = 0.7,
    random_seed: int = 42
) -> tuple[Dict[str, Dict[str, List[str]]], Dict[str, Dict[str, List[str]]]]:
    """
    Process Data2 and Data3 using group-first approach to prevent overlap.
    
    This function:
    1. Loads complete datasets
    2. Groups examples by rule first
    3. Splits each rule's examples into train/holdout
    
    This prevents overlap between train and holdout sets for the same rule.
    
    Parameters
    ----------
    data2_pos_path : str
        Path to data2_positive.csv file
    data2_neg_path : str  
        Path to data2_negative.csv file
    data3_pos_path : str
        Path to data3_positive.csv file
    data3_neg_path : str
        Path to data3_negative.csv file
    threshold : float, optional
        Threshold above which a comment will be considered positive for a rule.
        Defaults to 0.5.
    train_split : float, optional
        Fraction of data to use for training (remainder goes to holdout).
        Defaults to 0.7.
    random_seed : int, optional
        Random seed for reproducible splits. Defaults to 42.
        
    Returns
    -------
    tuple[Dict[str, Dict[str, List[str]]], Dict[str, Dict[str, List[str]]]]
        A tuple containing (train_dict, holdout_dict) where each dict has the format:
        {rule_text: {"positives": [...], "negatives": [...]}}
    """
    
    print("Loading complete datasets...")
    
    # Load complete datasets
    data2_pos = pd.read_csv(data2_pos_path)
    data2_neg = pd.read_csv(data2_neg_path)
    data3_pos = pd.read_csv(data3_pos_path)
    data3_neg = pd.read_csv(data3_neg_path)
    
    print(f"Data2 - Positives: {len(data2_pos)}, Negatives: {len(data2_neg)}")
    print(f"Data3 - Positives: {len(data3_pos)}, Negatives: {len(data3_neg)}")
    
    # Group by rules first (using complete datasets)
    print("Creating rule groups from complete datasets...")
    complete_grouped = _create_rule_dict_from_full_data(
        data2_pos, data2_neg, data3_pos, data3_neg, threshold
    )
    
    print(f"Created {len(complete_grouped)} rules from complete datasets")
    
    # Split each rule's examples into train/holdout
    print("Splitting grouped examples by rule...")
    train_dict, holdout_dict = split_grouped_examples_by_rule(
        complete_grouped,
        train_split=train_split,
        random_seed=random_seed
    )
    
    # Print summary statistics
    _print_summary_stats(train_dict, holdout_dict)
    
    return train_dict, holdout_dict


def load_data2_data3_for_ttt(
    threshold: float = 0.5,
    train_split: float = 0.7, 
    random_seed: int = 42,
    data_dir: str = "Data"
) -> tuple[Dict[str, Dict[str, List[str]]], Dict[str, Dict[str, List[str]]]]:
    """
    Convenience function to load Data2 and Data3 split datasets using default paths.
    Uses split-first approach with deduplication on comment_text to prevent overlap.
    
    Parameters
    ----------
    threshold : float, optional
        Threshold above which a comment will be considered positive for a rule.
        Defaults to 0.5.
    train_split : float, optional
        Fraction of data to use for training. Defaults to 0.7.
    random_seed : int, optional
        Random seed for reproducible splits. Defaults to 42.
    data_dir : str, optional
        Base directory containing Data2 and Data3 folders. Defaults to "Data".
        
    Returns
    -------
    tuple[Dict[str, Dict[str, List[str]]], Dict[str, Dict[str, List[str]]]]
        A tuple containing (train_dict, holdout_dict)
        
    Examples
    --------
    >>> train_data, holdout_data = load_data2_data3_for_ttt(threshold=0.6)
    >>> # Use with existing TTT pipeline
    >>> from utility import TTTDataset, build_dataloader
    >>> dataset = TTTDataset(some_df, train_data, tokenizer)
    """
    import os
    
    # Construct file paths
    data2_pos_path = os.path.join(data_dir, "Data2", "split", "data2_positive.csv")
    data2_neg_path = os.path.join(data_dir, "Data2", "split", "data2_negative.csv") 
    data3_pos_path = os.path.join(data_dir, "Data3", "split", "data3_positive.csv")
    data3_neg_path = os.path.join(data_dir, "Data3", "split", "data3_negative.csv")
    
    return process_data2_data3_splits(
        data2_pos_path=data2_pos_path,
        data2_neg_path=data2_neg_path,
        data3_pos_path=data3_pos_path, 
        data3_neg_path=data3_neg_path,
        threshold=threshold,
        train_split=train_split,
        random_seed=random_seed
    )

def verify_shared_negatives(grouped_data: Dict[str, Dict[str, List[str]]]) -> Dict[str, bool]:
    """
    Analyze the negatives structure in grouped data.
    
    Since the new structure combines rule-specific Data1 negatives with shared Data2/3 negatives,
    this function provides information about the sharing pattern.
    
    Parameters
    ----------
    grouped_data : dict
        Grouped data in format {rule_text: {"positives": [...], "negatives": [...]}}
        
    Returns
    -------
    dict
        Dictionary with analysis results:
        - 'fully_shared': bool - True if all rules have identical negatives lists
        - 'partially_shared': bool - True if there's some overlap in negatives
        - 'unique_counts': dict - Number of unique negatives per rule
        - 'total_unique': int - Total unique negatives across all rules
        
    Examples
    --------
    >>> train_data, _ = load_grouped_data()
    >>> analysis = verify_shared_negatives(train_data)
    >>> print(f"Fully shared: {analysis['fully_shared']}")
    >>> print(f"Partially shared: {analysis['partially_shared']}")
    """
    if not grouped_data:
        return {'fully_shared': True, 'partially_shared': True, 'unique_counts': {}, 'total_unique': 0}
    
    # Get negatives from all rules
    all_negatives = {}
    for rule, data in grouped_data.items():
        all_negatives[rule] = set(data['negatives'])
    
    # Check if all rules have identical negatives
    first_negatives = next(iter(all_negatives.values()))
    fully_shared = all(negatives == first_negatives for negatives in all_negatives.values())
    
    # Check for partial sharing (common negatives across rules)
    if len(all_negatives) > 1:
        intersection = set.intersection(*all_negatives.values())
        partially_shared = len(intersection) > 0
    else:
        partially_shared = True
    
    # Count unique negatives per rule
    unique_counts = {rule: len(negatives) for rule, negatives in all_negatives.items()}
    
    # Total unique negatives across all rules
    all_unique = set.union(*all_negatives.values()) if all_negatives else set()
    total_unique = len(all_unique)
    
    return {
        'fully_shared': fully_shared,
        'partially_shared': partially_shared,
        'unique_counts': unique_counts,
        'total_unique': total_unique
    }


def print_grouped_data_stats(
    train_data: Dict[str, Dict[str, List[str]]],
    holdout_data: Dict[str, Dict[str, List[str]]]
) -> None:
    """
    Print detailed statistics about grouped training data.
    
    Parameters
    ----------
    train_data : dict
        Training grouped data
    holdout_data : dict
        Holdout grouped data
    """
    print("\n=== Grouped Data Statistics ===")
    
    print(f"\nTrain Data:")
    print(f"  Rules: {len(train_data)}")
    if train_data:
        total_positives = sum(len(data['positives']) for data in train_data.values())
        print(f"  Total positives: {total_positives}")
        
        # Analyze negatives structure
        neg_analysis = verify_shared_negatives(train_data)
        print(f"  Total unique negatives: {neg_analysis['total_unique']}")
        print(f"  Negatives fully shared: {neg_analysis['fully_shared']}")
        print(f"  Negatives partially shared: {neg_analysis['partially_shared']}")
    
    print(f"\nHoldout Data:")
    print(f"  Rules: {len(holdout_data)}")
    if holdout_data:
        total_positives = sum(len(data['positives']) for data in holdout_data.values())
        print(f"  Total positives: {total_positives}")
        
        # Analyze negatives structure
        neg_analysis = verify_shared_negatives(holdout_data)
        print(f"  Total unique negatives: {neg_analysis['total_unique']}")
        print(f"  Negatives fully shared: {neg_analysis['fully_shared']}")
        print(f"  Negatives partially shared: {neg_analysis['partially_shared']}")
    
    print(f"\nDetailed breakdown:")
    print("Train rules:")
    for rule, data in train_data.items():
        print(f"  {rule[:50]}...: {len(data['positives'])} positives, {len(data['negatives'])} negatives")
    
    print("Holdout rules:")
    for rule, data in holdout_data.items():
        print(f"  {rule[:50]}...: {len(data['positives'])} positives, {len(data['negatives'])} negatives")

def load_data1(
    data1_path: str = "Data/Data1/train.csv"
) -> pd.DataFrame:
    """
    Load Data1 dataset without splitting.
    
    Parameters
    ----------
    data1_path : str
        Path to Data1 train.csv file
        
    Returns
    -------
    pd.DataFrame
        Complete Data1 dataset
    """
    print(f"Loading Data1 from {data1_path}...")
    df = pd.read_csv(data1_path)
    print(f"Data1 total entries: {len(df)}")
    return df


def split_grouped_examples_by_rule(
    grouped_data: Dict[str, Dict[str, List[str]]],
    train_split: float = 0.7,
    random_seed: int = 42
) -> tuple[Dict[str, Dict[str, List[str]]], Dict[str, Dict[str, List[str]]]]:
    """
    Split grouped examples by rule into train/holdout sets.
    
    This ensures no overlap between train and holdout - each rule's examples
    are split individually, so the same rule cannot have overlapping examples
    between train and holdout sets.
    
    Special handling for shared negatives: If all rules have identical negatives
    lists (common in Data2/3), splits negatives once and reuses the split
    across all rules to maintain consistency and avoid overlap.
    
    Parameters
    ----------
    grouped_data : dict
        Grouped data in format {rule_text: {"positives": [...], "negatives": [...]}}
    train_split : float
        Fraction for training set
    random_seed : int
        Random seed for reproducible splits
        
    Returns
    -------
    tuple[Dict[str, Dict[str, List[str]]], Dict[str, Dict[str, List[str]]]]
        (train_grouped, holdout_grouped)
    """
    import random
    random.seed(random_seed)
    
    # Check if all rules have identical negatives (shared negatives case)
    # Filter out NaN values and ensure all items are strings
    def clean_negatives(neg_list):
        return set(str(item) for item in neg_list if pd.notna(item) and item != '')
    
    negatives_lists = [clean_negatives(data['negatives']) for data in grouped_data.values()]
    has_shared_negatives = len(set(frozenset(neg_set) for neg_set in negatives_lists)) == 1
    
    shared_train_negatives = None
    shared_holdout_negatives = None
    
    if has_shared_negatives and negatives_lists:
        print("Detected shared negatives - splitting once for all rules")
        # Split negatives once for all rules
        negatives = list(negatives_lists[0])  # All are identical, so use the first
        random.shuffle(negatives)
        
        neg_split_idx = int(len(negatives) * train_split)
        shared_train_negatives = negatives[:neg_split_idx]
        shared_holdout_negatives = negatives[neg_split_idx:]
        
        print(f"Shared negatives - Train: {len(shared_train_negatives)}, Holdout: {len(shared_holdout_negatives)}")
    
    train_grouped = {}
    holdout_grouped = {}
    
    for rule_text, data in grouped_data.items():
        positives = data['positives'].copy()
        
        # Shuffle and split positives (always done per rule)
        random.shuffle(positives)
        pos_split_idx = int(len(positives) * train_split)
        train_positives = positives[:pos_split_idx]
        holdout_positives = positives[pos_split_idx:]
        
        # Handle negatives based on whether they're shared
        if has_shared_negatives:
            # Use pre-split shared negatives
            train_negatives = shared_train_negatives
            holdout_negatives = shared_holdout_negatives
        else:
            # Split negatives individually for this rule
            negatives = data['negatives'].copy()
            random.shuffle(negatives)
            neg_split_idx = int(len(negatives) * train_split)
            train_negatives = negatives[:neg_split_idx]
            holdout_negatives = negatives[neg_split_idx:]
        
        train_grouped[rule_text] = {
            "positives": train_positives,
            "negatives": train_negatives
        }
        
        holdout_grouped[rule_text] = {
            "positives": holdout_positives,
            "negatives": holdout_negatives
        }
    
    return train_grouped, holdout_grouped


def combine_data1_and_data23(
    data1_grouped: Dict[str, Dict[str, List[str]]],
    data23_grouped: Dict[str, Dict[str, List[str]]]
) -> Dict[str, Dict[str, List[str]]]:
    """
    Combine Data1 and Data2/3 grouped examples.
    
    Since rules don't overlap between datasets, this is a simple dictionary merge.
    Data1 rules keep their rule-specific negatives.
    Data2/3 rules keep their shared negatives structure.
    
    Parameters
    ----------
    data1_grouped : dict
        Grouped examples from Data1 (negatives are rule-specific)
    data23_grouped : dict
        Grouped examples from Data2/3 (negatives are already shared)
        
    Returns
    -------
    dict
        Combined grouped examples (simple merge since no rule overlap)
    """
    print("Combining Data1 and Data2/3 grouped examples...")
    
    # Simple merge since rules don't overlap
    result = {}
    
    # Add all Data1 rules (with rule-specific negatives)
    result.update(data1_grouped)
    print(f"Added {len(data1_grouped)} rules from Data1")
    
    # Add all Data2/3 rules (with shared negatives)
    result.update(data23_grouped)
    print(f"Added {len(data23_grouped)} rules from Data2/3")
    
    print(f"Total combined rules: {len(result)}")
    
    # Print summary
    for rule, data in result.items():
        source = "Data1" if rule in data1_grouped else "Data2/3"
        print(f"  {rule[:50]}... ({source}): {len(data['positives'])} positives, {len(data['negatives'])} negatives")
    
    return result


def save_grouped_data(
    train_data: Dict[str, Dict[str, List[str]]],
    holdout_data: Dict[str, Dict[str, List[str]]],
    output_dir: str = "Data/grouped"
) -> None:
    """
    Save grouped data to disk in a memory-efficient format.
    
    The negatives are saved separately and referenced by all rules to maintain
    the shared structure when loaded.
    
    Parameters
    ----------
    train_data : dict
        Training grouped data
    holdout_data : dict
        Holdout grouped data
    output_dir : str
        Directory to save the files
    """
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"Saving grouped data to {output_dir}...")
    
    # Save train data
    train_path = os.path.join(output_dir, "train_grouped.pkl")
    with open(train_path, 'wb') as f:
        pickle.dump(train_data, f)
    print(f"Saved train data to {train_path}")
    
    # Save holdout data
    holdout_path = os.path.join(output_dir, "holdout_grouped.pkl")
    with open(holdout_path, 'wb') as f:
        pickle.dump(holdout_data, f)
    print(f"Saved holdout data to {holdout_path}")
    
    # Save metadata
    metadata = {
        'train_rules': list(train_data.keys()),
        'holdout_rules': list(holdout_data.keys()),
        'train_negatives_count': len(next(iter(train_data.values()))['negatives']),
        'holdout_negatives_count': len(next(iter(holdout_data.values()))['negatives']),
        'description': 'Grouped training data with shared negatives structure'
    }
    
    metadata_path = os.path.join(output_dir, "metadata.pkl")
    with open(metadata_path, 'wb') as f:
        pickle.dump(metadata, f)
    print(f"Saved metadata to {metadata_path}")


def main():
    """Main function to generate and save grouped data."""
    print("=== Generating Grouped Training Data ===")
    
    # Configuration
    data1_path = "Data/Data1/train.csv"
    train_split = 0.7
    threshold = 0.33
    random_seed = 42
    output_dir = "Data/grouped"
    
    # Step 1: Load entire Data1 dataset
    print("\n1. Loading Data1...")
    data1_full = load_data1(data1_path=data1_path)
    
    # Step 2: Group Data1 examples by rule (entire dataset)
    print("\n2. Grouping Data1 examples by rule...")
    data1_full_grouped = group_examples_by_rule(data1_full, include_body=True)
    print(f"Data1 total rules: {len(data1_full_grouped)}")
    
    # Step 3: Split each rule's examples into train/holdout
    print("\n3. Splitting Data1 grouped examples into train/holdout...")
    data1_train_grouped, data1_holdout_grouped = split_grouped_examples_by_rule(
        data1_full_grouped,
        train_split=train_split,
        random_seed=random_seed
    )
    
    print(f"Data1 train rules: {len(data1_train_grouped)}")
    print(f"Data1 holdout rules: {len(data1_holdout_grouped)}")
    
    # Step 4: Load Data2/3 grouped data
    print("\n4. Loading Data2/3 grouped data...")
    data23_train_grouped, data23_holdout_grouped = load_data2_data3_for_ttt(
        threshold=threshold,
        train_split=train_split,
        random_seed=random_seed
    )
    
    # Step 5: Combine datasets
    print("\n5. Combining datasets...")
    combined_train = combine_data1_and_data23(data1_train_grouped, data23_train_grouped)
    combined_holdout = combine_data1_and_data23(data1_holdout_grouped, data23_holdout_grouped)
    
    # Step 6: Save to disk
    print("\n6. Saving to disk...")
    save_grouped_data(combined_train, combined_holdout, output_dir)
    
    print("\n=== Generation Complete ===")
    print(f"Train rules: {len(combined_train)}")
    print(f"Holdout rules: {len(combined_holdout)}")
    print(f"Output directory: {output_dir}")


if __name__ == "__main__":
    main()