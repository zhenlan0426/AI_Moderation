#!/usr/bin/env python3
"""
Script to generate grouped training data from Data1 and Data2/Data3 datasets.

This script:
1. Loads Data1 and splits it into train/holdout
2. Groups Data1 examples by rule using group_examples_by_rule
3. Loads Data2/Data3 using load_data2_data3_for_ttt  
4. Combines both datasets maintaining shared negative structure
5. Saves the result to disk in a memory-efficient format

Output format: {rule_text: {"positives": [...], "negatives": shared_list}}
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
    Uses pre-split positive/negative files for efficiency.
    
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
    
    print(f"Data2 - Positives: {len(data2_pos)}, Negatives: {len(data2_neg)}")
    print(f"Data3 - Positives: {len(data3_pos)}, Negatives: {len(data3_neg)}")
    
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


def load_data2_data3_for_ttt(
    threshold: float = 0.5,
    train_split: float = 0.7, 
    random_seed: int = 42,
    data_dir: str = "Data"
) -> tuple[Dict[str, Dict[str, List[str]]], Dict[str, Dict[str, List[str]]]]:
    """
    Convenience function to load Data2 and Data3 split datasets using default paths.
    
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

def load_and_split_data1(
    data1_path: str = "Data/Data1/train.csv",
    train_split: float = 0.7,
    random_seed: int = 42
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load Data1 and split into train/holdout sets.
    
    Parameters
    ----------
    data1_path : str
        Path to Data1 train.csv file
    train_split : float
        Fraction for training set
    random_seed : int
        Random seed for reproducible splits
        
    Returns
    -------
    tuple[pd.DataFrame, pd.DataFrame]
        (train_df, holdout_df)
    """
    print(f"Loading Data1 from {data1_path}...")
    df = pd.read_csv(data1_path)
    print(f"Data1 total entries: {len(df)}")
    
    # Shuffle and split
    df_shuffled = df.sample(frac=1, random_state=random_seed).reset_index(drop=True)
    split_idx = int(len(df_shuffled) * train_split)
    
    train_df = df_shuffled[:split_idx]
    holdout_df = df_shuffled[split_idx:]
    
    print(f"Data1 split - Train: {len(train_df)}, Holdout: {len(holdout_df)}")
    return train_df, holdout_df


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
    threshold = 0.5
    random_seed = 42
    output_dir = "Data/grouped"
    
    # Step 1: Load and split Data1
    print("\n1. Loading and splitting Data1...")
    data1_train, data1_holdout = load_and_split_data1(
        data1_path=data1_path,
        train_split=train_split,
        random_seed=random_seed
    )
    
    # Step 2: Group Data1 examples by rule
    print("\n2. Grouping Data1 examples by rule...")
    data1_train_grouped = group_examples_by_rule(data1_train, include_body=True)
    data1_holdout_grouped = group_examples_by_rule(data1_holdout, include_body=True)
    
    print(f"Data1 train rules: {len(data1_train_grouped)}")
    print(f"Data1 holdout rules: {len(data1_holdout_grouped)}")
    
    # Step 3: Load Data2/3 grouped data
    print("\n3. Loading Data2/3 grouped data...")
    data23_train_grouped, data23_holdout_grouped = load_data2_data3_for_ttt(
        threshold=threshold,
        train_split=train_split,
        random_seed=random_seed
    )
    
    # Step 4: Combine datasets
    print("\n4. Combining datasets...")
    combined_train = combine_data1_and_data23(data1_train_grouped, data23_train_grouped)
    combined_holdout = combine_data1_and_data23(data1_holdout_grouped, data23_holdout_grouped)
    
    # Step 5: Save to disk
    print("\n5. Saving to disk...")
    save_grouped_data(combined_train, combined_holdout, output_dir)
    
    print("\n=== Generation Complete ===")
    print(f"Train rules: {len(combined_train)}")
    print(f"Holdout rules: {len(combined_holdout)}")
    print(f"Output directory: {output_dir}")


if __name__ == "__main__":
    main()