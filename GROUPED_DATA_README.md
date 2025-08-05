# Grouped Training Data Pipeline

This document describes the new pipeline for generating and using grouped training data that combines Data1 and Data2/3 datasets while maintaining memory-efficient shared negatives.

## Overview

The pipeline creates grouped training data in the format:
```python
{rule_text: {"positives": [...], "negatives": shared_list}}
```

Key features:
- **Memory efficient**: All rules share the same negatives list (not copied)
- **Combined datasets**: Merges Data1 and Data2/3 examples
- **Train/holdout splits**: Maintains separate splits for both datasets
- **Easy integration**: Works with existing TTT pipeline components

## Files

- `generate_grouped_data.py` - Main script to generate grouped data
- `utility.py` - Contains utility functions for loading and working with grouped data
- `example_usage.py` - Example showing how to use the grouped data
- `Data/grouped/` - Output directory for generated files

## Usage

### 1. Generate Grouped Data

Run the generation script to create the grouped data files:

```bash
python generate_grouped_data.py
```

This will:
1. Load Data1 and split into train/holdout (70/30 by default)
2. Group Data1 examples by rule using `group_examples_by_rule`
3. Load Data2/3 and split using `load_data2_data3_for_ttt`
4. Combine both datasets maintaining shared negatives
5. Save results to `Data/grouped/`

Output files:
- `train_grouped.pkl` - Training data
- `holdout_grouped.pkl` - Holdout data  
- `metadata.pkl` - Dataset statistics and info

### 2. Load Grouped Data

Use the utility functions to load the data:

```python
from utility import load_grouped_data, print_grouped_data_stats

# Load the data
train_data, holdout_data = load_grouped_data()

# Print statistics
print_grouped_data_stats(train_data, holdout_data)
```

### 3. Use with TTT Pipeline

The grouped data works directly with existing TTT components:

```python
from utility import TTTDataset, build_dataloader, load_grouped_data

# Load grouped data
train_data, holdout_data = load_grouped_data()

# Use with TTT dataset (requires DataFrame with required columns)
dataset = TTTDataset(your_df, train_data, tokenizer)
dataloader = build_dataloader(dataset, batch_size=8)
```

## Configuration

You can customize the generation process by modifying these parameters in `generate_grouped_data.py`:

```python
# Data paths
data1_path = "Data/Data1/train.csv"
output_dir = "Data/grouped"

# Split parameters
train_split = 0.7  # 70% train, 30% holdout
threshold = 0.5    # Threshold for Data2/3 positive classification
random_seed = 42   # For reproducible splits
```

## Negatives Structure

Since Data1 and Data2/3 have completely different rules with no overlap, the combination is a simple merge:

- **Data1 rules**: Each rule has its own specific negative examples
- **Data2/3 rules**: All rules share the same negatives list (memory efficient)
- **Final structure**: Simple dictionary merge - no rule conflicts

You can analyze the structure with:

```python
from utility import verify_shared_negatives

train_data, _ = load_grouped_data()
analysis = verify_shared_negatives(train_data)
print(f"Fully shared: {analysis['fully_shared']}")      # Should be False (Data1 rules have specific negatives)
print(f"Partially shared: {analysis['partially_shared']}")  # Should be True (Data2/3 rules share negatives)
print(f"Total unique negatives: {analysis['total_unique']}")
```

## Data Structure

The generated data has this structure:

```
Data/grouped/
├── train_grouped.pkl      # Training data
├── holdout_grouped.pkl    # Holdout data
└── metadata.pkl          # Statistics and metadata
```

Each pickle file contains:
```python
{
    # Data1 rules (rule-specific negatives)
    "Data1 Rule A": {
        "positives": ["data1_pos1", "data1_pos2", ...],
        "negatives": ["data1_ruleA_neg1", "data1_ruleA_neg2", ...]  # Unique to this rule
    },
    "Data1 Rule B": {
        "positives": ["data1_pos3", "data1_pos4", ...], 
        "negatives": ["data1_ruleB_neg1", "data1_ruleB_neg2", ...]  # Unique to this rule
    },
    
    # Data2/3 rules (shared negatives)
    "Data2/3 Rule X": {
        "positives": ["data23_pos1", "data23_pos2", ...],
        "negatives": shared_negatives_list  # Same object for all Data2/3 rules
    },
    "Data2/3 Rule Y": {
        "positives": ["data23_pos3", "data23_pos4", ...], 
        "negatives": shared_negatives_list  # Same reference as Rule X
    },
    ...
}
```

Note: 
- **Data1 rules**: Each has unique rule-specific negatives
- **Data2/3 rules**: All share the same negatives list (memory efficient)
- **No rule overlap**: Simple dictionary merge, no conflicts

## Integration with Existing Code

This pipeline is designed to work seamlessly with your existing TTT training code. Simply replace:

```python
# Old approach
train_data, holdout_data = load_data2_data3_for_ttt()
```

With:

```python
# New approach  
train_data, holdout_data = load_grouped_data()
```

The data format is identical, so no other changes are needed.