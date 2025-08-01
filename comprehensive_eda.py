#!/usr/bin/env python3
"""
Comprehensive Exploratory Data Analysis (EDA)
AI Content Moderation Dataset

This script performs a thorough analysis of the training and test datasets for content moderation.
"""

# Import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
from collections import Counter
import re

# Optional imports with graceful fallback
try:
    from wordcloud import WordCloud
    WORDCLOUD_AVAILABLE = True
except ImportError:
    print("⚠️ WordCloud not available. Word cloud visualizations will be skipped.")
    WORDCLOUD_AVAILABLE = False

try:
    import plotly.express as px
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    PLOTLY_AVAILABLE = True
except ImportError:
    print("⚠️ Plotly not available. Some interactive visualizations will be skipped.")
    PLOTLY_AVAILABLE = False

# Configure display options
pd.set_option('display.max_columns', None)
pd.set_option('display.max_colwidth', 100)
warnings.filterwarnings('ignore')

# Set style for matplotlib
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

print("Libraries imported successfully!")

# =============================================================================
# 1. Data Loading and Basic Information
# =============================================================================

def load_and_inspect_data():
    """Load datasets and display basic information"""
    print("=" * 60)
    print("1. DATA LOADING AND BASIC INFORMATION")
    print("=" * 60)
    
    # Load the datasets
    train_df = pd.read_csv('Data/train.csv')
    test_df = pd.read_csv('Data/test.csv')
    sample_submission = pd.read_csv('Data/sample_submission.csv')

    print("Dataset Shapes:")
    print(f"Training data: {train_df.shape}")
    print(f"Test data: {test_df.shape}")
    print(f"Sample submission: {sample_submission.shape}")

    print("\nColumn names comparison:")
    print("Train columns:", list(train_df.columns))
    print("Test columns:", list(test_df.columns))
    print("Sample submission columns:", list(sample_submission.columns))
    
    # Display basic info about the datasets
    print("\n=== TRAINING DATA INFO ===")
    print(train_df.info())
    print("\n=== TEST DATA INFO ===")
    print(test_df.info())
    
    # First few rows of each dataset
    print("\n=== TRAINING DATA - FIRST 3 ROWS ===")
    print(train_df.head(3))

    print("\n=== TEST DATA - FIRST 3 ROWS ===")
    print(test_df.head(3))

    print("\n=== SAMPLE SUBMISSION ===")
    print(sample_submission.head())
    
    return train_df, test_df, sample_submission

# =============================================================================
# 2. Missing Values Analysis
# =============================================================================

def analyze_missing_values(df, dataset_name):
    """Analyze and visualize missing values in the dataset"""
    print(f"\n=== {dataset_name.upper()} - MISSING VALUES ===")
    missing_count = df.isnull().sum()
    missing_percent = (missing_count / len(df)) * 100
    
    missing_df = pd.DataFrame({
        'Column': missing_count.index,
        'Missing Count': missing_count.values,
        'Missing Percentage': missing_percent.values
    })
    
    missing_df = missing_df[missing_df['Missing Count'] > 0].sort_values('Missing Count', ascending=False)
    
    if len(missing_df) == 0:
        print("No missing values found!")
    else:
        print(missing_df)
        
        # Visualize missing values
        plt.figure(figsize=(12, 6))
        sns.heatmap(df.isnull(), cbar=True, yticklabels=False, cmap='viridis')
        plt.title(f'{dataset_name} - Missing Values Heatmap')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()
    
    return missing_df

def missing_values_analysis(train_df, test_df):
    """Perform missing values analysis on both datasets"""
    print("=" * 60)
    print("2. MISSING VALUES ANALYSIS")
    print("=" * 60)
    
    train_missing = analyze_missing_values(train_df, 'Training Data')
    test_missing = analyze_missing_values(test_df, 'Test Data')
    
    return train_missing, test_missing

# =============================================================================
# 3. Target Variable Analysis
# =============================================================================

def target_variable_analysis(train_df):
    """Analyze target variable distribution"""
    print("=" * 60)
    print("3. TARGET VARIABLE ANALYSIS")
    print("=" * 60)
    
    target_counts = train_df['rule_violation'].value_counts()
    target_props = train_df['rule_violation'].value_counts(normalize=True)

    print("Value Counts:")
    print(target_counts)
    print("\nProportions:")
    print(target_props)

    # Visualize target distribution
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))

    # Count plot
    sns.countplot(data=train_df, x='rule_violation', ax=axes[0])
    axes[0].set_title('Rule Violation Distribution (Counts)')
    axes[0].set_xlabel('Rule Violation (0=No, 1=Yes)')

    # Pie chart
    axes[1].pie(target_counts.values, labels=['No Violation (0)', 'Violation (1)'], 
               autopct='%1.1f%%', startangle=90)
    axes[1].set_title('Rule Violation Distribution (Percentage)')

    plt.tight_layout()
    plt.show()

    # Check for class imbalance
    imbalance_ratio = target_counts.min() / target_counts.max()
    print(f"\nClass Imbalance Ratio: {imbalance_ratio:.3f}")
    if imbalance_ratio < 0.5:
        print("⚠️  Dataset shows class imbalance - consider stratified sampling or rebalancing techniques")
    else:
        print("✅ Dataset is reasonably balanced")
    
    return target_counts, target_props, imbalance_ratio

# =============================================================================
# 4. Text Content Analysis
# =============================================================================

def get_text_stats(text):
    """Extract various text statistics from a given text"""
    if pd.isna(text):
        return {
            'char_count': 0,
            'word_count': 0,
            'sentence_count': 0,
            'avg_word_length': 0,
            'url_count': 0,
            'special_char_count': 0
        }
    
    text = str(text)
    words = text.split()
    sentences = text.split('.')
    urls = re.findall(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', text)
    special_chars = re.findall(r'[^a-zA-Z0-9\\s]', text)
    
    return {
        'char_count': len(text),
        'word_count': len(words),
        'sentence_count': len([s for s in sentences if s.strip()]),
        'avg_word_length': np.mean([len(word) for word in words]) if words else 0,
        'url_count': len(urls),
        'special_char_count': len(special_chars)
    }

def display_text_summary(stats_df, dataset_name):
    """Display summary statistics and visualizations for text data"""
    print(f"\n=== {dataset_name.upper()} - TEXT STATISTICS SUMMARY ===")
    summary = stats_df.describe()
    print(summary)
    
    # Create visualizations
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle(f'{dataset_name} - Text Statistics Distribution', fontsize=16)
    
    # Character count
    axes[0, 0].hist(stats_df['char_count'], bins=50, alpha=0.7, edgecolor='black')
    axes[0, 0].set_title('Character Count Distribution')
    axes[0, 0].set_xlabel('Character Count')
    
    # Word count
    axes[0, 1].hist(stats_df['word_count'], bins=50, alpha=0.7, edgecolor='black')
    axes[0, 1].set_title('Word Count Distribution')
    axes[0, 1].set_xlabel('Word Count')
    
    # Average word length
    axes[0, 2].hist(stats_df['avg_word_length'], bins=30, alpha=0.7, edgecolor='black')
    axes[0, 2].set_title('Average Word Length Distribution')
    axes[0, 2].set_xlabel('Average Word Length')
    
    # URL count
    axes[1, 0].hist(stats_df['url_count'], bins=20, alpha=0.7, edgecolor='black')
    axes[1, 0].set_title('URL Count Distribution')
    axes[1, 0].set_xlabel('URL Count')
    
    # Sentence count
    axes[1, 1].hist(stats_df['sentence_count'], bins=30, alpha=0.7, edgecolor='black')
    axes[1, 1].set_title('Sentence Count Distribution')
    axes[1, 1].set_xlabel('Sentence Count')
    
    # Special character count
    axes[1, 2].hist(stats_df['special_char_count'], bins=50, alpha=0.7, edgecolor='black')
    axes[1, 2].set_title('Special Character Count Distribution')
    axes[1, 2].set_xlabel('Special Character Count')
    
    plt.tight_layout()
    plt.show()

def text_content_analysis(train_df, test_df):
    """Perform comprehensive text content analysis"""
    print("=" * 60)
    print("4. TEXT CONTENT ANALYSIS")
    print("=" * 60)
    
    print("Analyzing text statistics...")

    # Training data
    train_text_stats = train_df['body'].apply(get_text_stats)
    train_stats_df = pd.DataFrame(list(train_text_stats))

    # Test data
    test_text_stats = test_df['body'].apply(get_text_stats)
    test_stats_df = pd.DataFrame(list(test_text_stats))

    print("Text statistics calculated!")
    
    display_text_summary(train_stats_df, 'Training Data')
    display_text_summary(test_stats_df, 'Test Data')
    
    return train_stats_df, test_stats_df

# =============================================================================
# 5. Text Statistics by Rule Violation (Training Data)
# =============================================================================

def text_stats_by_violation(train_df, train_stats_df):
    """Analyze text statistics by rule violation"""
    print("=" * 60)
    print("5. TEXT STATISTICS BY RULE VIOLATION")
    print("=" * 60)
    
    # Analyze text statistics by rule violation
    train_with_stats = pd.concat([train_df, train_stats_df], axis=1)

    print("=== TEXT STATISTICS BY RULE VIOLATION ===")
    violation_stats = train_with_stats.groupby('rule_violation')[['char_count', 'word_count', 'sentence_count', 
                                                                 'avg_word_length', 'url_count', 'special_char_count']].agg(['mean', 'median', 'std'])
    print(violation_stats)

    # Box plots for text statistics by rule violation
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('Text Statistics by Rule Violation', fontsize=16)

    stats_columns = ['char_count', 'word_count', 'avg_word_length', 'url_count', 'sentence_count', 'special_char_count']
    titles = ['Character Count', 'Word Count', 'Average Word Length', 'URL Count', 'Sentence Count', 'Special Character Count']

    for i, (col, title) in enumerate(zip(stats_columns, titles)):
        row = i // 3
        col_idx = i % 3
        
        sns.boxplot(data=train_with_stats, x='rule_violation', y=col, ax=axes[row, col_idx])
        axes[row, col_idx].set_title(f'{title} by Rule Violation')
        axes[row, col_idx].set_xlabel('Rule Violation (0=No, 1=Yes)')

    plt.tight_layout()
    plt.show()
    
    return violation_stats, train_with_stats

# =============================================================================
# 6. Rule Analysis
# =============================================================================

def rule_analysis(train_df, test_df):
    """Analyze rules in both datasets"""
    print("=" * 60)
    print("6. RULE ANALYSIS")
    print("=" * 60)
    
    # Training data rules
    train_rules = train_df['rule'].value_counts()
    print("Training Data - Rule Distribution:")
    print(train_rules)

    # Test data rules
    test_rules = test_df['rule'].value_counts()
    print("\nTest Data - Rule Distribution:")
    print(test_rules)

    # Compare rule distributions
    all_rules = set(train_rules.index) | set(test_rules.index)
    rule_comparison = pd.DataFrame({
        'Train_Count': [train_rules.get(rule, 0) for rule in all_rules],
        'Test_Count': [test_rules.get(rule, 0) for rule in all_rules],
        'Rule': list(all_rules)
    })

    rule_comparison['Train_Prop'] = rule_comparison['Train_Count'] / len(train_df)
    rule_comparison['Test_Prop'] = rule_comparison['Test_Count'] / len(test_df)

    print("\nRule Distribution Comparison:")
    print(rule_comparison.sort_values('Train_Count', ascending=False))
    
    # Visualize rule distributions
    fig, axes = plt.subplots(2, 1, figsize=(15, 12))

    # Training data
    train_rules.plot(kind='bar', ax=axes[0], color='skyblue')
    axes[0].set_title('Rule Distribution - Training Data')
    axes[0].set_xlabel('Rules')
    axes[0].set_ylabel('Count')
    axes[0].tick_params(axis='x', rotation=45)

    # Test data
    test_rules.plot(kind='bar', ax=axes[1], color='lightcoral')
    axes[1].set_title('Rule Distribution - Test Data')
    axes[1].set_xlabel('Rules')
    axes[1].set_ylabel('Count')
    axes[1].tick_params(axis='x', rotation=45)

    plt.tight_layout()
    plt.show()
    
    # Rule violation rates by rule type (training data only)
    rule_violation_rates = train_df.groupby('rule')['rule_violation'].agg(['count', 'sum', 'mean']).reset_index()
    rule_violation_rates.columns = ['rule', 'total_posts', 'violations', 'violation_rate']
    rule_violation_rates = rule_violation_rates.sort_values('violation_rate', ascending=False)

    print("\n=== RULE VIOLATION RATES ===")
    print(rule_violation_rates)

    # Visualize violation rates
    plt.figure(figsize=(12, 8))
    sns.barplot(data=rule_violation_rates, x='violation_rate', y='rule', orient='h')
    plt.title('Rule Violation Rates by Rule Type')
    plt.xlabel('Violation Rate')
    plt.ylabel('Rule')
    plt.tight_layout()
    plt.show()
    
    return train_rules, test_rules, rule_comparison, rule_violation_rates

# =============================================================================
# 7. Subreddit Analysis
# =============================================================================

def subreddit_analysis(train_df, test_df):
    """Analyze subreddits in both datasets"""
    print("=" * 60)
    print("7. SUBREDDIT ANALYSIS")
    print("=" * 60)
    
    # Training data subreddits
    train_subreddits = train_df['subreddit'].value_counts()
    print(f"Training Data - Unique Subreddits: {len(train_subreddits)}")
    print("Top 10 Subreddits:")
    print(train_subreddits.head(10))

    # Test data subreddits
    test_subreddits = test_df['subreddit'].value_counts()
    print(f"\nTest Data - Unique Subreddits: {len(test_subreddits)}")
    print("All Subreddits:")
    print(test_subreddits)

    # Check overlap
    common_subreddits = set(train_subreddits.index) & set(test_subreddits.index)
    print(f"\nCommon Subreddits: {len(common_subreddits)}")
    print("Common Subreddits:", list(common_subreddits))
    
    # Visualize top subreddits
    fig, axes = plt.subplots(2, 1, figsize=(15, 12))

    # Training data - top 20
    train_subreddits.head(20).plot(kind='bar', ax=axes[0], color='lightgreen')
    axes[0].set_title('Top 20 Subreddits - Training Data')
    axes[0].set_xlabel('Subreddit')
    axes[0].set_ylabel('Count')
    axes[0].tick_params(axis='x', rotation=45)

    # Test data - all
    test_subreddits.plot(kind='bar', ax=axes[1], color='orange')
    axes[1].set_title('Subreddits - Test Data')
    axes[1].set_xlabel('Subreddit')
    axes[1].set_ylabel('Count')
    axes[1].tick_params(axis='x', rotation=45)

    plt.tight_layout()
    plt.show()
    
    # Subreddit violation rates (training data)
    subreddit_violation_rates = train_df.groupby('subreddit')['rule_violation'].agg(['count', 'sum', 'mean']).reset_index()
    subreddit_violation_rates.columns = ['subreddit', 'total_posts', 'violations', 'violation_rate']
    subreddit_violation_rates = subreddit_violation_rates[subreddit_violation_rates['total_posts'] >= 10]  # Filter for subreddits with at least 10 posts
    subreddit_violation_rates = subreddit_violation_rates.sort_values('violation_rate', ascending=False)

    print("\n=== SUBREDDIT VIOLATION RATES (Min 10 posts) ===")
    print("Top 15 Subreddits by Violation Rate:")
    print(subreddit_violation_rates.head(15))

    print("\nBottom 15 Subreddits by Violation Rate:")
    print(subreddit_violation_rates.tail(15))

    # Visualize top violation rates
    plt.figure(figsize=(12, 8))
    top_violating_subreddits = subreddit_violation_rates.head(15)
    sns.barplot(data=top_violating_subreddits, x='violation_rate', y='subreddit', orient='h')
    plt.title('Top 15 Subreddits by Rule Violation Rate')
    plt.xlabel('Violation Rate')
    plt.ylabel('Subreddit')
    plt.tight_layout()
    plt.show()
    
    return train_subreddits, test_subreddits, subreddit_violation_rates

# =============================================================================
# 8. Example Analysis
# =============================================================================

def example_analysis(train_df, test_df):
    """Analyze positive and negative examples"""
    print("=" * 60)
    print("8. EXAMPLE ANALYSIS")
    print("=" * 60)
    
    # Check for missing examples
    example_columns = ['positive_example_1', 'positive_example_2', 'negative_example_1', 'negative_example_2']

    for col in example_columns:
        train_missing = train_df[col].isnull().sum()
        test_missing = test_df[col].isnull().sum()
        print(f"{col}:")
        print(f"  Training missing: {train_missing} ({train_missing/len(train_df)*100:.1f}%)")
        print(f"  Test missing: {test_missing} ({test_missing/len(test_df)*100:.1f}%)")

    # Analyze example lengths
    def analyze_example_lengths(df, dataset_name):
        print(f"\n=== {dataset_name.upper()} - EXAMPLE LENGTHS ===")
        
        example_stats = {}
        for col in example_columns:
            lengths = df[col].dropna().str.len()
            example_stats[col] = {
                'mean': lengths.mean(),
                'median': lengths.median(),
                'std': lengths.std(),
                'min': lengths.min(),
                'max': lengths.max()
            }
        
        example_stats_df = pd.DataFrame(example_stats).T
        print(example_stats_df)
        
        return example_stats_df

    train_example_stats = analyze_example_lengths(train_df, 'Training Data')
    test_example_stats = analyze_example_lengths(test_df, 'Test Data')
    
    return train_example_stats, test_example_stats

# =============================================================================
# 9. Content Pattern Analysis
# =============================================================================

def extract_patterns(text):
    """Extract various patterns from text content"""
    if pd.isna(text):
        return {
            'has_urls': False,
            'has_email': False,
            'has_phone': False,
            'has_caps_words': False,
            'has_exclamation': False,
            'has_question': False,
            'has_money_symbols': False
        }
    
    text = str(text).lower()
    
    return {
        'has_urls': bool(re.search(r'http[s]?://', text)),
        'has_email': bool(re.search(r'\\S+@\\S+\\.\\S+', text)),
        'has_phone': bool(re.search(r'\\b\\d{3}[-.]?\\d{3}[-.]?\\d{4}\\b', text)),
        'has_caps_words': bool(re.search(r'\\b[A-Z]{3,}\\b', str(text).upper())),
        'has_exclamation': '!' in text,
        'has_question': '?' in text,
        'has_money_symbols': bool(re.search(r'[$€£¥₹]', text))
    }

def content_pattern_analysis(train_df):
    """Analyze common patterns in violating vs non-violating content"""
    print("=" * 60)
    print("9. CONTENT PATTERN ANALYSIS")
    print("=" * 60)
    
    # Apply pattern analysis to training data
    train_patterns = train_df['body'].apply(extract_patterns)
    train_patterns_df = pd.DataFrame(list(train_patterns))

    # Analyze patterns by rule violation
    train_with_patterns = pd.concat([train_df[['rule_violation']], train_patterns_df], axis=1)

    print("=== CONTENT PATTERNS BY RULE VIOLATION ===")
    pattern_by_violation = train_with_patterns.groupby('rule_violation').mean()
    print(pattern_by_violation)

    # Visualize patterns
    fig, ax = plt.subplots(figsize=(12, 8))
    pattern_by_violation.T.plot(kind='bar', ax=ax, rot=45)
    ax.set_title('Content Patterns by Rule Violation')
    ax.set_xlabel('Pattern Features')
    ax.set_ylabel('Proportion')
    ax.legend(['No Violation', 'Violation'])
    plt.tight_layout()
    plt.show()
    
    return pattern_by_violation, train_with_patterns

# =============================================================================
# 10. Word Cloud Analysis
# =============================================================================

def create_wordcloud(text_data, title, max_words=100):
    """Create and display word cloud"""
    if not WORDCLOUD_AVAILABLE:
        print(f"⚠️ Skipping {title} - WordCloud library not available")
        return
        
    # Combine all text
    combined_text = ' '.join(text_data.dropna().astype(str))
    
    # Remove URLs and common stop words
    combined_text = re.sub(r'http\S+', '', combined_text)
    combined_text = re.sub(r'www\S+', '', combined_text)
    
    # Create word cloud
    wordcloud = WordCloud(width=800, height=400, 
                         background_color='white',
                         max_words=max_words,
                         colormap='viridis').generate(combined_text)
    
    plt.figure(figsize=(12, 6))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    plt.title(title, fontsize=16)
    plt.show()

def wordcloud_analysis(train_df):
    """Create word clouds for violating and non-violating content"""
    print("=" * 60)
    print("10. WORD CLOUD ANALYSIS")
    print("=" * 60)
    
    # Word clouds for training data
    violating_posts = train_df[train_df['rule_violation'] == 1]['body']
    non_violating_posts = train_df[train_df['rule_violation'] == 0]['body']

    print("Creating word clouds...")
    create_wordcloud(violating_posts, 'Word Cloud: Rule Violating Posts')
    create_wordcloud(non_violating_posts, 'Word Cloud: Non-Violating Posts')

# =============================================================================
# 11. Data Quality Assessment
# =============================================================================

def data_quality_assessment(train_df, test_df, sample_submission):
    """Assess data quality across datasets"""
    print("=" * 60)
    print("11. DATA QUALITY ASSESSMENT")
    print("=" * 60)
    
    # Check for duplicates
    train_duplicates = train_df.duplicated().sum()
    test_duplicates = test_df.duplicated().sum()
    print(f"Training data duplicates: {train_duplicates}")
    print(f"Test data duplicates: {test_duplicates}")

    # Check for duplicate bodies
    train_body_duplicates = train_df['body'].duplicated().sum()
    test_body_duplicates = test_df['body'].duplicated().sum()
    print(f"\nTraining data duplicate bodies: {train_body_duplicates}")
    print(f"Test data duplicate bodies: {test_body_duplicates}")

    # Check for empty or very short content
    train_empty = (train_df['body'].str.len() < 10).sum()
    test_empty = (test_df['body'].str.len() < 10).sum()
    print(f"\nTraining data posts with <10 characters: {train_empty}")
    print(f"Test data posts with <10 characters: {test_empty}")

    # Check row_id consistency
    train_row_ids = set(train_df['row_id'])
    test_row_ids = set(test_df['row_id'])
    sample_row_ids = set(sample_submission['row_id'])

    print(f"\nRow ID consistency:")
    print(f"Test and sample submission match: {test_row_ids == sample_row_ids}")
    print(f"Train and test overlap: {len(train_row_ids & test_row_ids)} IDs")

    # Data type consistency
    print("\nData type consistency:")
    for col in train_df.columns:
        if col in test_df.columns:
            train_type = train_df[col].dtype
            test_type = test_df[col].dtype
            consistent = train_type == test_type
            print(f"{col}: Train({train_type}) vs Test({test_type}) - {'✅' if consistent else '❌'}")

# =============================================================================
# 12. Train vs Test Distribution Comparison
# =============================================================================

def train_test_comparison(train_stats_df, test_stats_df):
    """Compare distributions between train and test sets"""
    print("=" * 60)
    print("12. TRAIN VS TEST DISTRIBUTION COMPARISON")
    print("=" * 60)
    
    # Combine statistics for comparison
    comparison_stats = pd.DataFrame({
        'Train_Mean': train_stats_df.mean(),
        'Test_Mean': test_stats_df.mean(),
        'Train_Std': train_stats_df.std(),
        'Test_Std': test_stats_df.std()
    })

    comparison_stats['Mean_Diff'] = abs(comparison_stats['Train_Mean'] - comparison_stats['Test_Mean'])
    comparison_stats['Std_Ratio'] = comparison_stats['Test_Std'] / comparison_stats['Train_Std']

    print("Statistical Comparison:")
    print(comparison_stats)

    # Visualize distributions
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('Train vs Test Distribution Comparison', fontsize=16)

    stats_columns = ['char_count', 'word_count', 'avg_word_length', 'url_count', 'sentence_count', 'special_char_count']
    titles = ['Character Count', 'Word Count', 'Average Word Length', 'URL Count', 'Sentence Count', 'Special Character Count']

    for i, (col, title) in enumerate(zip(stats_columns, titles)):
        row = i // 3
        col_idx = i % 3
        
        axes[row, col_idx].hist(train_stats_df[col], bins=30, alpha=0.7, label='Train', density=True)
        axes[row, col_idx].hist(test_stats_df[col], bins=30, alpha=0.7, label='Test', density=True)
        axes[row, col_idx].set_title(title)
        axes[row, col_idx].legend()
        axes[row, col_idx].set_xlabel('Value')
        axes[row, col_idx].set_ylabel('Density')

    plt.tight_layout()
    plt.show()
    
    return comparison_stats

# =============================================================================
# 13. Key Insights and Recommendations
# =============================================================================

def generate_insights_and_recommendations(train_df, test_df, train_stats_df, test_stats_df):
    """Generate comprehensive insights and recommendations"""
    print("=" * 60)
    print("13. KEY INSIGHTS AND RECOMMENDATIONS")
    print("=" * 60)
    
    # Dataset size insights
    print("📊 DATASET SIZE:")
    print(f"• Training set: {len(train_df):,} samples")
    print(f"• Test set: {len(test_df):,} samples")
    print(f"• Test set is {len(test_df)/len(train_df)*100:.1f}% of training set size")
    print()

    # Class balance insights
    violation_rate = train_df['rule_violation'].mean()
    print("⚖️ CLASS BALANCE:")
    print(f"• Overall violation rate: {violation_rate:.1%}")
    if violation_rate < 0.3 or violation_rate > 0.7:
        print("• ⚠️ Class imbalance detected - consider stratified sampling or class weights")
    else:
        print("• ✅ Classes are reasonably balanced")
    print()

    # Rule insights
    print("📋 RULE ANALYSIS:")
    print(f"• Unique rules in training: {len(train_df['rule'].unique())}")
    print(f"• Unique rules in test: {len(test_df['rule'].unique())}")
    most_common_rule = train_df['rule'].value_counts().index[0]
    print(f"• Most common rule: '{most_common_rule[:50]}...'")
    print()

    # Subreddit insights
    print("🏷️ SUBREDDIT ANALYSIS:")
    print(f"• Unique subreddits in training: {len(train_df['subreddit'].unique())}")
    print(f"• Unique subreddits in test: {len(test_df['subreddit'].unique())}")
    common_subreddits = len(set(train_df['subreddit'].unique()) & set(test_df['subreddit'].unique()))
    print(f"• Common subreddits: {common_subreddits}")
    print()

    # Text insights
    avg_train_length = train_stats_df['char_count'].mean()
    avg_test_length = test_stats_df['char_count'].mean()
    print("📝 TEXT CHARACTERISTICS:")
    print(f"• Average post length (train): {avg_train_length:.0f} characters")
    print(f"• Average post length (test): {avg_test_length:.0f} characters")
    print(f"• Length difference: {abs(avg_train_length - avg_test_length)/avg_train_length*100:.1f}%")
    print()

    # URL insights
    train_url_rate = train_stats_df['url_count'].mean()
    test_url_rate = test_stats_df['url_count'].mean()
    print("🔗 URL ANALYSIS:")
    print(f"• Average URLs per post (train): {train_url_rate:.2f}")
    print(f"• Average URLs per post (test): {test_url_rate:.2f}")
    print(f"• Posts with URLs are more likely to violate rules")
    print()

    print("🎯 RECOMMENDATIONS:")
    print("• Use text length, URL presence, and special characters as features")
    print("• Consider rule-specific models due to varying violation rates")
    print("• Apply text preprocessing to normalize content")
    print("• Use subreddit information as a categorical feature")
    print("• Consider ensemble methods combining different text representations")
    print("• Implement cross-validation with stratification by rule type")

# =============================================================================
# Main Execution Function
# =============================================================================

def main():
    """Main function to run the comprehensive EDA"""
    print("🚀 Starting Comprehensive EDA for AI Content Moderation Dataset")
    print("=" * 80)
    
    try:
        # 1. Load and inspect data
        train_df, test_df, sample_submission = load_and_inspect_data()
        
        # 2. Missing values analysis
        train_missing, test_missing = missing_values_analysis(train_df, test_df)
        
        # 3. Target variable analysis
        target_counts, target_props, imbalance_ratio = target_variable_analysis(train_df)
        
        # 4. Text content analysis
        train_stats_df, test_stats_df = text_content_analysis(train_df, test_df)
        
        # 5. Text statistics by rule violation
        violation_stats, train_with_stats = text_stats_by_violation(train_df, train_stats_df)
        
        # 6. Rule analysis
        train_rules, test_rules, rule_comparison, rule_violation_rates = rule_analysis(train_df, test_df)
        
        # 7. Subreddit analysis
        train_subreddits, test_subreddits, subreddit_violation_rates = subreddit_analysis(train_df, test_df)
        
        # 8. Example analysis
        train_example_stats, test_example_stats = example_analysis(train_df, test_df)
        
        # 9. Content pattern analysis
        pattern_by_violation, train_with_patterns = content_pattern_analysis(train_df)
        
        # 10. Word cloud analysis
        wordcloud_analysis(train_df)
        
        # 11. Data quality assessment
        data_quality_assessment(train_df, test_df, sample_submission)
        
        # 12. Train vs test comparison
        comparison_stats = train_test_comparison(train_stats_df, test_stats_df)
        
        # 13. Generate insights and recommendations
        generate_insights_and_recommendations(train_df, test_df, train_stats_df, test_stats_df)
        
        print("\n" + "=" * 80)
        print("✅ Comprehensive EDA completed successfully!")
        print("=" * 80)
        
    except Exception as e:
        print(f"❌ Error during EDA execution: {str(e)}")
        raise

if __name__ == "__main__":
    main()