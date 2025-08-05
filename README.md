# AI_Moderation
### Overview
If you’ve ever had a comment taken down on Reddit and wondered “why?”, you’re not alone. Each subreddit has its own set of guidelines, and trying to understand individual subreddit moderation can feel like chaos.

In this competition, you’ll bring some ‘comment sense’ to the table and work with real data to build models that predict which rule (if any) a comment may have broken.

### Description
Your task is to create a binary classifier that predicts whether a Reddit comment broke a specific rule. The dataset comes from a large collection of moderated comments, with a range of subreddit norms, tones, and community expectations.

The rules you’ll be working with are based on actual subreddit guidelines, but the dataset itself is drawn from older, unlabeled content. A small labeled dev set has been created to help you get started.

This is a chance to explore how machine learning can support real-world content moderation, particularly in communities with unique rules and norms.

### Background
Inspired by the work of our colleagues Deepak Kumar, Yousef AbuHashem, and Zakir Durumeric where large language models were deployed to try to guess the reasons that moderators used to remove comments. This work builds upon the work of Eshwar Chandrasekharan and Eric Gilbert which collected a set of millions of moderated comments.

This several-year-old dataset is unlabeled. It is accompanied by a list of hypothetical rules—derived from real rules on a variety of subreddits—to help identify potential comment violations.

### Rules Classification
Participants have access to a small subset of the data, which can be used as a dev resource. This information is suitable for use as training data or for few-shot examples. The remainder of the labels will be used, in a 30%:70% to form the public and private test sets.

### Key challenges
The training dataset contains only two rules. The test dataset contains additional rules that models must be able to generalize to. (The number of unseen rules is not specified as part of the competition.)

### Problem formulation
a positive and a negative example are provided for each rule. During fine-tuning, model will be trained on label1 and label2. At inference time, prompt will be the same and prediction from last token will be the model output.


```python
prompt = (
    "You are given a comment on reddit. Your task is to classify if it violates the given rule.\n"
    f"Subreddit: r/{subreddit}\n"
    f"Rule: {rule}\n"
    f"Comment: {comment1}\n"
    f"Violation: {label1}\n"
    f"Comment: {comment2}\n"
    f"Violation: {label2}\n"
    f"Comment: {row['body']}\n"
    f"Violation:"
)
```