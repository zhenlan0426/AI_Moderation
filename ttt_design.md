Rules + randomly sampled positive example 1 + ground truth (GT) for example 1 + randomly sampled negative example 2 (randomly decides if positive or negative appears first) + ground truth for example 2 + test example (only for dataset 1 and for example under the same rule). model was be trained on GT1 and GT2. At the same time, we can record prediction for test example along the way.


prompts = []
for i, row in df.iterrows():
    text = f"""
            You are given a comment on reddit. Your task is to classify if it violates the given rule.
            Subreddit: r/{row.subreddit}
            Rule: {row.rule}
            Comment: {row.positive_example_1}
            Violation: Yes
            Comment: {row.negative_example_1}
            Violation: No
            Comment: {row.body}
            Violation:
            """

    prompts.append(text)