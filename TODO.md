- remove url link with link + domain
- to use subreddit or not
- rules + positive/negative examples + test examples, backprop from positive/negative examples and keep track of test example logit predictions
- text normalization
- iterative data cleaning. top false positives and false negatives by model -> GPT rerate GT -> retrain model -> repeat
- due to target noise, introduce soft labels
- maybe URL is enough, no need for domain as there are too many domains?
- bias or not for lm_head?

- ✅ data2 and data3 needs to be split into positive/negative examples