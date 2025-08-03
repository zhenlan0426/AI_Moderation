There are 3 datasets in the data folder from different Kaggle competitions.
1. the first one is the competition I am working on. The taks is given a particular Reddit rule, to predict if a comment breaks that rule. Two positive examples and two negative examples are provided.
2. the second one is to predict severe_toxicity (toxic, severe_toxic combined), obscene, threat, insult, identity_hate given a comment. ✅ CLEANED & FIXED: Combined test/test_labels, removed -1 values (89,186 rows), merged with train set. CSV parsing issues resolved with proper quoting. Final dataset: 223,549 samples saved as data2_cleaned.csv
3. the third one is to predict severe_toxicity, obscene, threat, insult, identity_hate, sexual_explicit given a comment. ✅ CLEANED & FIXED: Removed 37 extra columns, kept essential columns. CSV parsing issues resolved with proper quoting. Final dataset: 1,804,874 samples saved as data3_cleaned.csv

📁 **Data cleaning scripts are organized in the `data_cleaning/` folder:**
- `fix_both_datasets.py` - Main script that processes both Data2 and Data3
- `validate_fixed_files.py` - Validation script to verify datasets work properly
- `README.md` - Detailed documentation of the cleaning process

