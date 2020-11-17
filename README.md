# Prediction of repeated-dose intravenous ketamine response in major depressive disorder by using the GWAS-based machine learning approach

- `step1_split_dataset.py` : Randomly divide the initial dataset into six folds.
- `step2_feature_selection.py` : Calculate random forest importance score based on GWAS result.
- `step3_model_construction.py` : Model construction.
- `plink.sh` : Conduct quality control and genome-wide logistic regression in PLINK v.1.9 and encode the the genotype data as 0, 1 or 2.
- `models` : The models conducted in this study.

