# Machine Learning System for Credit Card Fraud Detection

Dataset: https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud (free, Kaggle account required)

## Run order
1. `pip install -r requirements.txt`
2. Put `creditcard.csv` in `data/`
3. `python 01_prepare.py` -> cleaning, split, scaling
4. `python 02_imbalance.py` -> imbalance strategy comparison
5. `python 03_models.py` -> train/tune/evaluate 4 models (about 10 min on 1 CPU)
6. `python 04_explain.py` -> figures, SHAP, error analysis
7. `streamlit run app.py` -> web app (uses `models/` and `sample_transactions.csv`)

Or run `Fraud_Detection_Project.ipynb` on Google Colab.

## Data
283,726 transactions after removing 1,081 duplicates; 473 fraud (0.167%). Split 70/15/15 stratified: train 198,608 (331 fraud), validation 42,559 (71), test 42,559 (71). Random seed 42.

## Imbalance strategies (validation set, threshold tuned on F1)
| model   | strategy      |   precision |   recall |    f1 |   pr_auc |
|:--------|:--------------|------------:|---------:|------:|---------:|
| LogReg  | Baseline      |       0.821 |    0.775 | 0.797 |    0.69  |
| LogReg  | Class weights |       0.887 |    0.775 | 0.827 |    0.699 |
| LogReg  | SMOTE         |       0.889 |    0.789 | 0.836 |    0.696 |
| LogReg  | Undersampling |       0.675 |    0.732 | 0.703 |    0.545 |
| XGBoost | Baseline      |       0.949 |    0.789 | 0.862 |    0.826 |
| XGBoost | Class weights |       0.932 |    0.775 | 0.846 |    0.819 |
| XGBoost | SMOTE         |       1     |    0.732 | 0.846 |    0.8   |
| XGBoost | Undersampling |       0.761 |    0.718 | 0.739 |    0.675 |

## Final models (test set, threshold tuned on validation)
| model               |   precision |   recall |    f1 |   pr_auc |   roc_auc |   tp |   fp |   fn |
|:--------------------|------------:|---------:|------:|---------:|----------:|-----:|-----:|-----:|
| Logistic Regression |       0.809 |    0.775 | 0.791 |    0.681 |     0.961 |   55 |   13 |   16 |
| Random Forest       |       0.887 |    0.775 | 0.827 |    0.793 |     0.967 |   55 |    7 |   16 |
| XGBoost             |       0.95  |    0.803 | 0.87  |    0.819 |     0.968 |   57 |    3 |   14 |
| Neural Network      |       0.789 |    0.789 | 0.789 |    0.745 |     0.965 |   56 |   15 |   15 |

Best model: XGBoost (precision 0.95, recall 0.80, F1 0.87, PR-AUC 0.82): 57 of 71 frauds caught with 3 false alarms.
