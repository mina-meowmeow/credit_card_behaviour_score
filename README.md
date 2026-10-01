# Credit Card Behaviour Score

A behaviour score for an existing credit card portfolio: the probability that an open, not-past-due account defaults going forward. The project covers the full path from exploration to a credit-line and collections policy.

Dev sample: 96,806 accounts (1.42% default). Validation sample: 41,792 accounts. Features: 1,214 raw columns plus 14 engineered ones. The column names are anonymised and there is no data dictionary.

## Results (held-out test data)

| Model | KS | Notes |
|---|---|---|
| WoE / IV scorecard (logistic regression, 16 variables, PDO-scaled) | 46.0% | Interpretable, test AUC ~0.80 |
| LightGBM (SMOTE + isotonic calibration) | 53.0% | Higher discrimination |

PSI between the dev test split and validation is 0.0003, so the populations are stable. The riskiest three deciles hold about 27% of accounts and capture 79.3% of defaults.

## Notebooks

Run in this order. Each one is standalone and reloads the raw data.

| Notebook | Purpose |
|---|---|
| `eda.ipynb` | Class imbalance, missingness, constant columns, dev vs. validation drift |
| `feature_engineering.ipynb` | Recovering column semantics from statistical structure and building domain features |
| `scorecard.ipynb` | Monotonic WoE binning, IV screening, logistic regression, PDO scaling |
| `gbm_model.ipynb` | LightGBM, class weights vs. SMOTE vs. ADASYN, Platt vs. isotonic calibration |
| `model_evaluation.ipynb` | KS, Gini, rank ordering, expected/actual calibration, PSI |
| `risk_management.ipynb` | Risk bands, credit-line actions, pre-emptive collections, SHAP reason codes |

## Other files

- `Behaviour_Score_Dossier.html`: full write-up (approach, insights, limitations). Open it in a browser.
- `CODE_WALKTHROUGH.md`: reasoning behind the code and the bugs found along the way.
- `instruction.md`: the original problem statement.
- `submission.csv`, `submission_gbm.csv`: predicted default probabilities for the validation set.
- `risk_management_actions.csv`: per-account risk band, action and reason codes.

## Reproducing

The two raw data files are not in the repository because of their size (280 MB and 121 MB). Place `Dev_data_to_be_shared.csv` and `validation_data_to_be_shared.csv` in the project root, then:

```bash
pip install -r requirements.txt
jupyter notebook
```

## Limitations

Column meanings are inferred from the data, not documented, so labels such as "cash withdrawal" or "high-risk merchant" are working hypotheses. See the dossier for details.
