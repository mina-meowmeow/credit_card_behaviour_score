# Credit Card Behaviour Score

A behaviour score for an existing credit card portfolio: the probability that an open, not-past-due account defaults going forward. The project runs from exploration through two models to a credit-line and collections policy.

- Dev sample: 96,806 accounts, 1.42% defaults (1,372 bad)
- Validation sample: 41,792 accounts, no target
- Features: 1,214 raw columns plus 14 engineered ones. Column names are anonymised and there is no data dictionary.

The problem statement is in [instruction.md](instruction.md).

## Results

Held-out test data.

| Check | Benchmark | Scorecard | LightGBM |
|---|---|---|---|
| KS | > 40% | 46.0% | 53.0% |
| Gini | higher is better | 0.610 | 0.669 |
| ROC-AUC | n/a | 0.805 | 0.834 |
| Rank ordering monotonic | yes | yes | yes |
| Brier score | lower is better | 0.0137 | 0.0135 |
| Expected / actual ratio | about 1.0 | 0.997 | 0.988 |
| PSI, dev test vs. validation | < 0.10 | 0.0006 | 0.0003 |

The riskiest three deciles hold 27% of accounts and capture 79.3% of defaults.

## Approach

**Exploration** (`eda.ipynb`). Defaults are rare, so accuracy is useless and everything is judged on AUC, KS and calibration. 1,185 of the 1,214 features have missing values. Some columns are 88-100% missing, which looks structural (no bureau record, no such product), so missing values get their own bin or tree split instead of being imputed. 56 constant columns are dropped.

**Feature engineering** (`feature_engineering.ipynb`). With no data dictionary, column meanings were recovered from the data:

- `onus_attribute_1` looks like a credit limit (round currency values, 25,000 to 2,800,000). `onus_attribute_2`, `17`, `20` and `23` are bounded, highly correlated ratios, likely utilization at different snapshots. `onus_attribute_2` has no missing values and the strongest raw correlation with the target (0.108), so it is the internal utilization feature.
- The 664 transaction columns were grouped with hierarchical clustering on `1 - |corr|`. Three clusters stand out: broad everyday spend, rare large cash-like activity, and a small higher-risk cluster.
- `bureau_433` has the same bounded-ratio shape and serves as external utilization. Among 288 low-cardinality bureau columns, pairs ten apart are non-decreasing in more than 99.9% of rows, which points to counts over nested time windows. The 20 most target-correlated ones feed the delinquency features.
- `bureau_enquiry_1` to `50` split into five blocks of ten that never decrease from one block to the next, so they are cumulative enquiry counts over five lookback windows. Recent over long-term enquiries gives a credit-seeking intensity feature.
- No 30-day vs. 90-day structure could be found for the transaction columns, so recency there is approximated by activity breadth.

**Scorecard** (`scorecard.ipynb`). Monotonic WoE binning with `optbinning` on 1,172 candidates, then an IV filter of 0.02 to 0.5, which leaves 411. Ten variables above the 0.5 cap (up to IV 0.70) are dropped as likely leakage. A greedy correlation filter (|r| < 0.6, at most 20 variables) and iterative removal of wrong-sign coefficients give 16 variables across all four feature groups. Scores use the standard points-to-double-odds scaling: 600 points at 50:1 odds, 20 PDO.

**Gradient boosting** (`gbm_model.ipynb`). LightGBM on the full feature set with a 60/15/25 train, calibration and test split. Imbalance handling was compared with identical hyperparameters:

| Approach | Test AUC |
|---|---|
| Class weights | 0.819 |
| ADASYN | 0.835 |
| SMOTE | 0.836 |

SMOTE won, but it needs median imputation, which blurs the missingness signal. Class weights are the safer fallback. For calibration, isotonic (Brier 0.01351) beat Platt (0.01390) and the raw model (0.01356).

**Validation** (`model_evaluation.ipynb`). KS, Gini, rank ordering, Brier, expected/actual by decile, and PSI.

**Risk strategy** (`risk_management.ipynb`). Ten bands are cut on the dev test predictions, not on validation, so each band has a real out-of-sample bad rate.

| Band | Share of portfolio | Bad rate vs. average | Credit-line action | Collections |
|---|---|---|---|---|
| 1-7 | 72.6% | 0.02x to 0.67x | none | no |
| 8 | 9.7% | 1.15x | enhanced monitoring | pre-emptive outreach |
| 9 | 8.4% | 2.28x | reduce line | pre-emptive outreach |
| 10 | 9.3% | 5.14x | freeze line, block cash | pre-emptive outreach |

Bands 8 to 10 cover 11,471 validation accounts. Each gets up to three SHAP-based reason codes in plain English. The most common first reason is high revolving utilization (6,348 accounts). Reason text only claims what the earlier analysis supports, never a specific loan or merchant type.

## Findings

- Utilization dominates. One raw column, `onus_attribute_2`, beats every other single feature and drives the top reason code for over half of the adverse-action accounts.
- Oversampling beat class weighting here, the opposite of the usual result for tree ensembles.
- Aggregate calibration hid a problem. The GBM's two safest deciles have expected/actual ratios of 3.4 and 4.8, from only 1 or 2 defaults in roughly 3,000 accounts. The scorecard stays between 0.83 and 1.49 in every decile.

## Limitations

- Column meanings are inferred, so labels like "cash withdrawal" and "high-risk merchant" are hypotheses to confirm with the data owner, especially before using the reason codes.
- PSI is a single dev-vs-validation snapshot. In production it should be tracked over time and per variable.
- The GBM's safe-end calibration needs a larger calibration sample before it drives tight approval cutoffs.
- Suggested use: the scorecard for sign-off, since it is auditable and monotonic, and the GBM behind the risk strategy, where its extra discrimination matters more than interpretability.

## Files

| File | What it is |
|---|---|
| `eda.ipynb` | Exploration |
| `feature_engineering.ipynb` | Domain features |
| `scorecard.ipynb` | WoE scorecard, writes `submission.csv` |
| `gbm_model.ipynb` | LightGBM, writes `submission_gbm.csv` |
| `model_evaluation.ipynb` | Validation checks |
| `risk_management.ipynb` | Policy and reason codes, writes `risk_management_actions.csv` |
| `CODE_WALKTHROUGH.md` | Why the code looks the way it does, including bugs hit along the way |

Every notebook re-derives the shared feature code instead of importing it, so each runs on its own.

## Running it

The raw data files are not in the repository (280 MB and 121 MB). Put `Dev_data_to_be_shared.csv` and `validation_data_to_be_shared.csv` in the project root, then:

```bash
pip install -r requirements.txt
jupyter notebook
```
