# Code walkthrough

This file explains why each notebook's code looks the way it does, including the bugs I hit and how I fixed them. The results and recommendations are in [README.md](README.md).

Every notebook re-derives the feature-engineering setup instead of importing a shared module. That is on purpose: any notebook can be opened and run top to bottom without first running four others in the right order. The cost is duplicated code.

## 1. eda.ipynb
**WHY**: before building anything, need to know what shape the data is in - how
imbalanced the target is, how much is missing, whether the four documented
feature groups (onus / transaction / bureau / bureau_enquiry) are clean to
work with, and whether dev and validation actually look like the same
population.

**HOW**:
- Both CSVs are read directly with pandas.read_csv (280MB and 121MB,
  small enough to load whole - no chunking needed).
- Class imbalance: dev['bad_flag'].value_counts(normalize=True) - this
  single number (1.42% bad) is what rules out accuracy as an evaluation
  metric for every model built later and forces AUC/KS/PSI-style metrics
  instead.
- Missingness: dev.isna().mean() per column, sorted descending. This is
  not just a data-quality check - the columns that turned out ~88-100%
  missing (onus_attribute_43-48, bureau_447/436/449/148) later informed
  the decision to let missing values form their own WoE/tree bin instead
  of being imputed, since near-universal missingness on a subset of
  accounts looks structural ("this product feature doesn't apply to you")
  rather than random.
- Constant columns: nunique() per column, flagged where <= 1. These 56
  columns are dropped before any modeling because a column with a single
  value carries zero information and can crash some binning routines
  (see scorecard.ipynb below) if left in with all-NaN edge cases.
- Feature groups: built with c.startswith(prefix) list comprehensions.
  The one subtlety: bureau_enquiry_* columns also start with "bureau_",
  so the bureau_cols list is built as
  [c for c in cols if c.startswith('bureau_') and c not in enq_cols]
  computing enq_cols first and excluding it, rather than a naive
  startswith('bureau_') that would silently double-count 50 columns into
  both groups.
- Raw univariate signal: dev[num_cols].corrwith(dev['bad_flag']) gives a
  first, crude read on which columns matter before any real feature
  engineering - this ranking is what feature_engineering.ipynb later uses
  to pick, e.g., the top-20 bureau "delinquency-like" columns.


## 2. feature_engineering.ipynb
**WHY**: the raw columns are anonymized (onus_attribute_17, bureau_442, ...)
with no data dictionary. The brief asks for named domain features (credit
utilization, cash-vs-spend ratio, delinquency trajectory, inquiry velocity)
that require knowing which raw column means what. Since that mapping was
never provided, it had to be recovered from the data itself before any
formula could be written - this notebook is as much reverse-engineering
as it is feature engineering.

HOW, group by group:

  2.1 Credit Utilization & Limit Dynamics (onus_attribute)
- onus_attribute_1 was identified as "credit limit" by its raw scale:
  round-number currency values from 25,000 to 2,800,000, matching the
  example the brief itself gives ("on-us attributes like credit
  limit").
- onus_attribute_2/17/20/23 were tested pairwise with .corr() and
  found mutually correlated at r=0.85-0.9997, all bounded roughly in
  [-1.1, 1.25]. That combination (bounded near 0-1, high cardinality,
  near-perfectly correlated with each other) is the signature of the
  same ratio computed at different snapshots, not four unrelated
  columns. A competing hypothesis - that these are utilization
  computed as balance/limit from two other raw columns - was tested
  directly: onus_attribute_5 / onus_attribute_1 was correlated
  against onus_attribute_2 and only reached r=0.125, far below the
  0.85+ seen among the ratio-looking columns themselves. That result
  ruled out the "reconstruct it from balance and limit" idea and
  settled on using onus_attribute_2 directly (0% missing, highest
  single raw correlation with bad_flag at 0.108) as internal
  utilization.
- limit_stress_persistence just counts, per account, how many of the
  four correlated snapshots are >= 0.8 - a direct count-based read on
  "stayed near the limit," not a threshold applied once.

  2.2 Transaction Velocity & RFM (transaction_attribute, 664 columns)
- With no naming to go on and 664 columns, a positional/naming
  approach is impossible, so a purely statistical clustering approach
  was used instead: build a distance matrix as 1 - |correlation|
  between all 664 columns, feed it to scipy's hierarchical
  linkage (average linkage), and cut the dendrogram into 14 clusters
  with fcluster(..., criterion='maxclust').
- Each of the 14 clusters is then summarized (column count, fraction
  of accounts with a nonzero value, mean |correlation| with the
  target) and the three clusters of interest are picked
  programmatically, not by hardcoding cluster ID numbers - because
  cluster IDs from hierarchical clustering are an arbitrary
  relabeling and would be fragile to hardcode:
  spend_cluster    = cluster with the most columns (broad, common
  spend - argmax on n_cols)
  cash_cluster     = among clusters with >=10 columns, the one with
  the lowest fraction of accounts using it at all
  (rare-but-large is the cash-advance signature -
  argmin on nonzero_frac)
  highrisk_cluster = among the remaining sizable clusters, the one
  with the highest mean |correlation| to the
  target (argmax on mean_abs_corr_target)
- The same three clusters come out every time this code re-runs in a fresh notebook: the clustering
  itself is deterministic (no randomness in hierarchical clustering),
  and the selection logic is a data-driven argmax/argmin, not a
  remembered number.
- A parallel attempt was made to find a 30-day-vs-90-day window
  structure the way it was found for bureau_enquiry (below): sample
  pairs of columns at many candidate index offsets (1, 2, 3, ...,
  332) and check what fraction of rows are non-decreasing from one to
  the other. For bureau_enquiry this test cleanly separated one
  offset at ~100% from everything else at ~85-90%; for
  transaction_attribute, no offset stood out (all candidates sat in
  the same 86-90% band). That negative result is reported in
  the notebook rather than picking an offset anyway - recency is
  approximated with a breadth/concentration proxy instead of a
  specific day-window ratio, because the data didn't support one.

  2.3 Bureau Stress & Tradeline Health (bureau, 452 columns)
- Scanned all bureau_* columns for ones that are bounded (roughly
  -1 to 2) with high cardinality - the same shape as the onus
  utilization columns above. Five columns matched
  (bureau_433/437/438/444/448); the one with the lowest missing rate,
  bureau_433, became external_utilization.
- Separately, scanned for "count/status-like" columns: <=15 unique
  values, all >= 0 - 288 columns matched. Rather than trust that
  every one of these is a real delinquency signal, they were ranked
  by |correlation| with bad_flag and only the top 20 were kept as the
  delinquency_flag_count / worst_bureau_status inputs - the top 20
  show a visibly higher correlation (0.037-0.052) than the rest of
  the 288, which is the actual justification for the cutoff at 20
  rather than an arbitrary round number.
- To confirm these 288 columns really do represent something
  temporal (not just unrelated categorical fields), pairs offset by
  10 (e.g. bureau_13 vs bureau_23) were checked the same way as the
  enquiry columns below and found non-decreasing in >99.9% of rows -
  the same nested-window evidence pattern used for bureau_enquiry.

  2.4 Inquiry Velocity (bureau_enquiry, 50 columns)
- The stride-search described above was run properly here and found
  a clean signal: columns 10 apart (e.g. bureau_enquiry_1 vs
  bureau_enquiry_11 vs ...31... vs ...41) are non-decreasing in
  100% of sampled rows, and the max value per block of 10 columns
  increases sharply block-to-block (42 -> 72 -> 81 -> 172 -> 297).
  That is the signature of 5 nested, increasingly long lookback
  windows over the same 10 enquiry "types."
- enquiries_recent sums columns 1-10 (shortest window),
  enquiries_longterm sums columns 41-50 (longest window), and
  credit_seeking_intensity is their ratio - a direct, justified
  implementation of "recent vs. long-term enquiries" from the brief,
  built only after the nested-window claim was actually verified
  against the data.


## 3. scorecard.ipynb
**WHY**: the brief explicitly calls for a regulatory-style interpretable model:
monotonic WoE bins, IV-based feature filtering, a logistic regression, and a
PDO-scaled score.

**HOW**:
- The optbinning library's OptimalBinning does exactly the binning half of
  this out of the box: monotonic_trend defaults to 'auto', which already
  produces monotonic bins with no extra configuration, and it computes
  IV per bin (and per variable) internally.
- Rather than fitting 1,172 separate OptimalBinning objects by hand,
  optbinning.BinningProcess does that in one call across every candidate
  column and applies a selection rule directly:
  selection_criteria={'iv': {'min': 0.02, 'max': 0.5}}
  which is a literal implementation of the brief's "IV > 0.02 for
  predictive power; cap at 0.5 to prevent overfitting."
- Bug hit and fixed: BinningProcess crashed with a DecisionTreeClassifier
  error on the first run. The cause was columns that are 100% missing
  (bureau_447, bureau_436) - optbinning's default CART pre-binning step
  tries to fit a tree on the non-missing values of each column, and with
  zero non-missing values there is nothing to fit. Fix: before building
  the candidate list, drop any column where nunique(dropna=True) <= 1
  (covers both fully-missing and truly-constant columns) - the same
  degenerate-column check from eda.ipynb, applied here to the combined
  raw + engineered feature set.
- Result: 411 of 1,172 candidates pass the IV filter. Ten variables
  exceed the 0.5 cap (up to IV=0.70) and are dropped rather
  than kept - a variable that predicts the target that well essentially
  on its own is far more likely to be an artifact (near target leakage)
  than a useful linear signal, which is exactly why the brief
  asks for the cap in the first place.
- 411 variables is not a reviewable scorecard, so two further passes
  narrow it down:
  (a) Redundancy pruning: rank the 411 by IV descending, walk the list,
  and keep a variable only if its WoE-transformed values are not
  highly correlated (|r| < 0.6) with any variable already kept,
  stopping at 20. This is a greedy heuristic, not an exhaustive
  search - it is simple, deterministic, and in practice removes
  near-duplicate variables (e.g. the same signal captured by two
  slightly different snapshots) without needing a full
  combinatorial variable-subset search.
  (b) Sign-consistency cleaning: fit a plain sklearn LogisticRegression
  on the WoE-transformed variables, and if any coefficient comes
  out positive, drop the single worst offender and refit, looping
  until every coefficient is negative. The reasoning: with
  optbinning's WoE convention, a higher WoE means a safer bin, so
  every variable's coefficient should point the same direction: a
  coefficient with the wrong sign is not a real effect, it is a
  multicollinearity artifact from having several correlated
  variables in the same regression, and removing it is standard
  scorecard-building practice, not just a technicality.
  - The result is 16 variables, all one sign, spanning all four input
    groups.
- PDO scaling is a direct algebraic implementation of the standard
  formula: Factor = PDO/ln(2), Offset = Base_Score - Factor*ln(Base_Odds).
  Because the logistic regression's linear score is ln(odds_bad), and
  ln(odds_good) = -ln(odds_bad), the per-variable-per-bin point
  contribution is derived as -Factor*coef*WoE plus an even share of
  (Offset - Factor*intercept) split across the 16 variables - this
  specific split is what makes summing every variable's assigned points
  for a given account reproduce the Score formula exactly, which is the
  whole point of a "points table" a reviewer can add up by hand.
- Second bug hit and fixed while building that points table: optbinning's
  binning_table.build() includes a "Totals" summary row (with Bin='')
  and a "Special" row, and because the Totals row's WoE value is a blank
  string rather than a number, the WoE column's dtype silently becomes
  object once that row is included - which then makes pandas' .round()
  fail with "Expected numeric dtype, got object instead." Fix: filter out
  rows where Bin is '' or 'Special', and explicitly cast the remaining
  WoE column back to float before using it in arithmetic.


## 4. gbm_model.ipynb
**WHY**: build a higher-capacity model as a challenger to the scorecard, using
the full feature set and letting a tree ensemble find nonlinear effects and
interactions the WoE/logistic pipeline can't, while explicitly handling
class imbalance and calibrating the output probabilities.

**HOW**:
- LightGBM was chosen over XGBoost/CatBoost mainly for two practical
  reasons: it is fast on this many columns and rows, and it handles
  missing values natively (relevant here, since most feature columns
  have some missingness, per eda.ipynb).
- A three-way split (60% train / 15% calib / 25% test) is used instead of
  a plain train/test split, because this model needs two things a
  scorecard-style single split can't safely provide at once: an
  early-stopping validation set for the boosting rounds, and a
  calibration-fitting set for the probability recalibration step later.
  Using the final test set for either of those would leak information
  into the reported metrics, so calib is a third, separate slice, and
  test is only ever touched once, at the very end, for reporting.
- Bug hit and fixed - this one silently produced a badly undertrained
  model and is worth documenting: LightGBM's sklearn API, when
  given both scale_pos_weight and eval_metric='auc', still also computes
  its own default metric (binary_logloss) internally, and the
  early-stopping callback, if not told which metric to use, can end up
  tracking that default metric instead of the one that was actually
  asked for. Under a large scale_pos_weight (~70 here, since bad_flag is
  only 1.42% of the data), the model's predicted probabilities get pushed
  toward the extremes very quickly, and binary_logloss on that gets worse
  almost immediately even while ranking (AUC) is still improving - so
  early stopping on logloss halted training after a single tree
  (best_iteration_=1), giving a test AUC of 0.75 instead of the ~0.82 the
  same setup reaches once fixed. The fix: pass metric='auc' directly to
  the LGBMClassifier constructor (not eval_metric to .fit()), and pass
  first_metric_only=True to the early_stopping callback, so there is no
  ambiguity about which metric governs stopping. This was caught by
  printing the per-round AUC trajectory with lgb.record_evaluation and
  noticing it kept climbing well past the point early stopping had
  already halted at.
- Class imbalance was tested three ways under otherwise identical
  hyperparameters and identical early stopping, specifically so the
  comparison isolates the imbalance-handling technique and nothing else:
  - class weight: scale_pos_weight = (#good / #bad) in the training
    split, passed straight into LightGBM's loss function - no
    resampling, so missing values pass through untouched.
  - SMOTE / ADASYN: both are k-nearest-neighbour-based oversamplers
    from imbalanced-learn and neither accepts NaN input, so the
    training (and calibration/test, for consistent scoring) features
    are median-imputed first, purely for this branch. This is a real
    methodological trade-off, flagged explicitly in the notebook: the
    median fill can blur the "missingness itself is informative"
    pattern found in eda.ipynb (e.g. a fully-missing bureau column
    probably means "no such tradeline," not "value near the median").
  - SMOTE came out narrowly ahead of ADASYN and clearly ahead of class
    weighting on test AUC, so it was the one carried forward, with the
    trade-off above documented rather than ignored.
- Calibration: CalibratedClassifierCV needs a way to wrap an
  already-fitted model without touching its fit; the old way to do that
  was cv='prefit', which is deprecated as of recent scikit-learn.
  The current mechanism, sklearn.frozen.FrozenEstimator, wraps the fitted
  LightGBM model so that calling .fit() on the calibrator only fits the
  recalibration map (on the calib split), never re-touches the trees.
  Both Platt (method='sigmoid') and Isotonic (method='isotonic') were
  fit this way and compared on Brier score computed on the untouched test
  split - Isotonic won narrowly; Platt's parametric sigmoid assumption
  slightly overcorrected relative to the empirical, non-parametric
  isotonic fit on this particular rare-event distribution.


## 5. model_evaluation.ipynb
**WHY**: a single AUC number does not tell a credit-risk reviewer everything
they need - it says nothing about whether the score can support a hard
cutoff (rank-ordering), whether a "10% predicted default rate" segment
really does default 10% of the time (calibration), or whether the model
still applies to a population that might have drifted since it was built
(stability). This notebook implements those checks explicitly rather than
assuming AUC is sufficient.

**HOW**:
- Both models are rebuilt inside this notebook (same code as
  scorecard.ipynb and gbm_model.ipynb) rather than loading saved model
  objects, to keep this notebook independently runnable. The GBM
  rebuild only reproduces the already-known winning configuration
  (SMOTE + LightGBM + Isotonic), not the full three-way comparison,
  specifically to avoid re-spending the runtime on a comparison whose
  answer this notebook doesn't need to re-derive.
- The two models intentionally use different train/test splits (the
  scorecard's single 75/25 split vs. the GBM's 60/15/25 split) - this is
  fine, and not an inconsistency, because each model is being evaluated
  on its own unseen data; the splits don't need to be identical
  to each other for that to be valid, they only need to be unseen by
  that model.
- ks_stat implements the textbook Kolmogorov-Smirnov statistic directly:
  sort accounts by predicted score, compute the cumulative fraction of
  goods and the cumulative fraction of bads as you sweep through that
  sorted order, and take the maximum absolute gap between the two curves.
- rank_order_table buckets predictions into deciles and checks that the
  observed bad rate is monotonically non-decreasing once the deciles are
  ordered from safest to riskiest. Bug caught during development: the
  first version of this check had the comparison direction backwards for
  the "higher score = safer" case (it required bad rate to be
  non-increasing instead of non-decreasing after sorting safest-first),
  which would have silently reported a monotonicity failure as a pass.
  It was caught by re-deriving the logic by hand - once the table is
  sorted safest-to-riskiest, the desired property is always "bad rate
  goes up as you move down the table," regardless of which raw direction
  (higher score or higher probability) means "safer" - and fixed by
  removing the direction-dependent branch entirely, since it was
  unnecessary once the table itself is already ordered consistently.
- expected_actual_table computes, per probability decile, the mean
  predicted probability ("expected") against the actual observed bad
  rate ("actual"). This is computed per-decile and not just
  as one overall ratio, because the overall ratio can look fine while
  hiding a problem in one part of the range - which is exactly what
  happened for the GBM: the aggregate ratio was close to 1, but the two
  safest deciles individually showed ratios of 3.4x and 4.8x (a handful
  of actual defaults landing in a very large, very-low-predicted-risk
  bucket is enough to produce that on a 1.4%-base-rate target).
- psi() bins the reference distribution into deciles using
  numpy.quantile, forces the outermost bin edges to -inf/+inf so no
  validation-set value can fall outside the defined bins, applies the
  same edges to the current (validation) distribution, and computes
  the standard formula sum((cur_pct - ref_pct) * ln(cur_pct/ref_pct)).
  Percentages are clipped away from exactly zero before the log/ratio to
  avoid divide-by-zero or log(0) if a bin happens to be empty in one of
  the two populations.


## 6. risk_management.ipynb
**WHY**: a predicted probability by itself is not a decision. This notebook is
where the score becomes an actual operating policy - who gets a reduced
credit line, who gets contacted proactively, and what reason a customer
would be told if their line is cut.

**HOW**:
- The GBM (not the scorecard) is used as the scoring engine here, for two
  reasons: it has higher KS/Gini in model_evaluation.ipynb, and its tree
  structure supports SHAP's TreeExplainer, which gives exact, fast
  per-account attributions - a linear model's coefficients would already
  give per-feature attribution just as validly, but the brief specifically
  asked for SHAP-based reason codes.
- Risk bands are cut on the dev-test PD distribution (via the same
  np.quantile approach as the PSI function above), not on the validation
  set's own distribution. If the bands were re-cut on
  validation directly, there would be no way to attach a meaningful
  "expected bad rate" to each band, since validation has no ground truth.
  By fixing the edges on dev-test data (where the true outcome is known)
  and then simply sorting validation accounts into those same edges, each
  band keeps a real, previously-observed bad rate - and the PSI result
  from model_evaluation.ipynb (PSI=0.0003, "stable") is what
  justifies carrying that number over to validation at all.
- Credit-line action thresholds are set from the resulting
  bad-rate-vs-portfolio-average column of that band table (band 10 at
  ~5.1x, band 9 at ~2.3x, band 8 at ~1.15x) rather than from round
  probability numbers, because a multiplier relative to the portfolio's
  own average is a more defensible cutoff basis than an arbitrary
  absolute probability that means something different on every portfolio.
  Band 8 is deliberately routed to "enhanced monitoring" rather than an
  automated line action, since 1.15x average is a borderline
  signal, not a clear one - it goes to the wider collections net instead,
  where a false positive costs much less than a wrongly frozen card.
- The collections-outreach cutoff (bands 8-10) is intentionally wider
  than the credit-line-action cutoff (bands 9-10), because outreach is a
  cheap and fully reversible action while a line freeze is not: it is
  worth accepting more false positives on the cheap action to catch more
  true positives before they become 30+ DPD, which is the entire premise
  of "pre-emptive."
- shap.TreeExplainer was benchmarked before committing to computing SHAP
  for the full population: 500 rows took 0.65 seconds, which extrapolates
  to under a minute for all ~11,000 adverse-action accounts - fast enough
  to just run on the real population rather than a sample.
- SHAP values are computed for every band-8/9/10 account, but reason
  codes are only generated from the positive-SHAP contributors (features
  pushing the prediction toward higher risk), and only for those
  adverse-tier accounts - business-as-usual accounts get no reason code
  at all. This mirrors how adverse action notices actually work: a
  reason is owed when a decision is adverse, not for an approval, and a
  "protective factor" isn't a reason a line was cut.
- The reason-code text itself is produced by a small dispatch function
  (reason_text) that checks, in order: is this one of the 14 engineered
  features (if so, use its already-known plain description); does it
  start with bureau_enquiry_ (if so, compute which of the five lookback
  windows it belongs to directly from the column's numeric suffix via
  block = (n-1)//10 + 1, and phrase the reason around that window rather
  than guessing a loan purpose); does it start with transaction_attribute_
  (if so, look up its cluster membership from the clustering done in
  feature_engineering.ipynb and phrase accordingly); does it start with
  bureau_ (check membership in the previously-identified utilization-ratio
  or delinquency-count column sets, or fall back to a generic tradeline
  phrase); does it start with onus_attribute_ (similar membership checks
  against the utilization/limit-stress column sets). Every fallback
  phrase is deliberately generic ("adverse indicator on a bureau-reported
  credit tradeline") rather than specific, because inventing a specific
  claim like "excessive personal loan enquiries" for a column that could
  equally be an auto-loan or credit-card enquiry field would be fabricating
  a fact the anonymized data cannot actually support.
- One environment issue hit while installing the shap library: importing
  shap failed with "module 'coverage' has no attribute 'types'" - shap
  pulls in numba, and numba's coverage-integration module expects an API
  (coverage.types) that only exists in coverage.py 7.x, while the
  environment had 6.5.0 installed for an unrelated reason. Fixing this
  was a one-line "pip install -U coverage" and had nothing to do with the
  modeling code itself - noted because it looks like a code bug at first glance.


## Appendix: why the notebooks repeat code
Every notebook from feature_engineering.ipynb onward repeats the same
~60 lines: rebuilding the four column-prefix groups, re-running the
transaction_attribute clustering, and re-computing the 14 engineered
features. This is intentional: it means
scorecard.ipynb, gbm_model.ipynb, model_evaluation.ipynb, and
risk_management.ipynb can each be opened and run top-to-bottom on their
own, in any order, without a hidden dependency on having already run a
different notebook first. I avoided a shared .py module because the deliverables are notebooks only.
