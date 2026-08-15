# Results Summary — Hannah's Notebooks (02–04)

Re-run completed 2026-08-14 after the methodology-review upgrade (extended grid, temporal
holdout, BaselineOnly ablation, CI'd 5-star-relevance ranking, per-tag partially pooled
posteriors, item-prior sensitivity scan, blend-weight sweep, stars/probability track
separation, probability-space hybrid, popularity-blend ranker, cold ranking eval,
calibrated confidence). All three notebooks executed end-to-end with zero error outputs;
every number below is from this run's printed summary cells and reproduces from the
artifacts listed at the bottom. RMSE **and** MAE are reported everywhere (both are
primary metrics). Rating rows with `rating == 0` are excluded per
`config.yaml: zero_rating_policy` (asserted in every notebook).

## 02 — SVD baseline (Surprise)

Extended grid (12 configs on the inner warm holdout) moved the winner to
**n_factors=25, reg_all=0.2** (inner RMSE 0.6793 vs 0.6815 for the old 50/0.1). The
winner still sits on the grid edge (fewer factors / heavier regularization may do
better) — recorded as `winner_interior: false` in `hannah_svd_params.json`.

| model | warm RMSE / MAE | temporal RMSE / MAE | cold val RMSE / MAE | cold test RMSE / MAE |
|---|---|---|---|---|
| global mean | 0.7273 / 0.4990 | 0.7231 / 0.4876 | 0.9125 / 0.6282 | 0.9228 / 0.6313 |
| user mean | 0.7899 / 0.4420 | 0.7914 / 0.4385 | 0.8876 / **0.5493** | 0.9308 / **0.5622** |
| item mean (shrunk) | 0.7175 / 0.4762 | — | (collapses to global) | — |
| bias only (ALS) | 0.6958 / 0.4434 | 0.6945 / 0.4340 | 0.8722 / 0.5890 | 0.8912 / 0.6006 |
| **SVD (tuned)** | **0.6936** / 0.4393 | **0.6923** / 0.4306 | **0.8658** / 0.5793 | **0.8858** / 0.5916 |

- **Temporal honesty check:** 75.1% of random warm-holdout targets have strictly-later
  same-user interactions retained in train. Holding each user's chronologically *last*
  rating instead changes almost nothing (SVD 0.6923 vs 0.6936) — the leakage concern was
  real in principle but does not distort the warm numbers on this data.
- **MAE honesty:** the raw user-mean baseline **wins cold MAE** on both provided splits;
  SVD's cold edge is RMSE-only and driven by bias shrinkage, not item identity (the
  provided splits are 100% item-cold).
- SVD beats bias-only by just 0.0022 warm RMSE — the latent structure is real but small.

Ranking (warm, positives = held **5-star** ratings, 1+99 candidates, seen = all train
interactions incl. rating 0, n = 3,000 users, 95% CIs):

| model | HR@10 [95% CI] | NDCG@10 [95% CI] |
|---|---|---|
| SVD | 0.335 [0.318, 0.352] | 0.184 [0.174, 0.195] |
| bias only | 0.415 [0.397, 0.432] | 0.258 [0.245, 0.270] |
| popularity | **0.698** [0.681, 0.714] | **0.501** [0.487, 0.516] |

## 03 — Bayesian updating (Beta–Binomial + per-tag layer)

- Sequential replay (681,944 date-ordered predict-before-observe steps): log-loss
  static 0.5506 → baseline-only **0.4881** (11.3% of static removed) → +families 0.4885.
  The family layer still adds nothing predictively on warm history (+0.0004) — its value
  is interpretability and cold-start content signal, and taste extraction must contrast
  family/tag means **against the user's base mean** (documented in the handoff).
- Empirical-Bayes prior Beta(3.802, 1.198), S_USER=5 chosen by sequential log-loss
  (0.4775/0.4759/0.4814 for S=2/5/20); PyMC hierarchical cross-check: corr 1.000 with the
  conjugate posteriors, κ=4.6 ≈ S=5, max R-hat 1.005.
- **NEW per-tag partially pooled posteriors**: 496,796 (user, specific-tag) Beta
  posteriors over 41 tags (30 cuisine/dietary/dish/time tags + calorie terciles + 8
  ingredient keywords), prior = Beta(5·m_family, 5·(1−m_family)). Predictive half-split
  check over 126,732 pairs: tag-level corr 0.504 vs family-only 0.528 (MAE 0.1760 vs
  0.1718) — **specific tags deliver granularity for explanations, not predictive lift**;
  the original sparsity concern is now measured rather than assumed.
- **Item prior sensitivity scan** (chronological half-split, recipes with n≥4):
  **S_ITEM=20** beats the hand-set 10 beyond noise; `hannah_recipe_quality.csv`
  regenerated with S=20 and the choice recorded in `hannah_bayes_params.json`.
- Confidence redefined as **1 − (95% credible-interval width)** — evidence-honest: a
  zero-observation user now scores low instead of ~0.83 under the old 1−√variance.
- The multi-component prediction blend is labeled a **heuristic ensemble score**
  (components double-count observations; not a coherent posterior) everywhere it is used.

## 04 — Hybrid, two tracks, ranking, confidence

- **Weight sweep** (w ∈ 0..1 step 0.05, chosen on cold *validation* only): **w\*=0.70 is
  the interior optimum** — the config guess is now a validated choice. Cold RMSE
  **0.8377** (val) / **0.8554** (test) vs SVD 0.8658/0.8858. Warm stars: SVD alone still
  wins (0.6936 vs hybrid 0.7113); cold MAE: refit user-mean still wins (0.5450/0.5532).
- **Probability track** (log-loss/Brier/AUC; star-space blends excluded by design):
  warm eval half — prob-space hybrid (isotonic P(loved|svd) + Bayesian, wp\*=0.75, tuned
  on the fit half only) **0.4717** beats Bayesian-alone 0.4811 and iso-SVD 0.4757. Cold
  stays Bayesian-only (no SVD item signal): log-loss 0.6219/0.6296, AUC 0.715/0.699.
- **Calibration:** raw ECE warm 0.0419, cold 0.1133; regime-matched isotonic repairs it
  (cold test 0.1133 → 0.0122; warm 0.0413 → 0.0091); cross-regime maps do not transfer.
  `confidence_calibrated` is persisted alongside the raw (relative) confidence.
- **Ranking** (same corrected protocol, decontaminated posteriors, 95% CIs): Bayesian
  0.356, hybrid-w\* 0.366, quality-only 0.401, bias-only 0.413, popularity **0.702**.
  The tuned **popularity-blend** (z(hybrid)+λ·z(log-pop), λ\*=2 tuned on 1,000 users,
  evaluated on the held 2,000) reaches HR@10 **0.693 vs popularity 0.697** — it closes
  ~99% of the hybrid→popularity gap **by converging toward popularity: matching it, not
  beating it**. Exposure modeling remains the honest next step.
- **Cold ranking (null result):** Bayesian content score HR@10 0.098 vs random 0.100 —
  indistinguishable. The family-level ensemble score barely varies across cold
  candidates (per-tag posteriors currently inform explanations, not the score); cold
  value shows up in star error and probability, not ranking.
- Top-10 recommendations for heavy/moderate/sparse users now carry **specific-tag
  explanations with evidence counts** ("because you loved garlic recipes (14 rated)"),
  raw + calibrated confidence, plus vegetarian and ≥70%-pantry constrained lists and a
  cold-candidate demo (all persisted, see Artifacts).

## Deviations from the written methodology (stated plainly)

- Hyperparameters are tuned on an **inner warm holdout**, not the provided validation
  split: the provided val/test are 100% item-cold, so validation-RMSE would rank configs
  on bias terms alone. Both provided splits remain untouched report-only sets.
- **PyMC validates** the conjugate updater rather than performing the updating.
- Nutrition features = **calorie terciles only** (not full macros).
- The combined Bayesian prediction is a **labeled heuristic ensemble**, not a coherent
  posterior; the coherent alternatives (single generative model) are future work.
- Family/tag preference layers add **interpretability, not measured predictive lift**,
  over the user-generosity baseline on warm history.

## Promise-vs-delivery checklist

- SVD collaborative baseline (Surprise), RMSE+MAE vs naive baselines — **realized**.
- Ranking metrics with a defensible relevance definition and CIs — **realized**.
- Continuously updating preference distributions (sequential replay) — **realized**.
- Preferences over specific cuisines/tags/dietary categories — **realized for
  explanations; partially realized for prediction** (no measured lift over families).
- Hybrid = weighted normalized SVD + Bayesian score, weights tuned — **realized** (w\*
  validated on validation; probability-space variant added).
- Confidence from posterior uncertainty — **realized** (CI-width, calibrated variant).
- Explanations from strongest matching features — **realized** (tag-level, evidence-cited).
- Competitive top-N *ranking* vs popularity — **not realized**: matched (pop-blend), not
  beaten; exposure modeling is future work.
- PyMC Bayesian updating — **partially realized by design** (validator, stretch goal).

## Artifacts (all under `outputs/`)

- `hannah_svd_params.json` — tuned SVD params, grid provenance, temporal-split settings,
  75.1% leakage stat. `models/hannah_svd.pkl`, `models/hannah_baseline_only.pkl`.
- `hannah_svd_preds_{warm_holdout,warm_holdout_temporal,validation,test}.csv` — per-row
  `svd_pred`, `bias_pred`, and `in_eval` flag (False = rating-0 row, excluded from metrics).
- `hannah_ranking_metrics.csv` (nb02) and `hannah_ranking_metrics_hybrid.csv` (nb04, 14
  rows: warm + pop-blend + cold) — HR/NDCG@10 with 95% CIs and protocol strings.
- `hannah_user_posteriors.csv` (24,961 users × 16 cols), `hannah_user_tag_posteriors.csv`
  (496,796 rows), `hannah_recipe_quality.csv` (159,131 recipes, S_ITEM=20),
  `hannah_bayes_params.json` (authoritative priors: S_USER=5, S_ITEM=20, S_TAG=5, p̄=0.760).
- `hannah_hybrid_metrics.csv` (every stars/probability table, long format),
  `hannah_calibration.csv` (deciles + raw/recalibrated ECE).
- `recommendations/hannah_topn_examples.csv` (30 rows, with `confidence_calibrated`),
  `recommendations/hannah_topn_constrained.csv` (vegetarian + pantry demos).

## Verification

Pipeline order 02 → 03 → 04 (03 needs only Data/ + src/). 32 unit tests pass
(`tests/`, run with the pinned anaconda env in `requirements.txt`). Cross-artifact
identities asserted at load time in nb04 (pseudo-count identities from
`hannah_bayes_params.json`, pooled-rate recount, feature-module version print). A
14-point post-run artifact audit (schemas, row counts, metric reproduction from saved
predictions) passed in full.
