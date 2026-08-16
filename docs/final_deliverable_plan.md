# Final Deliverable Plan

## Recommended Canonical Pipeline

Use Hannah's notebooks as the final modeling spine:

1. `notebooks/final/01_data_exploration.ipynb`
2. `notebooks/final/02_svd_baseline.ipynb`
3. `notebooks/final/03_bayesian_updating.ipynb`
4. `notebooks/final/04_hybrid_recommender.ipynb`

Fold in Hoang's EDA figures and simpler SVD baseline as supporting material:

- `notebooks/supporting/hoang_data_cleaning_eda.ipynb`
- `notebooks/supporting/hoang_svd_baseline.ipynb`
- `reports/figures/`
- `reports/model_results/svd_baseline_metrics.csv`

## What Is Done

- Rating `0` policy chosen: exclude from training/evaluation.
- EDA notebooks and plots exist from both Hoang and Hannah.
- SVD baseline exists, including tuned Surprise SVD and naive/bias-only
  comparisons.
- Bayesian updating exists with Beta-Binomial user/family/tag posteriors.
- Hybrid model exists with validated blending weight and saved metrics.
- Confidence and calibration analysis exist.
- Recommendation explanations exist through tag-level posteriors.
- Streamlit demo exists at `streamlit_app.py`.
- Unit tests pass with `PYTHONPATH=.`.

## Cleanup Completed

- Final notebooks moved to `notebooks/final/`.
- Hoang's useful EDA/SVD notebooks moved to `notebooks/supporting/`.
- Removed original one-cell scaffold notebooks for SVD/Bayesian/hybrid work.
- Removed Hoang's empty Bayesian placeholder notebook.

## What Still Needs Final Submission Work

- Decide which EDA tables/plots go into the final report.
- Reconcile Hoang's simpler SVD numbers with Hannah's fuller evaluation
  protocol.
- Write the final report sections:
  - motivation and problem statement
  - dataset and preprocessing
  - collaborative filtering method
  - Bayesian updating method
  - hybrid recommendation method
  - results and limitations
- Make one clean final notebook or appendix that points to the canonical
  notebooks.
- Add screenshots of the Streamlit demo if the presentation needs visuals.

## Honest Result Framing

- Hybrid improves cold rating prediction versus SVD in Hannah's evaluation.
- Bayesian tag preferences are most useful for explanations and cold-start
  signal.
- Popularity remains hard to beat for top-N ranking.
- The hybrid score is a practical ensemble, not a single coherent Bayesian
  posterior.

## Demo Command

```powershell
$env:PYTHONPATH='.'
python -m streamlit run streamlit_app.py
```
