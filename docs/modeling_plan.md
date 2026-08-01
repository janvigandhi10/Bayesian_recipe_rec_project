# Modeling Plan

## Baseline

Start with a simple baseline before SVD:

- global mean rating
- user mean rating
- recipe mean rating

This gives a sanity-check benchmark.

## Collaborative Filtering

Train SVD on `interactions_train.csv`.

Evaluate on:

- `interactions_validation.csv`
- `interactions_test.csv`

Primary metrics:

- RMSE
- MAE

Optional ranking metrics:

- Precision@K
- Recall@K
- NDCG@K

## Bayesian Layer

A practical first version can model feature-level user preferences.

Example:

- Convert ratings into positive/negative preference signals.
- Track user preference over recipe tags/cuisines/dietary categories.
- Use Beta-Binomial updating for interpretable feature probabilities.
- Use posterior mean as preference score.
- Use posterior variance or credible interval width as uncertainty.

This can be expanded into a PyMC model if time allows.

## Hybrid Score

Combine:

```text
hybrid_score = svd_weight * normalized_svd_score + bayesian_weight * bayesian_preference_score
```

Then return:

- top-N recipes by hybrid score
- confidence from Bayesian posterior uncertainty
- explanation from strongest matching recipe features

