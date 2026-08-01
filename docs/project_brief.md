# Project Brief

## Title

Bayesian Personalized Meal Recommendation System

## Objective

Build a recommender system that predicts whether a user will enjoy a recipe and reports uncertainty/confidence in that prediction.

## Method Summary

The planned approach is a hybrid model:

- **SVD collaborative filtering** learns from historical ratings across users and recipes.
- **Bayesian updating** adjusts recommendations based on a user's evolving preferences and provides uncertainty estimates.

## Dataset

Food.com Recipes and User Interactions dataset from Kaggle.

The local project includes raw recipe metadata, raw interactions, and train/validation/test interaction splits.

## Expected Output

For a selected user, the system should return top recipe recommendations with:

- recipe name
- predicted rating or enjoyment probability
- Bayesian preference adjustment
- confidence/uncertainty score
- optional explanation based on recipe features

