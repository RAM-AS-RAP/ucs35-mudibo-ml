# Notebooks

This folder contains exploration, data-cleaning, training, and evaluation notebooks.
Notebooks should import reusable code from `agrisence` and write generated outputs to
`../results/`, not back into `src/`.

## Current notebooks

- `01_initial_eda.ipynb` and `01_detailed_eda.ipynb`: data exploration.
- `02_data_cleaning.ipynb`: monthly panel construction and cleaning audit.
- `02_baseline_comparison.ipynb`: baseline model comparison.
- `04_feature_engineering.ipynb`: builds ratio features, the exact-calendar H1 target,
  observed-endpoint supervised tables, and expanding-window fold definitions without
  fitting models.
- `05_model_training.ipynb`: trains development-only persistence and linear baselines,
  tunes Random Forest and XGBoost with chronological folds, and fits Ridge on OOF base
  predictions.
- [06_final_evaluation.ipynb](06_final_evaluation.ipynb): refits frozen
    development models, evaluates the one-month (H1) held-out set once, and reports Linear
    Regression and the stacking ensemble, pooled over all held-out rows, in two tables:
    log-change MAE/MSE/RMSE/R2 and reconstructed-price MAE/MSE/RMSE/MAPE. Random Forest and
    XGBoost are trained as stack components but not reported separately. Persistence remains
    a training baseline only. Loads Task 2's combined `tuning_h1.json` file.
    Imports the final-evaluation workflow from `training/evaluate.py`, which calls the shared
    refit/prediction functions in `training/ensemble/stacking.py`. No functions are defined in
    the notebook; its cells configure paths, call the workflow, and display results. The cloned
    branch must include the Task 3 module changes before execution. Colab reads inputs directly
    from Drive `capstone-data/feature_engineering/` and `capstone-data/model_training/`, using
    the same environment-file paths as Task 2. Saves result/prediction CSVs under
    `results/artifacts/` and complete bundles under `results/artifacts/models/`, mirrored to
    Drive `capstone-data/final_evaluation/` and `capstone-data/models/` in Colab. Closure
    markers block repeat test inference; inspect saved CSVs instead of deleting markers or
    rerunning evaluation; the re-scoring cell recomputes both results tables from the saved
    `test_predictions_h1.csv` without refitting or predicting. MAPE is in percent, and price
    MAE/RMSE in KES/kg.
- [07_shap_explainability.ipynb](07_shap_explainability.ipynb): explains the saved H1 final
    stack on the matching held-out feature rows using exact Ridge-weighted TreeExplainer
    values. It loads models, features, and saved predictions from Drive and does not refit or
    rerun forecasts. Saves per-row SHAP values, mean-absolute-SHAP summaries, grouped farmer
    explanations (`shap_grouped_h1.json`), and beeswarm plots under
    `results/artifacts/`, mirrored to `MyDrive/capstone-data/explainability/`.

## Local setup

Install the project from the repository root before running a notebook:

```powershell
pip install -r requirements.txt
pip install -e .
```

Use relative paths rooted at the repository, and save generated metrics, figures, CSV
files, and serialized models under `results/`.