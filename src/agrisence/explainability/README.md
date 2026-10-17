# Explainability

This folder contains the model interpretation layer used to explain forecast drivers. It is centered on SHAP analysis so feature contributions can be communicated to farmers, project stakeholders, and technical reviewers.

## Files

- `shap_explainer.py` — combines Random Forest and XGBoost TreeExplainer values using the
	fitted Ridge coefficients to produce exact additive explanations for the stacking forecast.
	Groups all features into complete farmer-facing categories, reconstructs price changes,
	and verifies grouped contributions reconcile with the forecast. Exact decomposition
	requires a linear Ridge meta-model.
- Install SHAP from the root `requirements.txt` before running explainability workflows.
- H1 explanations are computed only from the held-out feature matrices corresponding to
	the saved final-evaluation forecasts; the implementation does not refit or retune models.
- `__init__.py` — marks the package for import.
