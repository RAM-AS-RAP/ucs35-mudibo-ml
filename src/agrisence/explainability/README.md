# Explainability

This folder contains the model interpretation layer used to explain forecast drivers. It is centered on SHAP analysis so feature contributions can be communicated to farmers, project stakeholders, and technical reviewers.

## Files

- `shap_explainer.py` — computes SHAP values for trained forecast models using TreeExplainer.
- `__init__.py` — marks the package for import.
