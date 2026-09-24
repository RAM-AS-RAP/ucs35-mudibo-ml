# Notebooks

This folder contains exploration, data-cleaning, training, and evaluation notebooks.
Notebooks should import reusable code from `agrisence` and write generated outputs to
`../results/`, not back into `src/`.

## Current notebooks

- `01_initial_eda.ipynb` and `01_detailed_eda.ipynb`: data exploration.
- `02_data_cleaning.ipynb`: monthly panel construction and cleaning audit.
- `02_baseline_comparison.ipynb`: baseline model comparison.

## Local setup

Install the project from the repository root before running a notebook:

```powershell
pip install -r requirements.txt
pip install -e .
```

Use relative paths rooted at the repository, and save generated metrics, figures, CSV
files, and serialized models under `results/`.