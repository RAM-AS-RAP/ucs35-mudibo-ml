# AgriSence

AgriSence is a capstone project exploring monthly wholesale crop-price forecasting in
Kenya. The current WFP experiment focuses on maize and beans in Nairobi, Kisumu,
Mombasa, and Nakuru. Prices are normalized to Kenyan shillings per kilogram and
aggregated to monthly observations.

The documented modeling experiment uses WFP data. The canonical
panel retains earlier WFP history as feature warm-up and preserves the provenance of
observed and short-gap-filled values. Forecast targets are cumulative log price changes
one and two months ahead; forecasts can be reconstructed to KES prices for evaluation
and display. FEWS NET data is reserved for inference-time use, not training or primary
evaluation.

> **Project status:** the cleaning notebook and reusable preprocessing utilities are in
> place. Feature engineering, model training, tuning, and evaluation are still under
> development. The top-level scheduled data and forecast pipeline modules are scaffolds,
> not operational command-line workflows.

## Repository map

```text
AgriSence/
├── documentation/
│   ├── model/                  # Cleaning analysis and pipeline design
│   └── tasks/                  # Project task specifications and working notes
├── model/                      # Legacy/placeholder model workspace
├── notebooks/                  # EDA, cleaning, baseline comparison
├── results/
│   ├── artifacts/
│   │   ├── cleaned_monthly_prices_wfp_2010_2019-10.csv
│   │   ├── fews_raw/           # Archived FEWS exports; not WFP training labels
│   │   └── models/             # Destination for serialized model artifacts
│   └── README.md               # Output organization guidance
├── src/agrisence/
│   ├── config/                 # Markets, crops, horizons, and settings
│   ├── ingestion/              # WFP, FEWS, and Open-Meteo fetchers
│   ├── preprocessing/          # Cleaning, monthly aggregation, feature engineering
│   ├── training/               # Baselines, tree models, stacking, evaluation
│   ├── inference/              # Model loading and batch forecast helpers
│   ├── explainability/         # SHAP explanation utilities
│   ├── storage/                # Supabase client integration
│   └── pipelines/              # Planned data and forecast orchestrators
├── tests/                      # Pytest checks for package functionality
├── pyproject.toml              # Package metadata and pytest configuration
└── requirements.txt            # Runtime and development dependencies
```

### Main areas

- `notebooks/01_initial_eda.ipynb` and `01_detailed_eda.ipynb` explore the WFP source
	and series coverage.
- `notebooks/02_data_cleaning.ipynb` builds the monthly WFP panel, audits gaps and
	provenance, and saves the canonical CSV to Google Drive when run in Colab.
- `notebooks/02_baseline_comparison.ipynb` contains the current baseline-comparison
	work.
- `src/agrisence/preprocessing/cleaning.py` contains reusable unit, filtering, overlap,
	calendar, gap, and diagnostic utilities. `aggregation.py` creates monthly prices;
	`feature_engineering.py` is the planned home for model features.
- `src/agrisence/training/` contains baseline, Random Forest, XGBoost, stacking, and
	evaluation modules. Several entry points are currently scaffolds.
- `results/artifacts/` stores the cutoff-specific WFP panel, archived source artifacts,
	and the destination for trained models. Metrics, figures, and predictions belong in
	task-specific subfolders under `results/`.
- `documentation/model/cleaning/analysis.md` records cleaning decisions and the current
	ratio-target modeling protocol. `documentation/model/pipeline.md` describes planned
	scheduled workflows.

## Requirements

- Python 3.10 or newer
- Git
- Internet access to download the WFP Kenya export when rerunning the cleaning notebook
- For the notebook's Drive save: Google Colab with Google Drive mounted

The project uses pandas, NumPy, scikit-learn, XGBoost, SHAP, statsmodels, requests, and
the plotting/test packages listed in `requirements.txt`.

## Setup

Run these commands from the repository root.

### Windows PowerShell

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pip install -e .
```

If PowerShell blocks virtual-environment activation, use the Python executable directly:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m pip install -e .
```

### macOS or Linux

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pip install -e .
```

To use the environment as a notebook kernel, install `ipykernel` into the environment
if it is not already available, then select that Python environment in VS Code or Jupyter:

```bash
python -m pip install ipykernel
```

## Run the project

### Explore and clean data

Open the notebooks in VS Code or Jupyter and run them in order:

1. `notebooks/01_initial_eda.ipynb`
2. `notebooks/01_detailed_eda.ipynb`
3. `notebooks/02_data_cleaning.ipynb`
4. `notebooks/02_baseline_comparison.ipynb`

The cleaning notebook downloads the WFP export from HDX. Its final save is Drive-only:
run it in Colab, authorize Drive mounting, and find the output at
`MyDrive/capstone-data/cleaned_monthly_prices_wfp_2010_2019-10.csv`. It intentionally
does not write this output into the repository. A repository copy should be added under
`model/artifacts/` when preparing a reproducible project snapshot.

### Run tests

```bash
python -m pytest
```

Some tests and model modules are still minimal or scaffolded; a passing test run should
not be interpreted as validation of a complete forecasting workflow.

### Import package utilities

After the editable install, reusable code is available from the `agrisence` package:

```python
from agrisence.preprocessing.cleaning import normalize_units
from agrisence.preprocessing.aggregation import aggregate_to_monthly
```

The modules `agrisence.pipelines.run_data_pipeline` and
`agrisence.pipelines.run_forecast_pipeline` currently contain placeholder orchestration
functions. They are not ready to run production ingestion or forecasts. Likewise,
feature generation, training, hyperparameter tuning, and evaluation need implementation
before the planned Linear Regression, Random Forest, XGBoost, and Ridge stacking
experiments can be run end to end.

## Data and provenance

- **WFP source:** Kenya food-price export hosted by the Humanitarian Data Exchange.
- **Canonical experiment panel:** `cleaned_monthly_prices_wfp_2010_2019-10.csv`, with
	WFP history ending October 2019 and provenance retained for observed and imputed rows.
- **FEWS NET exports:** stored under `results/artifacts/fews_raw/` for inference-related
	work. They are not training targets or primary evaluation labels in the WFP experiment.
- **Models:** `results/artifacts/models/` is the intended location for serialized model
	files; trained production models are not yet provided.

Never commit credentials. `.env.example` documents the `GITHUB_TOKEN` variable used by
the Colab notebook bootstrap when cloning the private repository; keep real tokens in a
local `.env` or Colab Secrets, and do not add `.env` to version control.

