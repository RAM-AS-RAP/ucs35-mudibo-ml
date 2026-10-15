# ucs35-mudibo-ml
## Project Title: SHAP-based explainable decision support system for crop price prediction and farmer-buyer linkage among smallholder farmers in Kenya
### Name: Sean Mudibo
### Adm No: 168329
### Capstone Project (Webmasters Kenya)

- Backend repository: https://github.com/RAM-AS-RAP/ucs35-mudibo-backend
- React PWA repository: https://github.com/RAM-AS-RAP/ucs35-mudibo-frontend

## Repository map

```text
AgriSence/
├── documentation/
│   ├── model/                  # Modeling analyses and pipeline documentation
│   │   └── Pipeline/           # Data ingestion, forecast inference, and operations guides
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
│   └── pipelines/              # FEWS data and gated forecast orchestrators
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
	`feature_engineering.py` is the single source for model features.
- `src/agrisence/training/` contains baseline, Random Forest, XGBoost, stacking, and
	evaluation modules. `src/agrisence/pipelines/` contains the data and forecast workflows.
- `results/artifacts/` stores the cutoff-specific WFP panel, archived source artifacts,
	and the destination for trained models. Metrics, figures, and predictions belong in
	task-specific subfolders under `results/`.
- `documentation/model/cleaning/analysis.md` records cleaning decisions and the current
	ratio-target modeling protocol. `documentation/model/pipeline.md` describes workflow
	architecture; `documentation/model/Pipeline/` contains operational guides.

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

### Set up and run the pipelines

The automated pipeline environment uses Python 3.13 and the pins in
`requirements-pipeline.txt`. Install them in the active virtual environment and expose
the source-layout package from the repository root:

```powershell
python -m pip install -r requirements-pipeline.txt
$env:PYTHONPATH = "$PWD\src"
```

Review the product/code pairs in `src/agrisence/config/fews_products.yaml` before
ingestion. Empty crop allow-lists stop the data pipeline. Product discovery does not
need a database:

```powershell
python scripts/discover_fews_products.py --all-markets
```

For local database modes, put `PIPELINE_DATABASE_URL` in a repository-root `.env`
file. `.env` is ignored and must never be committed. GitHub Actions reads the same
variable only from the repository secret named `PIPELINE_DATABASE_URL`.

Run the data stage from the repository root:

```powershell
python -m agrisence.pipelines.run_data_pipeline --no-db
python -m agrisence.pipelines.run_data_pipeline --dry-run
python -m agrisence.pipelines.run_data_pipeline
```

`--no-db` writes a FEWS preview without connecting to Supabase; `--dry-run` plans
against existing prices but writes nothing; no flag applies changed price upserts.
Forecast inference supports the same modes:

```powershell
python -m agrisence.pipelines.run_forecast_pipeline --no-db --prices-csv artifacts/fews_monthly_preview.csv
python -m agrisence.pipelines.run_forecast_pipeline --dry-run
python -m agrisence.pipelines.run_forecast_pipeline
```

The forecast loader requires the portable JSON/joblib models in
`results/artifacts/models/` and matching entries in `MODEL_CHECKSUMS.json`. The
original pickle files are archived and are not loaded by inference. See
`documentation/model/Pipeline/data_pipeline.md`,
`documentation/model/Pipeline/forecast_pipeline.md`, and
`documentation/model/Pipeline/operations.md` for product review, modes, checksums,
gates, database safety, reports, and scheduled workflow operations. GitHub Actions
runs the pipeline daily at 03:17 UTC; manual dispatch defaults to dry-run.

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

Database tests are skipped unless explicitly enabled with `--run-db`. The test suite
includes synthetic, mocked, feature-parity, and portable-model golden-parity checks.

### Import package utilities

After the editable install, reusable code is available from the `agrisence` package:

```python
from agrisence.preprocessing.cleaning import normalize_units
from agrisence.preprocessing.aggregation import aggregate_to_monthly
```

The data pipeline writes only market prices; forecast inference writes only forecasts.
Both workflows preserve missing months rather than filling or carrying values forward.

## Data and provenance

- **WFP source:** Kenya food-price export hosted by the Humanitarian Data Exchange.
- **Canonical experiment panel:** `cleaned_monthly_prices_wfp_2010_2019-10.csv`, with
	WFP history ending October 2019 and provenance retained for observed and imputed rows.
- **FEWS NET exports:** stored under `results/artifacts/fews_raw/` for inference-related
	work. They are not training targets or primary evaluation labels in the WFP experiment.
- **Models:** `results/artifacts/models/` contains portable XGBoost JSON and sklearn
	joblib inference artifacts, checksums, golden parity fixtures, and archived original
	pickle files that inference does not load.

Never commit credentials. `.env.example` contains only the empty
`PIPELINE_DATABASE_URL` placeholder. The Colab notebook bootstrap reads `GITHUB_TOKEN`
or `GH_TOKEN` from Colab Secrets when it needs to clone the private repository; keep
real values in secrets or a local ignored `.env`, never in tracked files.

