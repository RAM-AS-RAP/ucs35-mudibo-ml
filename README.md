# AgriSence

AgriSence forecasts monthly wholesale prices for maize and beans across Nairobi,
Kisumu, Mombasa, and Nakuru.

## Project layout

```text
AgriSence/
├── README.md
├── .gitignore
├── requirements.txt
├── notebooks/                 # exploration, cleaning, training, and evaluation
├── src/agrisence/             # reusable ingestion, preprocessing, training, and inference code
├── results/                   # metrics, plots, CSV outputs, and model artifacts
└── tests/                     # automated tests for reusable code
```

## Data and artifacts

- Source data: WFP Kenya food-price export, loaded in the notebooks from the HDX URL.
- Cleaned panel: `results/artifacts/cleaned_monthly_prices.csv`.
- Serialized model files: `results/artifacts/models/`.
- Evaluation metrics and visual outputs: `results/`.

## Setup

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
pip install -e .
python -m pytest
```

Open notebooks from `notebooks/`. Import reusable code through the
`agrisence` namespace, for example:

```python
from agrisence.preprocessing.cleaning import normalize_units
```

