# Forecast Pipeline

The inference stage loads the committed H1/H2 stacking ensembles, reads observed
monthly prices, reproduces training features using `preprocessing.feature_engineering`,
computes batch stack forecasts and exact grouped SHAP explanations, then optionally
upserts rows to `forecasts`. It never trains or modifies a model and writes no other
database table. SHAP keeps the existing additive reconciliation assertion unchanged.

## Model Files, Integrity, and Checksums

Inference loads `xgb_h1.json`/`xgb_h2.json` and
`stack_h1_sklearn.joblib`/`stack_h2_sklearn.joblib` from
`results/artifacts/models/`. XGBoost JSON is used because it is portable between
XGBoost runtime versions. The original `stacking_h1.pkl` and `stacking_h2.pkl` files
are archived and are not used by inference because their embedded XGBoost streams
do not deserialize reliably across environments. Joblib artifacts can execute code
when loaded; load only checksum-verified files from this repository.

Before either format is loaded, the inference loader verifies
`results/artifacts/models/MODEL_CHECKSUMS.json`. It requires scikit-learn 1.6.1; an
XGBoost version other than 3.4.1 emits a warning because the JSON model is portable.
Golden parity tests verify that RF, XGBoost, Ridge, and exact SHAP predictions remain
within their stated tolerances. Git attributes mark model formats as binary so line
ending conversion cannot alter their digests.

The checksum generator hashes the four portable model files, not the golden CSVs.
Run it manually after verifying intentionally created/replaced model artifacts, and
commit its resulting manifest with those artifacts:

```text
python scripts/write_model_checksums.py
```

The generator hashes bytes but does not deserialize, train, or change model files.
If the manifest is missing, corrupt, incomplete, or a digest differs, inference
refuses to load the artifact. A loaded result contains only RF, XGBoost, Ridge,
feature columns, and horizon.

### Regenerate Portable Artifacts in Colab

Only run this when the original training bundles are known-good. It creates a
separate `repackaged` directory, leaving the originals untouched. Run in the
training Colab runtime with Drive mounted:

```python
import hashlib
import joblib
import numpy as np
import pandas as pd
from pathlib import Path
import xgboost as xgb

source_dir = Path("/content/drive/MyDrive/capstone-data/models")
output_dir = source_dir / "repackaged"
output_dir.mkdir(exist_ok=True)
feature_dir = Path("/content/drive/MyDrive/capstone-data/feature_engineering")

for horizon in ("h1", "h2"):
	original = joblib.load(source_dir / f"stacking_{horizon}.pkl")
	columns = original["feature_cols"]
	original["xgb_model"].save_model(str(output_dir / f"xgb_{horizon}.json"))
	joblib.dump(
		{key: original[key] for key in ("rf_model", "ridge_meta", "feature_cols", "target_col", "horizon")},
		output_dir / f"stack_{horizon}_sklearn.joblib",
	)

	features = pd.read_csv(feature_dir / f"model_df_{horizon}.csv")[columns].astype("float64")
	features = features.sample(30, random_state=0)
	rf = original["rf_model"].predict(features)
	xgb_pred = original["xgb_model"].predict(features)
	stack = original["ridge_meta"].predict(np.column_stack([rf, xgb_pred]))
	golden = features.copy()
	golden["rf_pred"], golden["xgb_pred"], golden["stack_pred"] = rf, xgb_pred, stack
	golden.to_csv(output_dir / f"golden_{horizon}.csv", index=False)

	portable = xgb.XGBRegressor()
	portable.load_model(output_dir / f"xgb_{horizon}.json")
	difference = np.abs(portable.predict(features) - xgb_pred).max()
	print(horizon, "XGBoost JSON max abs difference:", difference)
	assert difference < 1e-6

for path in sorted(output_dir.iterdir()):
	print(path.name, path.stat().st_size, hashlib.sha256(path.read_bytes()).hexdigest())
```

Review the exports and golden predictions before copying the four portable model
artifacts and two golden CSVs into `results/artifacts/models/`; then run the checksum
generator and commit the updated manifest. Do not replace archived originals as part
of this process.

## Origins, Features, and Explanations

Enabled markets/crops and the minimum price window come from
`config/pipeline_config.py`. `plan_origins` first applies the shared 13-consecutive-
month and 2-month-freshness gate to the latest series. A failing series is skipped
as a whole. The latest month is always the live origin. Earlier origins are eligible
for backfill only when their own trailing history has at least 13 consecutive months
and at least one horizon row is missing; origin age does not limit backfill.

For each origin, prices are truncated there, reindexed to a complete month calendar,
then passed through the exact training functions `build_features` and
`apply_calendar_distance_guard` in that order. Missing/non-finite features fail with
`FeatureError`; no imputation or formula reimplementation is used. The RF and XGBoost
predictions are passed in that order to Ridge. One exact TreeExplainer stack SHAP
batch is computed per horizon; grouped explanations are reconciled to every stack
prediction. Non-positive prices and absolute log changes over 0.7 fail the run;
absolute changes over 0.3 warn. The explanation origin price must equal the stored
price at the origin.

## Idempotency and Writes

Each live horizon row is inserted when absent and updated only when its quantized
prices/log change or sorted JSON explanation differs. A live computation never
overwrites a stored backfill. Backfills are inserted only once; existing backfill
rows are never changed. Keys are never updated and no rows are deleted. All forecast
writes use one transaction with `Jsonb` explanations and decimal quantization.

## Run Modes and Exit Codes

```text
python -m agrisence.pipelines.run_forecast_pipeline --no-db --prices-csv PATH
python -m agrisence.pipelines.run_forecast_pipeline --dry-run
python -m agrisence.pipelines.run_forecast_pipeline
```

`--no-db` uses a CSV with `market`, `crop`, `month`, and `avg_price_per_kg`, assumes
no existing forecasts, and writes `artifacts/forecasts_preview.json` and
`artifacts/forecasts_preview.csv`. `--dry-run` reads prices and existing forecasts,
prints the plan, and writes nothing. No flag applies the plan. `--today YYYY-MM-DD`
overrides the date for testing. A successful applied run writes
`artifacts/forecast_run_report.json`; forecast table writes happen before that file,
so a later filesystem failure cannot undo an already committed database transaction.

Exit 0 means success, including skipped series; exit 1 means fetch, feature, inference,
validation, or database failure; exit 2 means configuration/model integrity failure.
The final table reports each forecast action and the summary counts eligible/skipped
series and inserted/updated/unchanged rows. Task 2a also reports counts of products
ignored by its allow-list; this diagnostic does not alter ingestion results.