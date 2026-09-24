# Pipeline Specifications

## Pipeline 1: Data Pipeline (Scheduled, Daily)

* **Trigger:** GitHub Actions scheduled workflow, `cron: '0 23 * * *'` (23:00 UTC = 2:00 EAT)

### Steps

1. **Fetch raw data**
   * Download latest WFP Kenya Food Prices dataset (HDX CKAN direct CSV).
   * Download Open-Meteo Historical Archive data (daily rainfall + temperature) for the five market coordinates (Nairobi, Nakuru, Mombasa, Kisumu, Eldoret), using a fixed `market → lat/lon` lookup table.

2. **Combine**
   * Join WFP price records to Open-Meteo climate records on (`market`, `date`), using the `market → coordinates` mapping.

3. **Clean**
   * Forward-fill short gaps in price data.
   * IQR-based outlier flagging.
   * Drop/flag records with unresolvable missingness.

4. **Aggregate**
   * Aggregate daily climate values to monthly (mean rainfall, mean temperature) per market.
   * Aggregate price records to monthly per (`market`, `crop`).

5. **Feature engineering**
   * Lag features ($t_{-1}$, $t_{-2}$ monthly prices), rolling averages, seasonal/month indicators.
   * Climate features included as candidate columns (monthly avg rainfall, monthly avg temperature) — flagged for the model training stage to evaluate via feature importance/ablation, not assumed useful at this stage. The pipeline's job is to make them available, not to decide their value.

6. **Store**
   * Upsert into a Supabase `features` table, keyed by (`market`, `crop`, `month`) — idempotent, so reruns don't duplicate rows.

> **Scope Boundary:** Model retraining is explicitly **not included** in this job. This pipeline only refreshes the feature store.

---

## Pipeline 2: Forecast Pipeline (Batch, Chained After Pipeline 1) + Farmer-Facing Inference (Real-Time)

*This system consists of two distinct components: a nightly batch generation job and a live read path.*

### 2a. Nightly Batch Forecast Generation
* **Execution:** GitHub Actions (same run or immediately following Pipeline 1).
* **Model State:** Load the already-trained stacking ensemble model (Random Forest + XGBoost base learners, Linear Regression meta-learner) — trained separately during your Model Training sprint, not retrained here.

**Execution steps for each of the 30 combinations (5 markets × 3 crops × 2 horizons):**
1. Pull the latest feature vector from the Supabase `features` table.
2. Run `model.predict()` $\rightarrow$ price forecast.
3. Run SHAP `TreeExplainer` on each base learner $\rightarrow$ weight/average by meta-learner coefficients $\rightarrow$ SHAP contribution values.
4. Upsert results into a Supabase `forecasts` table, keyed by (`market`, `crop`, `horizon`, `generated_date`):
   ```json
   {
     "forecast_value": "...",
     "shap_values": "...",
     "generated_at": "..."
   }