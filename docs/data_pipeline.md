# FEWS Monthly Data Pipeline

This stage fetches observed Kenya wholesale maize/beans prices and writes only
`market_prices`. It never loads models, builds features, forecasts, fills gaps, or
changes the database schema. Python 3.13 and the requested dependency pins are
listed in `requirements-pipeline.txt`; inference packages are not used here.

## Configuration and Discovery

Configuration lives in `src/agrisence/config/pipeline_config.py`: July 2025 start,
Nairobi/Kisumu enabled, maize/beans, all four market IDs, and gate/warning thresholds.
Enable Mombasa/Nakuru by adding names to `ENABLED_MARKETS`. Mombasa's ID has only
been checked for beans: verify maize first with discovery's `--all-markets` mode.

With the package installed or `src` on `PYTHONPATH`:

```text
python scripts/discover_fews_products.py --all-markets
python -m agrisence.pipelines.run_data_pipeline --no-db
python -m agrisence.pipelines.run_data_pipeline --dry-run
python -m agrisence.pipelines.run_data_pipeline
```

Discovery needs no secrets/database. Review product groups and explicit
missing-month lists in `artifacts/fews_discovery.json`, then fill
`src/agrisence/config/fews_products.yaml` with exact product/cpcv2 pairs. Both crop
lists intentionally start empty and block ingestion until reviewed. Beans varieties
share one crop allow-list across all markets. Multiple records are averaged within
a variety/month before averaging equally across varieties; only FEWS KES/kg is
accepted and Decimal half-up rounding produces two-decimal prices.

## Modes and Writes

`--no-db` writes `artifacts/fews_monthly_preview.csv` without a database connection.
Its gates use incoming observed months only. `--dry-run` reads and reports planned
inserts/updates/unchanged counts without writes; gates show the projected union of
stored and incoming observed months. No flag applies upserts and gates stored
months on the same connection after writing. All modes save `artifacts/run_report.json`.
The summary distinguishes planned changes from actual inserted/updated counts.

Every run fetches the complete window from the configured start to today, rather
than advancing a last-run cursor. `--start-month YYYY-MM-01` overrides that lower
bound in ingestion and discovery. If any enabled series has no usable rows, or
fetching/validation fails, no preview, report, or database writes occur. All price
upserts share one transaction; failures including post-write gate reads roll back
the entire batch. Reports are saved after a successful database transaction; a
filesystem error cannot roll back a transaction already committed. No missing
month is interpolated or carried forward. Variety changes and absolute monthly
log moves above 0.30 are warnings, not blockers.

Use the limited `pipeline_writer` connection only via `PIPELINE_DATABASE_URL` in
the environment or repository-root ignored `.env`; `.env.example` has an empty
placeholder. Prepared statements are disabled for the Supabase transaction pooler.
No credentials or URLs are logged. Only price SELECTs and the conflict-aware
upsert are issued: there are no deletes, schema changes, or forecast writes.

## Month and Gate Definitions

Prices belong to `start_date` floored to the first of the coverage month, not the
fetch date. Non-first-day dates get diagnostics; future months are excluded.
The gate counts consecutive calendar months backwards from the latest observed
month, stopping at the first gap. Eligibility needs at least 13 such months and
latest-month age at most 2 calendar months. Empty or future-only series are not
eligible. Skipped series do not fail a successful ingestion run.

## Exit Codes and Tests

Exit 0 means success even with ineligible series; 1 means fetch, validation,
database, or output failure; 2 means configuration/CLI failure. Tests use
synthetic records and mocked HTTP only. Real-database tests must use `@pytest.mark.db`
and are skipped unless explicitly enabled with `--run-db`.