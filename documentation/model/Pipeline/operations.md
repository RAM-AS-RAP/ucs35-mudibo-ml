# Pipeline Operations

The scheduled workflow runs both FEWS ingestion and forecast generation daily at
03:17 UTC (06:17 East Africa Time). Most runs write nothing: price upserts and live
forecast updates are idempotent, and backfills are inserted once.

## Monitoring

- **Actions tab:** inspect the `pipeline` run and identify the failed step.
- **Step summary:** review per-series gate status and insert/update/unchanged counts.
- **Report artifact:** download `pipeline-reports` for run reports and any generated previews; artifacts are retained for 14 days.
- **Failure issue:** scheduled failures open or update an issue labelled `pipeline-failure`. Its body contains only the run URL, date, and failing step.

## Outcomes

- **Success, all series eligible:** the run completed; rows are written only when new or changed.
- **Success, a series skipped:** its history did not meet the freshness/continuity gate. A late FEWS observation can make it eligible on the next daily run.
- **Red run:** investigate the failing FEWS request, validation, model/inference, or database step in Actions. Do not put database URLs or logs containing sensitive values into issues.

## Manual Dry Run and Secret Rotation

To test the configured secret and network without writing database rows, open **Actions → pipeline → Run workflow** and choose `dry-run`. Scheduled runs always apply. The only database credential is the Actions repository secret `PIPELINE_DATABASE_URL`; it is passed only to the secret check and pipeline steps and is never included in reports or artifacts.

To rotate credentials, set a new password for the limited `pipeline_writer` role in Supabase, update the repository's `PIPELINE_DATABASE_URL` Actions secret, then run the workflow once with mode `dry-run`.

## Public Repository Readiness

Before making the repository public:

1. Confirm no secrets exist in tracked files or repository history.
2. Confirm the database connection string exists only as the Actions `PIPELINE_DATABASE_URL` secret.
3. Confirm `.github/workflows/keepalive.yml` is present and enabled.

GitHub may disable scheduled workflows in a public repository after 60 days without
activity. The keepalive runs monthly only when the repository is public and may push
an empty commit after 30 inactive days. GitHub does not document every event that
resets its timer, so check the Actions tab occasionally.

Scheduled-workflow failure emails normally go to the person who last changed the
cron line. Review the repository's GitHub notification settings so that recipient
and email routing are appropriate.