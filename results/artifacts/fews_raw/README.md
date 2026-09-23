# FEWS NET raw gap-fill exports

Manually-saved Postman response bodies for the four WFP long-gap series, used by
`load_fews_gap_files` in `02_data_cleaning.ipynb` since the FEWS API returns 403 when called
programmatically from this environment. Each file is the unmodified JSON array returned by
`https://fdw.fews.net/api/marketpricefacts/` for one market/commodity gap query.

Expected filenames:

- `kisumu_maize.json`
- `mombasa_maize.json`
- `nairobi_beans.json`
- `nakuru_beans.json`
