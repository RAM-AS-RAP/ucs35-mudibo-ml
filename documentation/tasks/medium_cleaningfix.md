# Task: Restrict medium-gap interpolation to prior years only

In the medium-gap imputation logic (3 to 6 consecutive missing months) within
`preprocessing/cleaning.py`, change the calendar-month lookup so it searches only
backward in time, never forward:

1. Same calendar month, one year earlier
2. Same calendar month, two years earlier

Remove the "one year later" and "two years later" lookup steps entirely. If neither
prior-year value is available, the month must NOT be filled by any other means (no
forward interpolation fallback) — it should instead be reclassified with
`impute_method = 'long_gap_unfilled'`, consistent with how genuinely unfillable long
gaps are already handled.

This change is required because filling a historical gap using a later year's value
uses information that would not yet have existed at that point in time, which is
inconsistent with the walk-forward, no-future-information discipline applied
elsewhere in this pipeline (lag features, rolling statistics, chronological
train/test splitting).

Update the docstring for the medium-gap imputation function to state this
prior-years-only constraint explicitly, and update the printed/logged summary of
imputation methods to reflect that any row previously filled from a later year will
now either be filled from a prior year or fall through to `long_gap_unfilled`.

Do not change the short-gap forward-fill logic or the long-gap policy — only the
medium-gap search direction is affected.