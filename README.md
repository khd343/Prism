# Playbook feed — schema 1

PRISM's 🚀 Playbook Shortlist, one session per file, published by `ui/playbook_feed_job.py` on `main`. Each `sessions/<YYYY-MM-DD>.csv` is written once, never modified, never deleted; its `.json` manifest carries the session (from the sheet's own name), the row count, the CSV's sha256, the recipe fingerprint, the engine commit, the source workbook's sha256, and whether it was `recorded` as the session closed or `backfilled` later.

Only Playbook members are listed: for a valid session, absence means "not in the list today". A missing or invalid session means UNKNOWN, never "out of the list".

| # | Column |
|---|---|
| 1 | `symbol` |
| 2 | `name` |
| 3 | `sector` |
| 4 | `industry` |
| 5 | `cap` |
| 6 | `composite_score` |
| 7 | `crs_52w` |
| 8 | `dist_to_vstop` |
| 9 | `momentum_score` |
| 10 | `breakout_score` |
| 11 | `traded_value_cr` |
| 12 | `rsi_14d` |
| 13 | `stage` |
| 14 | `close_price` |
| 15 | `equity_shares` |

`cap`: LARGE MEGA MICRO MID NANO SMALL or empty. `stage`: Weinstein 1 2 3 4 or empty. Rows sorted by `symbol`; UTF-8, `\n`; empty = not known.
