# Kite v2's lists — schema 1

PRISM's two lists for Kite v2, published by `ui/v2_lists_job.py` on `main`. Every list of every session comes from ONE commit, named in each manifest as `engine_commit`. Per session:

- `sessions/<YYYY-MM-DD>.csv`: the 🚀 Playbook list, exactly the bytes the Playbook feed's code builds at the manifest's `engine_commit` (not the `playbook-feed` branch, which is pinned to an older commit);
- `rank/<YYYY-MM-DD>.csv`: the rank list, one row per stock PRISM scored that session;
- `sessions/<YYYY-MM-DD>.json`: the manifest, written LAST. It carries the session (from the sheet's own name); each list's `rows`, `columns` and `csv_sha256`; the recipe fingerprint; the engine commit; the source workbook's sha256; whether the session was `recorded` as it closed or `backfilled` later; the checks; the Playbook recipe's `blanks` (as the Playbook feed's schema 2 states them); the rank list's counts; and `carried_from`, the earlier session whose `last_result_date` values were carried.

Every file is written once, never modified, never deleted. A session is valid only when its manifest exists and both files match its hashes; a missing or invalid session means UNKNOWN.

## The Playbook list

Only members are listed: for a valid session, absence means "not in the list today".

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

## The rank list

Every scored stock, members or not.

| # | Column |
|---|---|
| 1 | `symbol` |
| 2 | `shariah_status` |
| 3 | `cap` |
| 4 | `traded_value_cr` |
| 5 | `momentum_score` |
| 6 | `momentum_pct` |
| 7 | `vstop_value` |
| 8 | `close_price` |
| 9 | `sector` |
| 10 | `stage` |
| 11 | `last_result_date` |
| 12 | `next_result_date` |
| 13 | `equity_shares` |

`shariah_status`: Shariah, Non-Shariah, Review, Doubtful or empty. `cap`: LARGE MEGA MICRO MID NANO SMALL or empty. `stage`: Weinstein 1 2 3 4 or empty. `momentum_pct`: the stock's momentum percentile over the WHOLE scored market, exactly as the 🚀 recipe reads it, written as the shortest text that reads back to the same number.

The result dates come from the vendor's `days_from_result` (negative = days since the last result). `next_result_date` is the session plus it when it is 0 or more. `last_result_date` is the session plus it when it is 0 or less, otherwise that stock's `last_result_date` on the newest earlier session of this branch (`carried_from`), or empty. A blank or non-whole `days_from_result` is unknown.

Both lists: rows sorted by `symbol`; UTF-8, `\n`; numbers `%.10g`; dates `YYYY-MM-DD`; empty = not known.
