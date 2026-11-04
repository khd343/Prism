# NSE index closes for PRISM's market box

Written by `ui/index_closes_job.py` (the `Index closes` workflow) on this branch only, never on main. Display only:
the sidebar's market box reads `index_closes.csv` for each index's stage and its line against the 200-week average. No score,
scan, record or Kite reads it.

`index_closes.csv`: `date,company_id,close`, one row per session and declared index, the close exactly as NSE wrote it in its
daily file (`ind_close_all_DDMMYYYY.csv`). A close, once kept, is never rewritten. Days with no NSE file (holidays,
weekends) have no rows.
