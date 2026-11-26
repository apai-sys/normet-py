# normet Scientific Guardrails

- Quote numbers strictly as the `normet` tools return them; never compute, extrapolate, or invent statistics.
- Propose the target and the meteorological covariates with your reasons, and wait for the user to confirm before starting a job. Say what you left out and why.
- Treat job-starting tools (`normet_start_do_all`, `normet_start_decompose`, `normet_start_scm`, `normet_start_deweather`) as asynchronous: report the `job_id` and its reproducible `normet` command, then poll `normet_job_status`; never load a whole table into context.
- Judge a model on its testing rows (R², RMSE, FAC2) before normalising with it; a poor held-out fit is inherited by the normalised series.
- Do not claim a trend or a cause the numbers do not show. A trend is what `normet_trend` returns: quote the block-bootstrap-corrected p it names, not the classical one.
- For a synthetic control, propose the treated unit, cutoff and donor pool, and judge the pre-period fit before reading the effect. Report the placebo-in-space and RMSPE-ratio results together.
- Use `normet_figure` for figures and `normet_write_report` for the HTML report; tell the user the paths of both.
- Respect `$NORMET_MCP_ROOTS` path boundaries; never move files outside the allowed roots silently.
