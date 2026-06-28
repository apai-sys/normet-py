# Synthetic-control modelling

`normet.causal` ships six backends, all callable through `nm.run_scm`:

| Backend  | Estimator                                  | When to use                                         |
|----------|--------------------------------------------|-----------------------------------------------------|
| `scm`    | Ridge-augmented SCM (default)              | Most cases; robust to noisy donors                  |
| `mlscm`  | ML-SCM via FLAML                           | Many donors, non-linear donor-outcome relationships |
| `abadie` | Classic Abadie simplex SCM                 | Sparse donor pool; transparent weights              |
| `did`    | Difference-in-differences baseline         | Sanity check against parallel-trends assumption     |
| `mcnnm`  | Matrix Completion (Athey et al. 2021)      | Highly unbalanced panels; missing observations      |
| `robust` | Robust SCM — HSVT de-noising (Amjad 2018)  | Noisy/low-rank donor matrices; measurement error    |

```python
out = nm.run_scm(
    df=panel,
    date_col="date", unit_col="ID", outcome_col="value",
    treated_unit="Beijing", donors=["Tianjin", "Shanghai", "Guangzhou"],
    cutoff_date="2020-01-23",
    scm_backend="scm",
)
out.tail()  # observed, synthetic, effect
```

## Diagnostics

After fitting, get a quality report and donor-stability summary:

```python
diag = nm.scm_diagnostics(out, cutoff_date="2020-01-23")
# pre_rmse, pre_r2, post_n, att, hhi, effective_n_donors, top_donors, ...

stability = nm.loo_weight_stability(panel, ..., donors=donors)
```

## Inference

Two flavours of significance test:

```python
# Conformal-style finite-sample CI for the post-period ATT
ci = nm.conformal_effect_interval(out, cutoff_date="2020-01-23",
                                  n_perm=1000, ci_level=0.95)

# Abadie's RMSPE-ratio placebo test
pl  = nm.placebo_in_space(panel, ..., scm_backend="scm")
rmt = nm.rmspe_ratio_test(pl, cutoff_date="2020-01-23")
# treated_ratio, placebo_ratios, p_value, rank
```
