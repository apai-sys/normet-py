# Example: Bayesian Synthetic Control Method (SCM) and Causal Inference

This guide demonstrates the full Bayesian Synthetic Control Method (SCM) workflow in `normet`. We will fit a simplex SCM with a Dirichlet prior, use **automatic event detection** to find the intervention cutoff date, extract robust highest density intervals (HDI) for donor weights, and plot posterior credible bands.

---

## 1. Prerequisites

Make sure the optional Bayesian modeling dependencies are installed:
```bash
pip install pymc arviz matplotlib pandas numpy
```

---

## 2. Generating a Causal Panel Dataset

Let's construct a synthetic air pollution panel containing one treated unit (e.g. `"Beijing"`) and three donor pool stations (e.g. `"Tianjin"`, `"Shanghai"`, `"Guangzhou"`). We will inject an intervention anomaly on Beijing starting from a specific date.

```python
import numpy as np
import pandas as pd

# Generate 60 days of hourly/daily observations
np.random.seed(42)
dates = pd.date_range("2024-01-01", periods=60, freq="D")
donors = ["Tianjin", "Shanghai", "Guangzhou"]
units = ["Beijing"] + donors

rows = []
for d in dates:
    # Under regular conditions, Beijing is a linear combination of donors:
    # Beijing = 0.5 * Tianjin + 0.3 * Shanghai + 0.2 * Guangzhou + noise
    val_tj = float(np.random.normal(12.0, 1.5))
    val_sh = float(np.random.normal(8.0, 1.0))
    val_gz = float(np.random.normal(6.5, 1.0))

    val_bj = 0.5 * val_tj + 0.3 * val_sh + 0.2 * val_gz + float(np.random.normal(0.0, 0.2))

    # Introduce a sharp intervention (e.g. lockdown) in Beijing starting on day 40 (Jan 10)
    if d >= pd.to_datetime("2024-02-10"):
        val_bj -= 3.5 # Sharp reduction in PM2.5

    rows.append({"date": d, "code": "Beijing", "poll": val_bj})
    rows.append({"date": d, "code": "Tianjin", "poll": val_tj})
    rows.append({"date": d, "code": "Shanghai", "poll": val_sh})
    rows.append({"date": d, "code": "Guangzhou", "poll": val_gz})

df_panel = pd.DataFrame(rows)
```

---

## 3. Auto-Cutoff Detection & Fitting Bayesian SCM

Instead of hardcoding a `cutoff_date`, we can omit it (`None`). `normet.bayesian_scm` will automatically integrate with `normet.detect_events` on the treated unit series to identify the earliest anomaly event start date and treat it as the treatment intervention cutoff.

```python
import normet as nm

# Fit the Bayesian SCM
# We set draws=500 and tune=500 for a fast execution
result = nm.bayesian_scm(
    df=df_panel,
    date_col="date",
    unit_col="code",
    outcome_col="poll",
    treated_unit="Beijing",
    cutoff_date=None, # Trigger auto-cutoff detection!
    donors=donors,
    draws=500,
    tune=500,
    chains=1
)
```

> [!NOTE]
> If no anomalies are detected in the treated unit outcome series, `bayesian_scm` will raise a helpful `ValueError`. In such cases, or when the exact intervention time is known, a manual `cutoff_date` (e.g., `"2024-02-10"`) should be provided.

---

## 4. Extracting Donor Weights & HDI Summary

After fitting, the method returns a `weights_summary` DataFrame containing the posterior mean weight and robust highest density intervals (HDI) computed using ArviZ directly from MCMC samples:

```python
# Access the robustly calculated donor weights and intervals
weights_df = result["weights_summary"]
print("Simplex Donor Weights Summary (with 95% HDI bounds):")
print(weights_df)
```

**Expected Output:**
```text
            mean   ci_low  ci_high
Tianjin    0.502    0.441    0.563
Shanghai   0.298    0.231    0.362
Guangzhou  0.200    0.142    0.258
```

---

## 5. Visualising Causal Outcomes and Posterior Bands

We can plot the observed vs. synthetic counterfactual curves and the treatment effect path using the posterior credible bands:

```python
import matplotlib.pyplot as plt

# Plot Observed vs Synthetic, plus Treatment Effect path
# This automatically renders a 2-panel figure:
# Panel 1: Observed Beijing vs. Synthetic (with 95% Credible Band)
# Panel 2: Treatment Effect (Observed - Synthetic, with Credible Band)
fig = nm.plot_bayesian_scm(
    result,
    cutoff_date="2024-02-10",
    ci_level=0.95,
    title="Beijing PM2.5 Intervention Analysis"
)

plt.tight_layout()
plt.savefig("beijing_bayesian_scm.png", dpi=150)
plt.show()
```

---

## 6. Contrast with Standard Abadie SCM

To cross-check the Bayesian fit against a deterministic Abadie simplex SCM, run the same panel through `run_scm`:

```python
# Run classic Abadie SCM
classic_res = nm.run_scm(
    df=df_panel,
    date_col="date",
    unit_col="code",
    outcome_col="poll",
    treated_unit="Beijing",
    cutoff_date="2024-02-10",
    donors=donors,
    scm_backend="abadie",
)

# run_scm returns a DataFrame indexed by date with columns
# [observed, synthetic, effect] — the synthetic counterfactual and its gap.
print("\nClassic Abadie SCM (observed vs. synthetic counterfactual):")
print(classic_res.tail())
```

> [!NOTE]
> `run_scm` returns the synthetic-control **series**, not a weights dict. For
> donor weights and fit diagnostics (HHI, effective donor count, top donors),
> pass the result to `nm.scm_diagnostics(classic_res, cutoff_date="2024-02-10")`.
