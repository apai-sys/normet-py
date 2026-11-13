# Example notebooks

Paper-reproducibility notebooks live in the repository's
[`notebooks/`](https://github.com/apai-sys/normet-py/tree/main/notebooks) folder — one
per case study of the normet model-description paper, each running end-to-end on the
bundled example data:

1. [Deweathering + adaptive convergence](https://github.com/apai-sys/normet-py/blob/main/notebooks/01_deweathering_and_adaptive_convergence.ipynb)
   — MY1 NO2, COVID lockdown signal, series-RSE stopping rule
2. [Decomposition + rolling normalisation](https://github.com/apai-sys/normet-py/blob/main/notebooks/02_decomposition_and_rolling.ipynb)
   — emission vs met components, moving-window trends
3. [ULEZ Synthetic Control](https://github.com/apai-sys/normet-py/blob/main/notebooks/03_scm_ulez.ipynb)
   — causal effect of the London ULEZ with placebo bands
4. [Transport-aware normalisation](https://github.com/apai-sys/normet-py/blob/main/notebooks/04_transport_aware_normalisation.ipynb)
   — HYSPLIT back-trajectory features and the transport contribution to PM2.5

Beyond the paper's case studies:

5. [Transport vs local attribution](https://github.com/apai-sys/normet-py/blob/main/notebooks/05_transport_vs_local_attribution.ipynb)
   — `decom_met` groups (Shapley) split PM2.5 into local-weather and transport
   contributions, against the average air mass or a clean-air reference
   (`resample_pools`)
6. [Skill benchmark](https://github.com/apai-sys/normet-py/blob/main/notebooks/06_skill_benchmark.ipynb)
   — CRPS of the model's probabilistic forecast against climatology and persistence
   baselines on the held-out MY1 hours
