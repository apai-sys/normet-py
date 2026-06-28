normet documentation
====================

**normet** — *Normalisation, Decomposition, and Counterfactual Modelling for
Environmental Time-series*.

A Python toolbox for air-quality and environmental time-series analysis with:

- AutoML-driven model training (FLAML).
- Weather normalisation, including resampling-based and SHAP-based decomposition.
- Synthetic-control-style counterfactual modelling (classic SCM, ML-SCM,
  Abadie, DiD, MC-NNM) with conformal and placebo inference.
- Multi-site batch pipelines, walk-forward cross-validation, and on-disk caching.

.. toctree::
   :maxdepth: 2
   :caption: User guide

   guide/examples_normalisation
   guide/examples_feature_engineering
   guide/examples_decomposition
   guide/examples_scm_guide
   guide/examples_multisite
   guide/examples_caching
   guide/examples_data_adapters
   guide/examples_openaq
   guide/examples_scm
   guide/examples_plotting

.. toctree::
   :maxdepth: 1
   :caption: Reference

   api
   roadmap

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
