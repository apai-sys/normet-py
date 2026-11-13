normet documentation
====================

**normet** — *Normalisation, Decomposition, and Counterfactual Modelling for
Environmental Time-series*.

``normet`` is a Python package for **deweathering**, **causal inference**, and
**policy evaluation** on environmental time-series data, such as air-quality
records. It brings together three ideas behind one API:

1. **AutoML model training** (FLAML or LightGBM) learns how a pollutant
   concentration depends on meteorology, time, and other predictors.
2. **Monte Carlo meteorological normalisation ("deweathering")** re-runs that model
   thousands of times against resampled, randomised meteorology to cancel out
   day-to-day meteorological noise and reveal the underlying emission-driven trend.
3. **Synthetic Control Methods (SCM)** answer "what would have happened
   without this policy?" by building a counterfactual from untreated donor
   units, with placebo tests and uncertainty bands.

Around them sit zero-shot de-weathering with the Chronos-2 time-series foundation
model (on the same result schema as the AutoML path), decomposition into emission-
and meteorology-driven components, trend and event analysis, uncertainty
attribution, multi-site pipelines, data adapters for ERA5, Open-Meteo, EEA, OpenAQ
and the UK air-quality networks, a command-line tool, an MCP server for AI
assistants, and a desktop app that runs the whole workflow without code.

.. code-block:: bash

   pip install "normet[flaml]"

.. code-block:: python

   import normet as nm

   df = nm.datasets.load_my1()
   met_vars = ["ws", "wd", "temp", "RH", "atmos_pres", "blh", "tcc", "tp", "ssrd"]
   predictors = met_vars + ["date_unix", "day_julian", "weekday", "hour"]

   out, model, df_prep = nm.do_all(df, target="NO2", covariates=predictors,
                                   variables_resample=met_vars)
   nm.normalise_plot(out)

Where to start
--------------

- **New to normet?** :doc:`getting_started/installation`, then
  :doc:`getting_started/quickstart`.
- **Doing an analysis?** The :doc:`guide/workflow` walks through every step; the
  other user-guide pages go deeper into each one.
- **Prefer not to write code?** The :doc:`guide/gui` covers the same workflow.
- **Looking up a function?** The :doc:`api` is generated from the docstrings.
- **Contributing?** See :doc:`development/contributing`.

.. _pdf-manuals:

PDF manuals
-----------

The desktop app's step-by-step user manual, with a screenshot of every panel, is
available as a PDF in English (:download:`normet_manual.pdf <manual/normet_manual.pdf>`)
and Chinese (:download:`normet_manual_zh.pdf <manual/normet_manual_zh.pdf>`).

.. toctree::
   :maxdepth: 2
   :caption: Getting started
   :hidden:

   getting_started/installation
   getting_started/quickstart

.. toctree::
   :maxdepth: 2
   :caption: User guide
   :hidden:

   guide/workflow
   guide/examples_normalisation
   guide/examples_foundation
   guide/examples_feature_engineering
   guide/examples_decomposition
   guide/examples_trend
   guide/examples_uncertainty
   guide/examples_scm_guide
   guide/examples_scm
   guide/examples_multisite
   guide/examples_caching
   guide/examples_data_adapters
   guide/examples_openaq
   guide/examples_plotting
   guide/notebooks

.. toctree::
   :maxdepth: 2
   :caption: Interfaces
   :hidden:

   guide/gui
   guide/cli
   guide/mcp

.. toctree::
   :maxdepth: 1
   :caption: API reference
   :hidden:

   api

.. toctree::
   :maxdepth: 1
   :caption: About
   :hidden:

   citing
   development/changelog
   development/contributing
   development/code_of_conduct
   development/security
   roadmap

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
