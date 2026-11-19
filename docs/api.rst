API reference
=============

The top-level ``normet`` package re-exports the most commonly used symbols.
The main sub-packages are also documented individually below.

.. autosummary::
   :toctree: _autosummary
   :recursive:

   normet
   normet.pipeline
   normet.analysis
   normet.causal
   normet.model
   normet.foundation
   normet.counterfactual
   normet.plotting
   normet.utils
   normet.io

The four sub-packages below re-export a function under the same name as the
module (``normet.analysis.trend``, ``normet.analysis.normalise``,
``normet.analysis.rolling`` and ``normet.pipeline.do_all``). Recursive
autosummary resolves those attributes to the function, so the names below
are listed explicitly to give the classes and helpers their own pages.

.. autosummary::
   :toctree: _autosummary

   normet.analysis.trend.TrendResult
   normet.analysis.trend.TrendTest
   normet.analysis.trend.TrendSlope
   normet.analysis.trend.TrendBreaks
   normet.analysis.trend.mann_kendall
   normet.analysis.trend.theil_sen
   normet.analysis.trend.detect_trend_breaks
   normet.analysis.normalise.NormaliseConfig
   normet.analysis.normalise.normalise_auto
   normet.analysis.rolling.RollingConfig
   normet.analysis.rolling.window_columns
   normet.pipeline.do_all.SingleConfig
   normet.pipeline.do_all.UncConfig
   normet.pipeline.do_all.do_all_unc
