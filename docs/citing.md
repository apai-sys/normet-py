# Citing normet

If you use `normet` in your work, please cite the package:

```bibtex
@Manual{normet-pkg,
  title        = {normet: Normalisation, Decomposition, and Counterfactual
                  Modelling for Environmental Time-series},
  author       = {Congbo Song and Contributors},
  year         = {2026},
  note         = {Python package version 1.0.0},
  organization = {University of Manchester},
  url          = {https://github.com/apai-sys/normet-py},
}
```

The same metadata is in
[`CITATION.cff`](https://github.com/apai-sys/normet-py/blob/main/CITATION.cff) at the
root of the repository, so GitHub's **Cite this repository** button (on the
repository's front page) gives the citation in APA and BibTeX form, and reference
managers that read the Citation File Format can import it directly.

Please also cite the methods you use. The functions' docstrings and the user guide
name the papers behind them (for example Abadie's synthetic control, Amjad et al. 2018
for Robust SCM, Athey et al. 2021 for matrix completion, Gneiting & Raftery 2005 for
the CRPS), and the example datasets list their sources in
{doc}`getting_started/quickstart`.
