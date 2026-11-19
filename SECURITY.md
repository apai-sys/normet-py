# Security policy

## Supported versions

`normet` is maintained by a small team, so security fixes are made on the latest
release only and are not backported to older versions. If you are on an older
version, upgrade before reporting:

```bash
pip install --upgrade normet
```

| Version | Supported |
|---|---|
| Latest release | Yes |
| Older releases | No |

## Reporting a vulnerability

Please **do not** report security problems in public GitHub issues, pull requests or
discussions.

Email the maintainers at congbo.song@manchester.ac.uk with:

- a description of the problem and its impact,
- the steps or a minimal example that reproduces it,
- the `normet` and Python versions (`normet info`) and your operating system,
- whether the problem is already public anywhere.

Please give us a reasonable time to release a fix before disclosing the problem
publicly.

## Scope

`normet` runs locally on data you supply. Parts of it reach the network or other
programs, and problems there are in scope:

- the data adapters in `normet.io` (HTTP requests to ERA5/CDS, Open-Meteo, EEA,
  OpenAQ and the UK air-quality networks),
- the `normet-mcp` server and the desktop app's assistant, which serve normet's tools
  to an AI model (the desktop app's tool server listens on 127.0.0.1 with a
  per-window token),
- loading files: run archives (`load_run`) and cached results are `joblib` pickles,
  and loading a pickle runs code, so only load archives you or someone you trust
  created.

Vulnerabilities in third-party dependencies should be reported to those projects;
tell us as well if `normet` needs to change its version bounds.
