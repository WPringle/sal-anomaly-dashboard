# SAL Anomaly Dashboard

GitHub Pages site for MPAS-Ocean SAL tidal validation against TPXO.

Live at <https://wpringle.github.io/sal-anomaly-dashboard/>

## Structure

```
├── index.html                 # redirect to tpxo.html
├── tpxo.html                  # TPXO harmonic validation dashboard (standalone)
├── css/style.css
└── img/
    ├── tpxo/
    │   ├── bpanomaly/         # harmonic plots + amp/phase error maps per run
    │   ├── atm-tide-only/
    │   ├── ZAE0.0/ ZAE0.1/ ZAE0.2/ ZAE0.4/
    │   └── salfix-45d/
    └── tidal_rmse_diff/       # per-constituent RMSE difference + amp bias maps
```

`tpxo.html` is self-contained: it loads no JSON and pulls image paths directly
from the `img/` tree, so adding plots is a matter of dropping files in and
updating the path maps in its inline `<script>`.

## Station time series

The per-station time-series dashboard (`comparison.html`, `sal-tide-only.html`,
`comparison-16mo.html`, `steven.html`, `zae.html` and `data/`) was removed from
this site — the JSON totalled ~2.4 GB against a 1 GB GitHub Pages site limit.
`data/` is now gitignored. Everything is recoverable from history:

```bash
git checkout 3f43332 -- data comparison.html js
```

## Local testing

```bash
python3 -m http.server 8000
# open http://localhost:8000
```

## Deployment

`.github/workflows/pages.yml` deploys the repo root to GitHub Pages on every
push to `main`. Because this repo is a fork, Actions must be enabled once from
the Actions tab before the workflow will run.
