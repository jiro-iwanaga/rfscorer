# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- `optimize(kind='mrc')`: new optimization model applying monotonicity + recency convexity constraint.
  Recency convexity enforces diminishing marginal penalty as recency grows
  (second difference ≥ 0 along the recency axis).
- `optimize(kind='mfc')`: new optimization model applying monotonicity + frequency concavity constraint.
  Frequency concavity enforces diminishing marginal returns as frequency grows
  (second difference ≤ 0 along the frequency axis).
- Corresponding attributes populated by `optimize(kind='mrc')`:
  `mrc_probability_`, `mrc_probability_table_`, `mrc_probability_dict_`
- Corresponding attributes populated by `optimize(kind='mfc')`:
  `mfc_probability_`, `mfc_probability_table_`, `mfc_probability_dict_`
- `predict()`, `transform()`, `plot_probability_surface()`, and `export_probability_csv()`
  now accept `kind='mrc'` and `kind='mfc'` in addition to the existing values.
- `export_probability_csv(kind='all')` now outputs all five models:
  `empirical_probability`, `mono_probability`, `mrc_probability`, `mfc_probability`, `mcc_probability`.

## [0.2.6] - 2026-06-07

### Added

- `plot_marginal_probability(axis='recency', ...)` method for visualizing marginal empirical revisit
  probability along the recency or frequency axis as a line chart.
  Returns a `matplotlib.figure.Figure` for inline display in Jupyter Lab / Colab.
  Use this method to check monotonicity before calling `optimize()`.
- Marginal probability attributes populated by `fit()` / `fit_period()`:
  - `recency_probability_` / `frequency_probability_`: DataFrames with aggregated N, cv, probability
  - `R2N`, `R2CV`, `R2Prob`: dicts mapping recency rank to sample count, conversion count, probability
  - `F2N`, `F2CV`, `F2Prob`: dicts mapping frequency to sample count, conversion count, probability
- `title`, `figsize`, `fontsize` parameters to `plot_probability_surface()` and `plot_marginal_probability()`
  for publication-ready output.
- `recency_label`, `frequency_label`, `probability_label` parameters to `plot_probability_surface()`
  for customizing axis labels.
- `xlabel`, `probability_label` parameters to `plot_marginal_probability()`
  for customizing axis labels.
- Optional `[ja]` extra dependency: `pip install rfscorer[ja]` installs `japanize-matplotlib`
  to enable Japanese axis labels and titles in both plot methods.

### Changed

- `img/` reference PNGs updated to reflect the new plot style (black wireframe, transparent panes).

## [0.2.5] - 2026-06-07

### Changed

- `basic_usage.ipynb`: updated to use the new `plot_probability_surface()` API.
  Assign the returned figure to a variable (`fig = scorer.plot_probability_surface(...)`)
  to prevent double rendering in Jupyter Lab / Colab.

### Removed

- `examples/access_log.csv` from version control. The notebook loads the data directly
  from the source URL, so the file no longer needs to be committed to the repository.
- `examples/README.md`, which existed solely to attribute the license of `access_log.csv`.
  The attribution note is retained in `basic_usage.ipynb`.

## [0.2.4] - 2026-06-07

### Changed

- `plot_probability_surface()` now returns a `matplotlib.figure.Figure` instead of saving directly to a file.
  In Jupyter Lab and Colab the returned figure renders inline automatically.
  To save to a file, call `fig.savefig("output.png")` on the returned figure.

### Removed

- `path` parameter from `plot_probability_surface()`.
  File saving is no longer handled by the method; use `fig.savefig()` instead.

### Added

- Tests for `plot_probability_surface()` covering all three `kind` values and error cases.
