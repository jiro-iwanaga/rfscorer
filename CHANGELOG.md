# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.5.0] - 2026-06-19

### Changed

- `split_by_date()`: `observation_days` and `gt_days` are now required arguments with no
  defaults (previously `observation_days=28, gt_days=7`). The `time_col="datetime"` default
  is preserved. **Breaking**: callers omitting these arguments
  (e.g. `split_by_date(df, target_date)`) will now receive a `TypeError`. Pass both arguments
  explicitly: `split_by_date(df, target_date, observation_days=N, gt_days=M)`.
- All tutorial notebooks updated to pass `observation_days` / `gt_days` as explicit keyword
  arguments (`tutorial_beginner_en/ja`, `tutorial_practical_en/ja`,
  `tutorial_advanced_fit_rolling_en/ja`).
- README: English section revised and aligned with the Japanese section — intro rewritten
  to clarify product-choice probability and downstream uses, Features simplified (Extensible
  bullet removed), Usage links updated to the Examples section, and advanced tutorials added
  to the Examples section.
- Documentation (`docs/`): terminology unified — "閲覧" replaced with "接触" in
  `functional-design.md`, `glossary.md`, and `product-requirements.md`; `split_by_date`
  signature corrected to reflect required arguments in `functional-design.md` and
  `product-requirements.md`; advanced tutorial notebooks added to `repository-structure.md`.

## [0.4.5] - 2026-06-17

### Added

- Spearman correlation diagnostic attributes, populated automatically by `fit()`:
  - `recency_corr_` / `frequency_corr_`: equal-weight Spearman correlation between the RF rank
    and the empirical probability. Negative recency / positive frequency correlation indicates
    the expected monotonic relationship.
  - `recency_corr_weighted_` / `frequency_corr_weighted_`: sample-size weighted variants.
  - `recency_corr_pvalue_` / `frequency_corr_pvalue_`: two-sided p-values for the equal-weight
    correlations.
  - `recency_slice_corr_` / `frequency_slice_corr_`: dicts of per-slice correlations for
    diagnosing 2D monotonicity (e.g. recency-vs-probability within each frequency level).
- `verbose` parameter to `optimize()` (default `False`): when `True`, prints solver progress and
  result summary.
- `path` parameter to `plot_probability_surface()` and `plot_marginal_probability()`: save the
  figure directly. A directory path writes a default filename
  (`surface_{kind}_probability.png` / `marginal_{kind}_probability.png`); a file path writes that
  name. Both methods now also show a default title based on `kind` when `title=None`.
- Practical tutorial notebooks (`examples/tutorial_practical_ja.ipynb` /
  `tutorial_practical_en.ipynb`): user-level train/test split, building all nine models,
  accuracy comparison, and model save/load (pickle and zip archive).

### Changed

- `show()` redesigned as a structured diagnostic report (data statistics, fit parameters,
  Spearman correlations, and the empirical probability table) instead of the previous
  profiling-style output.
- `plot_marginal_probability()` API redesign. **Breaking changes:**
  - `kind` values changed from `("emp", "er", "ef", "mr", "mf", "all")` to
    `("er", "ef", "mr", "mf", "rboth", "fboth")`. The `"all"` overlay is split into `"rboth"`
    (empirical + monotonic recency) and `"fboth"` (empirical + monotonic frequency); `"emp"` is
    removed.
  - The `axis` parameter is removed (the axis is now inferred from `kind`).
  - The separate `recency_label` / `frequency_label` parameters are consolidated into a single
    `axis_label=None`.
- `export_probability_csv()`: the default output filename changed from `{kind}_probability.csv`
  to `probability_{kind}.csv` (e.g. `probability_emp.csv`). **Breaking** only for callers relying
  on the auto-generated name when passing `path=None` or a directory; explicit file paths are
  unaffected. (The CSV names *inside* the `save_zip()` archive are unchanged.)
- Internal aggregation dicts renamed to private:
  `R/F/RF2N/RF2CV/RF2Prob/R2N/R2CV/R2Prob/F2N/F2CV/F2Prob` → `_R/_F/_RF2N/…`. These were
  implementation details, not public API; the public probability attributes
  (`emp_probability_dict_`, `er_probability_dict_`, etc.) are unchanged.
- Documentation overhaul across `docs/`: document titles renamed (アーキテクチャ構成書 /
  機能仕様書 / リポジトリ構成 / 用語集), `glossary.md` restructured (基本概念・期間とデータ分割・
  アルゴリズム・API 簡潔版) with terminology unification (推薦スコア＝商品選択確率, 対象イベント,
  behavior history), and a release procedure added to `development-guidelines.md`.
- Beginner tutorial notebooks (`tutorial_beginner_ja.ipynb` / `tutorial_beginner_en.ipynb`)
  updated for the revised API and terminology.
- Test suite expanded (+30 cases; 439 passing) covering transform with 2D optimized kinds,
  plot `path` saving, objective-function fit quality (analytic optima), datetime64 splits, and
  version-mismatch semantics for `load()` / `load_zip()`.

### Removed

- `recency_probability_` and `frequency_probability_` attributes; consolidated into
  `er_probability_` / `ef_probability_`.
- `axis` parameter, and the `"emp"` / `"all"` kind values, from `plot_marginal_probability()`.

### Fixed

- Dependency floor: `cvxpy>=1.3` → `cvxpy>=1.5`. The optimizer explicitly solves with the
  CLARABEL solver, which is bundled with cvxpy since 1.4 and the default since 1.5; the previous
  `>=1.3` floor could resolve an environment without CLARABEL available.

## [0.4.4] - 2026-06-15

### Added

- `save(path=None)` / `load(path)`: persist a fitted model to a pickle file and restore it
  without retraining. `path=None` saves `rfscorer.pkl` to the current directory; a directory
  path saves `rfscorer.pkl` inside it; a file path saves directly. On major or minor version
  mismatch, `load()` emits a `UserWarning` and continues loading.
- `save_zip(path=None)` / `load_zip(path)`: save/restore the model as a zip archive bundling
  `rfscorer.pkl`, `metadata.json` (version, parameters, fit statistics), probability-table CSVs,
  and plot PNGs for all computed model kinds. `path=None` saves `scorer.zip` to the current
  directory. Intended for research sharing and artifact management.
- Tutorial notebooks (`tutorial_beginner_en.ipynb` / `tutorial_beginner_ja.ipynb`): added
  Section 10 covering `save()` / `load()` usage with a Google Colab persistence guide.
  Also added a commented `# !pip install rfscorer` line to the import cell.

### Changed

- Terminology unification: renamed all `eval`-prefixed names to `gt` (ground truth)
  to align with the unified terminology in `docs/glossary.md` (正解データ / ground truth data).
  Breaking changes:
  - `fit(df_obs, df_eval, ...)` → `fit(df_obs, df_gt, ...)`
  - `evaluate(df_rec, df_eval, ...)` → `evaluate(df_rec, df_gt, ...)`
  - `split_by_date(..., evaluation_days=7, ...)` → `split_by_date(..., gt_days=7, ...)`;
    return value documented as `(df_obs, df_gt)` instead of `(df_obs, df_eval)`.
  - Attribute `record_num_eval` → `record_num_gt`
  - Attribute `evaluation_start_` → `gt_start_`
  - Attribute `evaluation_end_` → `gt_end_`
  - `show()` output label `evaluation:` → `ground_truth:`
  - Error message "No events observed in evaluation period" → "No events observed in ground truth period"
- `fit()`: the `datetime` column in `df_gt` is now optional. Only `user` and `item` columns
  are required for fitting. The `gt_start_` and `gt_end_` attributes (which depended on
  `df_gt`'s datetime column) have been removed.

## [0.4.3] - 2026-06-15

### Added

- Tutorial notebooks: Created bilingual beginner tutorials
  - `examples/tutorial_beginner_en.ipynb`: English version translated from Japanese tutorial
    using terminology from `docs/glossary.md` (behavior history, observation data, ground truth data, etc.)
  - Covers complete workflow: data loading, splitting, model building (emp/mono/mcc),
    probability visualization, scoring, and evaluation.
  - Updated README.md example references to point to the new tutorial notebooks
    (`examples/tutorial_beginner_en.ipynb` for English section,
    `examples/tutorial_beginner_ja.ipynb` for Japanese section).

## [0.4.2] - 2026-06-14

### Changed

- Documentation: Comprehensive README.md improvements including:
  - Added **Citation** section with in-text academic citation templates and BibTeX references
    for citing the package in research papers.
  - Added **Minimal Example** demonstrating end-to-end workflow with `split_by_date()`,
    `fit()`, `optimize()`, `visualize()`, `transform()`, and `evaluate()` methods.
  - Improved **Visualization** section with side-by-side comparison of three representative
    optimization methods (Empirical, Monotone, Monotonicity-Convex-Concave) using horizontal layout.
  - Resized visualization images for optimal display in documentation.
  - Simplified English language throughout Features, Usage, and method descriptions for clarity.
  - Created comprehensive **Japanese README** (`# RFScorer (日本語README)`) with complete
    translation of all explanatory text while preserving code examples and diagrams exactly.
  - Aligned English and Japanese versions to ensure consistent technical terminology
    (product-choice probabilities, optimization methods, feature descriptions).

## [0.4.1] - 2026-06-13

### Fixed

- `split_by_date()`: `observation_days=N` now produces an N-unit observation window
  `[target_date - N + 1, target_date]`, restoring symmetry with `evaluation_days=N`
  (which produces the N-unit window `[target_date + 1, target_date + N]`).
  Previously `observation_days=N` produced an `N+1`-unit window
  `[target_date - N, target_date]` due to an off-by-one in the inclusive start
  boundary. **Migration**: if you previously called
  `split_by_date(df, target_date, observation_days=N)` and want the same
  observation window, pass `observation_days=N+1`.
- `normalize_ref()`: invalid string dates (e.g., `"not a date"`) now consistently
  raise `ValueError("time value could not be normalized: ...")`. Previously the
  str-path bypassed the friendly error and surfaced a raw pandas error.
- Documentation: numerous accuracy fixes across `docs/` (glossary, functional-design,
  product-requirements, architecture, repository-structure, development-guidelines)
  including terminology unification (閲覧, 累積対象イベント発生数), `_plotting.py`
  reference in the module / repository layout, Python 3.11 minimum requirement,
  and `kind` enum corrections for `plot_probability_surface()` /
  `plot_marginal_probability()`.

### Changed

- Internal refactor: extracted `plot_probability_surface()` and
  `plot_marginal_probability()` to a new private module `src/rfscorer/_plotting.py`
  as `PlottingMixin`. `RecencyFrequencyScorer` now inherits from `PlottingMixin`,
  so the public API (`scorer.plot_*()`) is preserved with no caller-visible change.
- Internal refactor: reorganized `RecencyFrequencyScorer` methods by typical
  workflow (Initialization → Fitting → Optimization → Inference → Evaluation →
  Export → Inspection → Internal helpers) and added section divider comments.
  No behavior change.

## [0.4.0] - 2026-06-13

### Added

- `split_by_date(df, target_date, observation_days=28, evaluation_days=7, time_col="datetime")`:
  new top-level utility function (`from rfscorer import split_by_date`) that splits a single
  behavior history into an observation/evaluation pair at `target_date`.
  Returns `(df_obs, df_eval)`. Accepts the same datetime or integer `time_col` as the scorer.
- `unit` parameter to `RecencyFrequencyScorer.__init__()`: controls recency bin granularity.
  `unit=7` gives weekly recency, `unit=30` approximate monthly.  Default `unit=1` preserves
  the previous day-level behavior.
- Integer `time_col` support: `time_col` columns of integer dtype are now accepted in addition
  to datetime / string columns across `fit()`, `transform()`, and `split_by_date()`.
- `plot_marginal_probability(kind="er")` and `kind="ef"`: new 1-D marginal plot support for the
  empirical recency and frequency models.

### Changed

- `er_probability_` and `ef_probability_` are now true 1-D outputs, mirroring the earlier
  `mr` / `mf` refactor. Breaking changes:
  - `er_probability_`: columns reduced to `(recency, probability)`
    (previously `recency, frequency, probability` after 2-D broadcast)
  - `ef_probability_`: columns reduced to `(frequency, probability)`
    (previously `recency, frequency, probability` after 2-D broadcast)
  - `er_probability_dict_`: keys changed from `(r, f)` tuple to `int r`
  - `ef_probability_dict_`: keys changed from `(r, f)` tuple to `int f`
  - `predict(kind="er")`: `f` argument is now ignored; `r` is clamped to `recency_limit`
  - `predict(kind="ef")`: `r` argument is now ignored; `f` is clamped to `frequency_limit`
  - `plot_probability_surface(kind="er"|"ef")`: now raises `ValueError`
    (use `plot_marginal_probability()` instead)
- `empirical_probability_*` attributes renamed to `emp_probability_*` for consistency with all
  other short-form kind prefixes. Breaking changes:
  - `empirical_probability_`       → `emp_probability_`
  - `empirical_probability_table_` → `emp_probability_table_`
  - `empirical_probability_dict_`  → `emp_probability_dict_`
  - CSV column `"empirical_probability"` (from `export_probability_csv(kind="all")`) → `"emp_probability"`
  - The kind aliases `"empirical"`, `"empirical_recency"`, `"empirical_frequency"` are preserved.

### Removed

- Python 3.10 support. Minimum supported version is now Python 3.11.
- `er_probability_table_` and `ef_probability_table_` attributes.
  These were 2-D broadcast grids produced by the previous implementation and are no longer generated.

## [0.3.2] - 2026-06-11

### Changed

- `optimize(kind='mr')` and `optimize(kind='mf')` no longer broadcast results to the full RF grid.
  Results are now stored as true 1-D outputs:
  - `mr_probability_`: DataFrame with columns `recency, probability`
    (previously `recency, frequency, probability` after broadcast)
  - `mf_probability_`: DataFrame with columns `frequency, probability`
    (previously `recency, frequency, probability` after broadcast)
  - `mr_probability_dict_`: keyed by recency rank `r` (int)
    (previously keyed by `(r, f)` tuple)
  - `mf_probability_dict_`: keyed by frequency `f` (int)
    (previously keyed by `(r, f)` tuple)
- `plot_probability_surface()` now raises `ValueError` when `kind='mr'` or `kind='mf'` is specified,
  as 1-D models cannot be represented as a surface plot.

### Removed

- `mr_probability_table_` and `mf_probability_table_` attributes.
  These were 2-D broadcast grids produced by the previous implementation and are no longer generated.

## [0.3.1] - 2026-06-10

### Fixed

- `examples/basic_usage.ipynb`: corrected `transform()` call to use a pre-filtered observation
  window (`df_test_obs`) instead of the full test log, matching the documented API contract.
- `README.md`: rewrote the Usage section to reflect the current API — `fit()` now takes
  pre-split `df_obs` / `df_eval` DataFrames, and `transform()` requires a pre-filtered
  observation window. Added `plot_probability_surface()` commands alongside each surface image.

### Added

- Tests for `optimize()` kind aliases (`monotonic`, `monotonic_recency`, etc.) and
  `export_probability_csv()`.

## [0.3.0] - 2026-06-07

### Added

- `eps` parameter to `optimize()` and `RFOptimizer.build_model()` for strict monotonicity.
  When `eps > 0`, adjacent recency/frequency probability values are forced to differ by at least
  `eps`, preventing ties. Default `eps=0.0` preserves the existing weak monotonicity behavior.
  Applies to all `kind` values (`mono`, `mr`, `mf`, `mrc`, `mfc`, `mcc`).
- Automatic upper-bound validation for `eps`: raises `ValueError` if `eps` exceeds
  `p_max / (n - 1)` (where `p_max` is the empirical probability maximum and `n` is the number
  of recency or frequency levels), ensuring the problem remains feasible.

### Changed

- Kind aliases renamed from `monotone_*` to `monotonic_*` for consistent mathematical terminology.

  | Old alias | New alias | Canonical |
  |---|---|---|
  | `monotone` | `monotonic` | `mono` |
  | `monotone_recency` | `monotonic_recency` | `mr` |
  | `monotone_frequency` | `monotonic_frequency` | `mf` |
  | `monotone_recency_convex` | `monotonic_recency_convex` | `mrc` |
  | `monotone_frequency_concave` | `monotonic_frequency_concave` | `mfc` |
  | `monotone_convex_concave` | `monotonic_convex_concave` | `mcc` |

## [0.2.8] - 2026-06-07

### Added

- `optimize(kind='mr')`: new 1-D optimization model for the recency axis.
  Enforces monotonically decreasing + convex constraints on the marginal recency probability R2Prob,
  then broadcasts the result across all frequency values.
- `optimize(kind='mf')`: new 1-D optimization model for the frequency axis.
  Enforces monotonically increasing + concave constraints on the marginal frequency probability F2Prob,
  then broadcasts the result across all recency values.
- `er` model: empirical recency marginal probability (R2Prob) broadcast to the full RF grid.
  Computed automatically inside `fit()` / `fit_period()`; no extra call needed.
- `ef` model: empirical frequency marginal probability (F2Prob) broadcast to the full RF grid.
  Computed automatically inside `fit()` / `fit_period()`; no extra call needed.
- Corresponding attributes populated by `optimize(kind='mr')`:
  `mr_probability_`, `mr_probability_table_`, `mr_probability_dict_`
- Corresponding attributes populated by `optimize(kind='mf')`:
  `mf_probability_`, `mf_probability_table_`, `mf_probability_dict_`
- Corresponding attributes populated by `fit()` / `fit_period()`:
  `er_probability_`, `er_probability_table_`, `er_probability_dict_`,
  `ef_probability_`, `ef_probability_table_`, `ef_probability_dict_`
- Kind alias system: long descriptive names are accepted everywhere and normalized to their
  canonical short forms via `_normalize_kind()`.

  | Alias | Canonical |
  |---|---|
  | `empirical` | `emp` |
  | `empirical_recency` | `er` |
  | `empirical_frequency` | `ef` |
  | `monotonic` | `mono` |
  | `monotonic_recency` | `mr` |
  | `monotonic_frequency` | `mf` |
  | `monotonic_recency_convex` | `mrc` |
  | `monotonic_frequency_concave` | `mfc` |
  | `monotonic_convex_concave` | `mcc` |

- `plot_marginal_probability()` now accepts a `kind` parameter (`"emp"`, `"mr"`, `"mf"`, `"all"`).
  `kind="all"` overlays the empirical and optimized 1-D series on the same axes
  (solid line for `emp`, dashed line for `mr` / `mf`).

### Changed

- Internal canonical kind name changed from `"empirical"` to `"emp"` for consistency with all other
  short-form kind names (`mono`, `mr`, `mf`, `mrc`, `mfc`, `mcc`).
  The string `"empirical"` continues to work as an alias.
- `plot_marginal_probability()`: replaced `xlabel` parameter with `recency_label` / `frequency_label`
  to match the naming convention of `plot_probability_surface()`.
- `img/surface_empirical_probability.png` renamed to `img/surface_emp_probability.png`.
- `export_probability_csv(kind='all')` now outputs all nine models:
  `emp`, `er`, `ef`, `mono`, `mr`, `mf`, `mrc`, `mfc`, `mcc`.

## [0.2.7] - 2026-06-07

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
