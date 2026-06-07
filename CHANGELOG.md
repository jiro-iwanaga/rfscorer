# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

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
