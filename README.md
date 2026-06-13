# rfscorer

[![CI](https://github.com/jiro-iwanaga/rfscorer/actions/workflows/ci.yml/badge.svg)](https://github.com/jiro-iwanaga/rfscorer/actions/workflows/ci.yml)
[![PyPI version](https://img.shields.io/pypi/v/rfscorer.svg)](https://pypi.org/project/rfscorer/)
[![Python Versions](https://img.shields.io/pypi/pyversions/rfscorer.svg)](https://pypi.org/project/rfscorer/)

`rfscorer` is a Python package for Recency-Frequency based recommendation scoring.

It estimates **product-choice probabilities** — the preference score for each user-item pair, forming a matrix analogous to a rating matrix — from interaction histories, using two simple but powerful behavioral signals: **recency**, which captures how recently a user interacted with an item, and **frequency**, which captures how often the user has interacted with it. The target event whose probability is estimated (revisits, purchases, conversions, etc.) is configurable through the evaluation log.

The package is designed for product recommendation and repeat-engagement modeling, especially in settings where interpretable scoring based on interaction history is preferred over black-box recommendation models.

> Note: In this package, **RF** stands for **Recency-Frequency**, not Random Forest.

## Features

- **scikit-learn-style API** — familiar `fit()` / `transform()` interface makes it easy to integrate into existing data science workflows
- **Minimal data requirements** — works with any interaction log that has three columns: `user`, `item`, and a time column (`datetime` by default, configurable via `time_col`; accepts datetime or integer); no ratings or explicit feedback needed
- **Explainable scoring** — probabilities are derived through mathematical optimization under RF monotonicity constraints, making every score fully traceable and auditable; 3D surface visualization further supports intuitive understanding
- **Probabilistic output** — product-choice probabilities serve as preference scores, enabling expected value calculations and probabilistic ranking of recommendations
- **Extensible** — the user–item probability matrix produced by `transform()` can be directly used as input to collaborative filtering or other downstream recommendation models

## Installation

```bash
pip install rfscorer
```

## Usage

```python
import pandas as pd
from rfscorer import RecencyFrequencyScorer
```

Prepare an interaction log with three columns: `user`, `item`, and a time column (default column name: `datetime`).
The same user-item pair may appear multiple times, representing repeat visits.

| user  | item  | datetime   |
|-------|-------|------------|
| u_001 | i_032 | 2026-07-01 |
| u_001 | i_017 | 2026-07-03 |
| u_001 | i_032 | 2026-07-05 |
| u_002 | i_011 | 2026-07-02 |
| u_002 | i_058 | 2026-07-04 |

Split users into training and test sets, then split each by `target_date` into an observation window and an evaluation window.

```python
target_date = "2026-07-07"

df_train_obs  = df_train[df_train.datetime <= target_date]
df_train_eval = df_train[df_train.datetime >  target_date]
```

Call `fit()` to estimate empirical product-choice probabilities.
Recency and frequency are computed from the observation window; the evaluation window provides ground-truth event labels (revisits, purchases, conversions, etc.).

```python
scorer = RecencyFrequencyScorer()
scorer.fit(df_train_obs, df_train_eval)
```

The empirical surface reflects raw event rates and may be irregular due to sparse data.

```python
fig = scorer.plot_probability_surface(kind="emp")
```

![empirical probability surface](https://raw.githubusercontent.com/jiro-iwanaga/rfscorer/main/img/surface_emp_probability.png)

Call `optimize()` to smooth the surface under RF monotonicity constraints using convex quadratic programming.
`kind="mono"` enforces recency and frequency monotonicity.

```python
scorer.optimize(kind="mono")
fig = scorer.plot_probability_surface(kind="mono")
```

![mono probability surface](https://raw.githubusercontent.com/jiro-iwanaga/rfscorer/main/img/surface_mono_probability.png)

`kind="mcc"` additionally adds convexity in recency and concavity in frequency, yielding a smoother surface.

```python
scorer.optimize(kind="mcc")
fig = scorer.plot_probability_surface(kind="mcc")
```

![mcc probability surface](https://raw.githubusercontent.com/jiro-iwanaga/rfscorer/main/img/surface_mcc_probability.png)

Call `transform()` to score each user-item pair in the test observation window.
It returns a DataFrame with columns `user`, `item`, `recency`, `frequency`, `probability`, and `order` (rank within each user, sorted by probability descending).

```python
df_test_obs  = df_test[df_test.datetime <= target_date]
df_test_eval = df_test[df_test.datetime >  target_date]

df_rec = scorer.transform(df_test_obs, target_date, kind="mcc")
```

| user   | item   | recency | frequency | probability | order |
|--------|--------|--------:|----------:|------------:|------:|
| u_001  | i_032  |       1 |         4 |      0.1167 |     1 |
| u_001  | i_017  |       2 |         3 |      0.0789 |     2 |
| u_001  | i_045  |       3 |         1 |      0.0248 |     3 |
| u_002  | i_011  |       1 |         2 |      0.0621 |     1 |
| u_002  | i_058  |       4 |         1 |      0.0182 |     2 |

Within each user, rows are sorted by `probability` descending; `order` represents the recommendation rank.

Call `evaluate()` to measure recommendation quality at each rank cutoff.
It returns precision, recall, and F1 for each cutoff from 1 to `order`.

```python
scorer.evaluate(df_rec, df_test_eval, order=5)
```

## Examples

- [examples/basic_usage.ipynb](examples/basic_usage.ipynb) — end-to-end walkthrough: load data, fit, optimize, transform, and evaluate

## References
- [Jiro Iwanaga, Naoki Nishimura, Noriyoshi Sukegawa, and Yuichi Takano, “Estimating product-choice probabilities from recency and frequency of page views,” Knowledge-Based Systems, Volume 99, 2016, Pages 157–167.](https://www.sciencedirect.com/science/article/abs/pii/S0950705116000848)

- [Jiro Iwanaga, Kyota Ishihara, Naoki Nishimura, and Ikki Tanaka, *Pythonではじめる数理最適化 ―ケーススタディでモデリングのスキルを身につけよう―*(in Japanese), Ohmsha, 2021.](https://www.ohmsha.co.jp/book/9784274231759/)
  - [Chapter 7: 商品推薦のための興味のスコアリング(in Japanese)](https://github.com/ohmsha/PyOptBook/tree/main/7.recommendation)

- [Jiro Iwanaga, Naoki Nishimura, Noriyoshi Sukegawa, and Yuichi Takano, “Improving collaborative filtering recommendations by estimating user preferences from clickstream data,” Electronic Commerce Research and Applications, Volume 37, Article 100877, 2019.](https://www.sciencedirect.com/science/article/abs/pii/S1567422319300547)


## Citation

If you use `rfscorer` in academic work, please cite the following paper:

- [Jiro Iwanaga, Naoki Nishimura, Noriyoshi Sukegawa, and Yuichi Takano, “Estimating product-choice probabilities from recency and frequency of page views,” Knowledge-Based Systems, Volume 99, 2016, Pages 157–167.](https://www.sciencedirect.com/science/article/abs/pii/S0950705116000848)

```bibtex
@article{Iwanaga2016,
  author  = {Jiro Iwanaga and Naoki Nishimura and Noriyoshi Sukegawa and Yuichi Takano},
  title   = {Estimating product-choice probabilities from recency and frequency of page views},
  journal = {Knowledge-Based Systems},
  volume  = {99},
  pages   = {157--167},
  year    = {2016},
  url     = {https://www.sciencedirect.com/science/article/abs/pii/S0950705116000848}
}
```

If you additionally use the probability matrix as input to a collaborative filtering model, please also cite:

- [Jiro Iwanaga, Naoki Nishimura, Noriyoshi Sukegawa, and Yuichi Takano, “Improving collaborative filtering recommendations by estimating user preferences from clickstream data,” Electronic Commerce Research and Applications, Volume 37, Article 100877, 2019.](https://www.sciencedirect.com/science/article/abs/pii/S1567422319300547)


```bibtex
@article{Iwanaga2019,
  author  = {Jiro Iwanaga and Naoki Nishimura and Noriyoshi Sukegawa and Yuichi Takano},
  title   = {Improving collaborative filtering recommendations by estimating user preferences from clickstream data},
  journal = {Electronic Commerce Research and Applications},
  volume  = {37},
  pages   = {100877},
  year    = {2019},
  url     = {https://www.sciencedirect.com/science/article/abs/pii/S1567422319300547}
}
```

## License

MIT License
