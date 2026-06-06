# rfscorer

[![CI](https://github.com/jiro-iwanaga/rfscorer/actions/workflows/ci.yml/badge.svg)](https://github.com/jiro-iwanaga/rfscorer/actions/workflows/ci.yml)
[![PyPI version](https://img.shields.io/pypi/v/rfscorer.svg)](https://pypi.org/project/rfscorer/)
[![Python Versions](https://img.shields.io/pypi/pyversions/rfscorer.svg)](https://pypi.org/project/rfscorer/)

`rfscorer` is a Python package for Recency-Frequency based recommendation scoring.

It estimates **revisit probabilities** — the preference score for each user-item pair, forming a matrix analogous to a rating matrix — from interaction histories, using two simple but powerful behavioral signals: **recency**, which captures how recently a user interacted with an item, and **frequency**, which captures how often the user has interacted with it.

The package is designed for product recommendation and revisit modeling, especially in settings where interpretable scoring based on interaction history is preferred over black-box recommendation models.

> Note: In this package, **RF** stands for **Recency-Frequency**, not Random Forest.

## Features

- **scikit-learn-style API** — familiar `fit()` / `transform()` interface makes it easy to integrate into existing data science workflows
- **Minimal data requirements** — works with any interaction log that has three columns: `user`, `item`, and `datetime`; no ratings or explicit feedback needed
- **Explainable scoring** — probabilities are derived through mathematical optimization under RF monotonicity constraints, making every score fully traceable and auditable; 3D surface visualization further supports intuitive understanding
- **Probabilistic output** — revisit probabilities serve as preference scores, enabling expected value calculations and probabilistic ranking of recommendations
- **Extensible** — the user–item probability matrix produced by `transform()` can be directly used as input to collaborative filtering or other downstream recommendation models

## Installation

```bash
pip install rfscorer
```

## Usage

```python
from rfscorer import RecencyFrequencyScorer
```

Prepare an interaction log with at least three columns: user ID, item ID, and timestamp.
Split it into a training set and a test set.

```python
df_train = ...  # training interaction log (columns: user, item, datetime)
df_test  = ...  # test interaction log  (columns: user, item, datetime)
```

| user  | item  | datetime   |
|-------|-------|------------|
| u_001 | i_032 | 2026-07-01 |
| u_001 | i_017 | 2026-07-03 |
| u_001 | i_032 | 2026-07-05 |
| u_002 | i_011 | 2026-07-02 |
| u_002 | i_058 | 2026-07-04 |

The same user-item pair may appear multiple times, representing repeat visits.

Instantiate the scorer, specifying the column names if they differ from the defaults (`user`, `item`, `datetime`).

```python
scorer = RecencyFrequencyScorer()
```

Call `fit()` to estimate empirical revisit probabilities from the training log.
Pass `target_date` as the split point: data up to `target_date` forms the observation window (default: 28 days back), and data after `target_date` forms the evaluation window (default: 7 days forward).

```python
scorer.fit(df_train, target_date="2026-07-07")
```

The empirical surface reflects raw revisit rates and may be irregular due to sparse data.

![empirical probability surface](https://raw.githubusercontent.com/jiro-iwanaga/rfscorer/main/img/empirical_probability_surface.png)

Optionally, call `optimize()` to smooth the surface under RF monotonicity constraints using convex quadratic programming.
`kind="mono"` enforces recency and frequency monotonicity.

```python
scorer.optimize(kind="mono")
```

![mono probability surface](https://raw.githubusercontent.com/jiro-iwanaga/rfscorer/main/img/mono_probability_surface.png)

`kind="mcc"` additionally adds convexity in recency and concavity in frequency, yielding a smoother surface.

```python
scorer.optimize(kind="mcc")
```

![mcc probability surface](https://raw.githubusercontent.com/jiro-iwanaga/rfscorer/main/img/mcc_probability_surface.png)

Call `transform()` to score each user-item pair in the test log.
It returns a DataFrame with columns `user`, `item`, `recency`, `frequency`, `probability`, and `order` (rank within each user, sorted by probability descending).
Pass `kind="empirical"`, `kind="mono"`, or `kind="mcc"` to select which probabilities to use.

```python
df_rec_mcc = scorer.transform(df_test, target_date="2026-07-07", kind="mcc")
```

| user   | item   | recency | frequency | probability | order |
|--------|--------|--------:|----------:|------------:|------:|
| u_001  | i_032  |       1 |         4 |      0.1167 |     1 |
| u_001  | i_017  |       2 |         3 |      0.0789 |     2 |
| u_001  | i_045  |       3 |         1 |      0.0248 |     3 |
| u_002  | i_011  |       1 |         2 |      0.0621 |     1 |
| u_002  | i_058  |       4 |         1 |      0.0182 |     2 |

Within each user, rows are sorted by `probability` descending; `order` represents the recommendation rank.

## Examples

- [examples/basic_usage.ipynb](examples/basic_usage.ipynb) — end-to-end walkthrough: load data, fit, optimize, transform, and evaluate

## References
- [Jiro Iwanaga, Naoki Nishimura, Noriyoshi Sukegawa, and Yuichi Takano, “Estimating product-choice probabilities from recency and frequency of page views,” Knowledge-Based Systems, Volume 99, 2016, Pages 157–167.](https://www.sciencedirect.com/science/article/abs/pii/S0950705116000848)

- [Jiro Iwanaga, Kyota Ishihara, Naoki Nishimura, and Ikki Tanaka, *Pythonではじめる数理最適化 ―ケーススタディでモデリングのスキルを身につけよう―*(in Japanese), Ohmsha, 2021.](https://www.ohmsha.co.jp/book/9784274231759/)
  - [Chapter 7: 商品推薦のための興味のスコアリング(in Japanese)](https://github.com/ohmsha/PyOptBook/tree/main/7.recommendation)

- [Jiro Iwanaga, Naoki Nishimura, Noriyoshi Sukegawa, and Yuichi Takano, “Improving collaborative filtering recommendations by estimating user preferences from clickstream data,” Electronic Commerce Research and Applications, Volume 37, Article 100877, 2019.](https://www.sciencedirect.com/science/article/abs/pii/S1567422319300547)


## Citing

If you use `rfscorer` in academic work, please cite the following paper:

Jiro Iwanaga, Naoki Nishimura, Noriyoshi Sukegawa, and Yuichi Takano,
"Estimating product-choice probabilities from recency and frequency of page views,"
*Knowledge-Based Systems*, Volume 99, 2016, Pages 157–167.

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

Jiro Iwanaga, Naoki Nishimura, Noriyoshi Sukegawa, and Yuichi Takano,
"Improving collaborative filtering recommendations by estimating user preferences from clickstream data,"
*Electronic Commerce Research and Applications*, Volume 37, Article 100877, 2019.

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
