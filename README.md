# RFscorer

[![CI](https://github.com/jiro-iwanaga/rfscorer/actions/workflows/ci.yml/badge.svg)](https://github.com/jiro-iwanaga/rfscorer/actions/workflows/ci.yml)
[![PyPI version](https://img.shields.io/pypi/v/rfscorer.svg)](https://pypi.org/project/rfscorer/)
[![Python Versions](https://img.shields.io/pypi/pyversions/rfscorer.svg)](https://pypi.org/project/rfscorer/)

`rfscorer` is a Python package for Recency-Frequency based recommendation scoring.

It estimates **product-choice probabilities** — the preference score for each user-item pair, forming a matrix like a rating matrix — from interaction history using two key signals: **recency** (how recently a user interacted with an item) and **frequency** (how often). You can configure which events to predict (revisits, purchases, conversions, etc.) using your evaluation data.

The package is designed for product recommendation and repeat-engagement modeling, especially when you prefer interpretable scoring based on interaction history over black-box models.

> Note: In this package, **RF** stands for **Recency-Frequency**, not Random Forest.

## Features

- **scikit-learn-style API** — familiar `fit()` / `transform()` interface makes it easy to integrate into existing data science workflows
- **Minimal data requirements** — works with any interaction log with three columns: user, item, and timestamp; no ratings or explicit feedback required
- **Explainable scoring** — probabilities are computed using optimization with Recency-Frequency monotonicity constraints, making every score fully traceable and auditable; 3D surface visualization further supports intuitive understanding
- **Probabilistic output** — product-choice probabilities work as preference scores, enabling expected value calculations and probabilistic ranking of recommendations
- **Extensible** — the probability matrix from `transform()` can be directly used as input to collaborative filtering or other downstream recommendation models
- **Calibration-free probabilities** — unlike typical ML models, probabilities are computed directly from interaction frequency without requiring calibration, making them reliable and easy to interpret.

## Installation

```bash
pip install rfscorer
```

## Usage

Below is a minimal example of building a model and scoring recommendations from an interaction log.
For complete, working code with data loading and evaluation, see [examples/basic_usage.ipynb](examples/basic_usage.ipynb).

### Minimal Example

```python
import pandas as pd
from rfscorer import RecencyFrequencyScorer, split_by_date

# Load your interaction log
df = ...  # columns: user, item, datetime

# Split by target date
target_date = "2026-07-07"
df_obs, df_eval = split_by_date(df, target_date)  # default: obs=28 days, eval=7 days

# Fit and optimize
scorer = RecencyFrequencyScorer()
scorer.fit(df_obs, df_eval)
scorer.optimize(kind="mono")

# Score recommendations (on test data)
df_test = ...  # test data (columns: user, item, datetime)
df_test_obs, _ = split_by_date(df_test, target_date)
df_scores = scorer.transform(df_test_obs, target_date, kind="mono")
```

| user   | item   | recency | frequency | probability | order |
|--------|--------|--------:|----------:|------------:|------:|
| u_001  | i_032  |       1 |         4 |      0.1167 |     1 |
| u_001  | i_017  |       2 |         3 |      0.0789 |     2 |
| u_001  | i_045  |       3 |         1 |      0.0248 |     3 |
| u_002  | i_011  |       1 |         2 |      0.0621 |     1 |
| u_002  | i_058  |       4 |         1 |      0.0182 |     2 |

The `probability` score determines recommendation rank. For each user, recommend items from highest to lowest probability. Because each score is a probability, you can also calculate expected values (e.g., expected revenue per recommendation). The `order` column makes it easy to implement business rules (e.g., "recommend top 2 items per user").     

### Visualization: Comparing Optimization Approaches
While the package supports many optimization approaches, here we visualize three key methods: 

```python
scorer.plot_probability_surface(kind="emp")  # empirical (raw rates)

scorer.optimize(kind="mono")  # RF monotonicity
scorer.plot_probability_surface(kind="mono")

scorer.optimize(kind="mcc")   # convex in R, concave in F
scorer.plot_probability_surface(kind="mcc")
```

<table>
  <tr>
    <td><img src="https://raw.githubusercontent.com/jiro-iwanaga/rfscorer/main/img/surface_emp_probability.png" width="300"/></td>
    <td><img src="https://raw.githubusercontent.com/jiro-iwanaga/rfscorer/main/img/surface_mono_probability.png" width="300"/></td>
    <td><img src="https://raw.githubusercontent.com/jiro-iwanaga/rfscorer/main/img/surface_mcc_probability.png" width="300"/></td>
  </tr>
  <tr>
    <td align="center"><i>Empirical</i></td>
    <td align="center"><i>Monotonicity</i></td>
    <td align="center"><i>Monotonicity-Convex-Concave</i></td>
  </tr>
</table>

Each surface reflects different assumptions about **recency** (how recently a user interacted) and **frequency** (how often):

- **Empirical**: Raw event rates; noisy and may violate monotonicity, sometimes recommending products in unnatural order.                  
- **Monotonicity**: Enforces monotonic relationships, ensuring products are recommended in natural and stable order.
- **Monotonicity-Convex-Concave**: Adds smoothness constraints with monotonically decreasing slopes in recency, producing the smoothest surface. Note: stronger constraints may overfit to training data; validate on test data.                            

## Examples

- [examples/basic_usage.ipynb](examples/basic_usage.ipynb) — end-to-end walkthrough: load data, fit, optimize, transform, and evaluate recommendation quality (precision, recall, F1 at each rank cutoff)

## References
- [Jiro Iwanaga, Naoki Nishimura, Noriyoshi Sukegawa, and Yuichi Takano, “Estimating product-choice probabilities from recency and frequency of page views,” Knowledge-Based Systems, Volume 99, 2016, Pages 157–167.](https://www.sciencedirect.com/science/article/abs/pii/S0950705116000848)

- [Jiro Iwanaga, Kyota Ishihara, Naoki Nishimura, and Ikki Tanaka, *Pythonではじめる数理最適化 ―ケーススタディでモデリングのスキルを身につけよう―*(in Japanese), Ohmsha, 2021.](https://www.ohmsha.co.jp/book/9784274231759/)
  - [Chapter 7: 商品推薦のための興味のスコアリング(in Japanese)](https://github.com/ohmsha/PyOptBook/tree/main/7.recommendation)

- [Jiro Iwanaga, Naoki Nishimura, Noriyoshi Sukegawa, and Yuichi Takano, “Improving collaborative filtering recommendations by estimating user preferences from clickstream data,” Electronic Commerce Research and Applications, Volume 37, Article 100877, 2019.](https://www.sciencedirect.com/science/article/abs/pii/S1567422319300547)


## Citation

If you use `rfscorer` in academic work, you can cite it as follows in the body of your paper:

> We used `rfscorer` (Iwanaga et al., 2016), a Python library for Recency-Frequency
> based recommendation scoring for product recommendation.¹
>
> ¹ https://github.com/jiro-iwanaga/rfscorer

The full reference is:

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
