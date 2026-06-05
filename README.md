# rfscorer

`rfscorer` is a Python package for Recency-Frequency based recommendation scoring.

It estimates **revisit probabilities** — the preference score for each user-item pair, forming a matrix analogous to a rating matrix — from interaction histories, using two simple but powerful behavioral signals: **recency**, which captures how recently a user interacted with an item, and **frequency**, which captures how often the user has interacted with it.

The package is designed for product recommendation and repeat-purchase modeling, especially in settings where interpretable scoring based on interaction history is preferred over black-box recommendation models.

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

df_train = ...  # training interaction log (columns: user, item, datetime)
df_test  = ...  # test interaction log  (columns: user, item, datetime)

scorer = RecencyFrequencyScorer()

# Estimate empirical revisit probabilities
scorer.fit(
    df_train,
    observation_period=("2026-07-01", "2026-07-07"),
    evaluation_period=("2026-07-08", "2026-07-08"),
)

# Score with empirical probabilities
# Returns DataFrame with user, item, recency, frequency, probability, order
df_rec_emp = scorer.transform(df_test, target_date="2026-07-07", kind="empirical")

# Estimate optimized probabilities under RF monotonicity constraints (optional)
scorer.optimize(kind="mono")

# Score with optimized probabilities
df_rec_mono = scorer.transform(df_test, target_date="2026-07-07", kind="mono")
```

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
