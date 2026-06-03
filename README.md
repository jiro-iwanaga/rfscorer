# rfscorer

`rfscorer` is a Python package for Recency-Frequency based recommendation scoring.

It estimates recommendation scores and purchase probabilities from user-item interaction histories by using two simple but powerful behavioral signals: **recency**, which captures how recently a user interacted with an item, and **frequency**, which captures how often the user has interacted with it.

The package is designed for product recommendation and repeat-purchase modeling, especially in settings where interpretable scoring based on purchase or interaction history is preferred over black-box recommendation models.

> Note: In this package, **RF** stands for **Recency-Frequency**, not Random Forest.

## Installation

```bash
pip install rfscorer
```

## Usage

```python
from rfscorer import RecencyFrequencyScorer

df_train = ...  # training interaction log (columns: user, item, datetime)
df_test  = ...  # test interaction log  (columns: user, item, datetime)

scorer = RecencyFrequencyScorer(df_train)

# Estimate empirical revisit probabilities
scorer.fit(
    observation_period=("2026-07-01", "2026-07-07"),
    evaluation_period=("2026-07-08", "2026-07-08"),
)

# Score with empirical probabilities
# Returns DataFrame with recency, frequency, probability, order per user-item pair
df_rec_emp = scorer.transform(df_test, target_date="2026-07-07", kind="empirical")

# Estimate optimized probabilities under RF monotonicity constraints (optional)
scorer.optimize()

# Score with optimized probabilities
df_rec_opt = scorer.transform(df_test, target_date="2026-07-07", kind="optimized")
```

## References
- [Jiro Iwanaga, Naoki Nishimura, Noriyoshi Sukegawa, and Yuichi Takano, “Estimating product-choice probabilities from recency and frequency of page views,” Knowledge-Based Systems, Volume 99, 2016, Pages 157–167.](https://www.sciencedirect.com/science/article/abs/pii/S0950705116000848)

- [Jiro Iwanaga, Kyota Ishihara, Naoki Nishimura, and Ikki Tanaka, *Pythonではじめる数理最適化 ―ケーススタディでモデリングのスキルを身につけよう―*(in Japanese), Ohmsha, 2021.](https://www.ohmsha.co.jp/book/9784274231759/)
  - [Chapter 7: 商品推薦のための興味のスコアリング(in Japanese)](https://github.com/ohmsha/PyOptBook/tree/main/7.recommendation)

## License

MIT License
