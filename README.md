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
import pandas as pd
from rfscorer import RecencyFrequencyScorer

# df is a user-item interaction history
# Required columns: user, item, datetime (column names can be customized)
df = pd.read_csv("access_log.csv")
scorer = RecencyFrequencyScorer(df, user_col="user", item_col="item", datetime_col="datetime")

# Estimate empirical revisit probabilities from observed interactions.
scorer.fit(
    observation_period=("2026-07-01", "2026-07-07"),
    evaluation_period=("2026-07-08", "2026-07-08"),
)

# Access the full probability table
df_empirical = scorer.empirical_probability_

# Get revisit probability for a specific (recency, frequency) pair
prob = scorer.predict(r=1, f=3)
```

## References
- [Jiro Iwanaga, Naoki Nishimura, Noriyoshi Sukegawa, and Yuichi Takano, “Estimating product-choice probabilities from recency and frequency of page views,” Knowledge-Based Systems, Volume 99, 2016, Pages 157–167.](https://www.sciencedirect.com/science/article/abs/pii/S0950705116000848)

- [Jiro Iwanaga, Kyota Ishihara, Naoki Nishimura, and Ikki Tanaka, *Pythonではじめる数理最適化 ―ケーススタディでモデリングのスキルを身につけよう―*(in Japanese), Ohmsha, 2021.](https://www.ohmsha.co.jp/book/9784274231759/)
  - [Chapter 7: 商品推薦のための興味のスコアリング(in Japanese)](https://github.com/ohmsha/PyOptBook/tree/main/7.recommendation)

## License

MIT License
