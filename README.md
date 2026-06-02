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

# df is a user-item interaction history
# df columns: user, item, datetime
scorer = RecencyFrequencyScorer() 

# Empirical probabilities estimated directly from observed interactions. 
scorer.fit(df) 
df_empirical = scorer.empirical_probability_.to_frame()

# Estimate optimized probabilities with additional RF constraints.
scorer.optimize() 
df_optimized = scorer.optimized_probability_.to_frame()
```

## References
- [Jiro Iwanaga, Naoki Nishimura, Noriyoshi Sukegawa, and Yuichi Takano, “Estimating product-choice probabilities from recency and frequency of page views,” Knowledge-Based Systems, Volume 99, 2016, Pages 157–167.](https://www.sciencedirect.com/science/article/abs/pii/S0950705116000848)

- [Jiro Iwanaga, Kyota Ishihara, Naoki Nishimura, and Ikki Tanaka, *Pythonではじめる数理最適化 ―ケーススタディでモデリングのスキルを身につけよう―*(in Japanese), Ohmsha, 2021.](https://www.ohmsha.co.jp/book/9784274231759/)
  - [Chapter 7: 商品推薦のための興味のスコアリング(in Japanese)](https://github.com/ohmsha/PyOptBook/tree/main/7.recommendation)

## License

MIT License
