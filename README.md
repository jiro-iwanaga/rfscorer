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
```

<!--
```python
from rfscorer import RecencyFrequencyScorer, RecencyScorer, FrequencyScorer
```
-->


## License

MIT License
