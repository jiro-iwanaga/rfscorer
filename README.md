# RFscorer

[![CI](https://github.com/jiro-iwanaga/rfscorer/actions/workflows/ci.yml/badge.svg)](https://github.com/jiro-iwanaga/rfscorer/actions/workflows/ci.yml)
[![PyPI version](https://img.shields.io/pypi/v/rfscorer.svg)](https://pypi.org/project/rfscorer/)
[![Python Versions](https://img.shields.io/pypi/pyversions/rfscorer.svg)](https://pypi.org/project/rfscorer/)

[日本語 README](#RFscorer-日本語readme)

`rfscorer` is a Python package for Recency-Frequency based recommendation scoring.

> Note: In this package, **RF** stands for **Recency-Frequency**, not Random Forest.

It estimates recommendation scores (product-choice probabilities) for items a user has interacted with, based on two signals: **recency** (time since last interaction) and **frequency** (number of interactions). You can choose any event as the prediction target (revisits, purchases, conversions, etc.).

In product recommendation, the key question is which of the items a user has previously interacted with should be prioritized. For example, consider the following comparisons:

- **[Decide by frequency]** Item A viewed once 🆚 Item B viewed **twice** ▶ recommend **Item B**, viewed more often
- **[Decide by recency]** Item A viewed **1 day ago** 🆚 Item B viewed 2 days ago ▶ recommend **Item A**, viewed more recently
- **[Trade-off]** Item A viewed once **1 day ago** 🆚 Item B viewed **twice** 2 days ago ▶ **hard to judge by intuition**

For such non-trivial comparisons, `rfscorer` uses mathematical optimization to estimate recommendation scores that satisfy the natural monotonicity of recency and frequency. This gives a data-driven, natural recommendation order over the items a user has previously interacted with.

Beyond serving as a standalone recommendation ranking, `rfscorer`'s scores can also be used as input to downstream models — for example, as a rating matrix for collaborative filtering or as features for ML models.

> 📄 **Based on the paper:** Jiro Iwanaga, Naoki Nishimura, Noriyoshi Sukegawa, and Yuichi Takano, “Estimating product-choice probabilities from recency and frequency of page views,” *Knowledge-Based Systems*, Vol. 99, 2016, pp. 157–167. [[paper]](https://www.sciencedirect.com/science/article/abs/pii/S0950705116000848)

## Features

| Feature | Description |
|---------|-------------|
| **scikit-learn style** | `fit()` / `transform()` interface |
| **Minimal data** | works with any behavior history with three columns: `user`, `item`, `datetime` |
| **Explainable** | scores are estimated by mathematical optimization under RF monotonicity, making the reasoning behind each recommendation easy to explain |
| **Stable probability estimation** | product-choice probabilities are estimated directly from recency and frequency, avoiding the instability of converting ML model outputs to a probability scale |
| **Downstream use** | usable not only as a standalone recommendation score but also as a rating matrix for collaborative filtering or as features for ML models |
| **Rich diagnostics & visualization** | extensive statistical outputs and visualization features let practitioners explain results in their work and researchers report them in papers |

## Installation

```bash
pip install rfscorer
```

## Usage

Below is a minimal example of building a model and scoring recommendations from a behavior history (for complete, runnable code, see [Examples](#examples)).

### Minimal Example

```python
import pandas as pd
from rfscorer import RecencyFrequencyScorer, split_by_date

# Load your behavior history
df = ...  # columns: user, item, datetime

# Split by target date
target_date = "2026-07-07"
df_obs, df_gt = split_by_date(df, target_date, 7, 1)  # observation: 7 days, ground truth: 1 day

# Fit and optimize
scorer = RecencyFrequencyScorer()
scorer.fit(df_obs, df_gt)
scorer.optimize(kind="mono")

# Score recommendations (on test data)
df_test = ...  # test data (columns: user, item, datetime)
df_test_obs, _ = split_by_date(df_test, target_date, 7, 1)
df_scores = scorer.transform(df_test_obs, target_date, kind="mono")
```

| user   | item   | recency | frequency | probability | order |
|--------|--------|--------:|----------:|------------:|------:|
| u_001  | i_032  |       1 |         4 |      0.1167 |     1 |
| u_001  | i_017  |       2 |         3 |      0.0789 |     2 |
| u_001  | i_045  |       3 |         1 |      0.0248 |     3 |
| u_002  | i_011  |       1 |         2 |      0.0621 |     1 |
| u_002  | i_058  |       4 |         1 |      0.0182 |     2 |

Recommend items to each user from highest to lowest `probability`. Since scores are probabilities, expected value calculations are straightforward (e.g., expected revenue per recommendation). Use the `order` column to apply business rules (e.g., recommend the top 2 items per user).

### Visualization: Comparing Optimization Approaches
The package supports many optimization approaches. Here we visualize three representative methods:

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

Each surface clearly captures how the product-choice probability behaves with respect to **recency** (time since last interaction) and **frequency** (number of interactions):

- **Empirical**: Raw probabilities without constraints. Noisy and may violate monotonicity, sometimes recommending items in unnatural order.
- **Monotonicity**: Probabilities with RF monotonicity constraints. Guarantees items are recommended in natural order.
- **Monotonicity-Convex-Concave**: Probabilities with RF monotonicity and convexity-concavity constraints. Produces the smoothest surface.

## Examples

- [examples/tutorial_beginner_en.ipynb](examples/tutorial_beginner_en.ipynb) — end-to-end walkthrough: load data, fit, optimize, visualize, transform, and evaluate
- [examples/tutorial_practical_en.ipynb](examples/tutorial_practical_en.ipynb) — practical workflow: chronological train/test split, build the various models, compare accuracy, and save/load the model
- [examples/tutorial_advanced_fit_rolling_en.ipynb](examples/tutorial_advanced_fit_rolling_en.ipynb) — advanced workflow: time-series rolling training with `fit_rolling()` to stabilize empirical probabilities across multiple reference dates

For the complete list of tutorials, see [examples/](examples/).

## References

- [Jiro Iwanaga, Naoki Nishimura, Noriyoshi Sukegawa, and Yuichi Takano, “Estimating product-choice probabilities from recency and frequency of page views,” Knowledge-Based Systems, Volume 99, 2016, Pages 157–167.](https://www.sciencedirect.com/science/article/abs/pii/S0950705116000848)

- [Jiro Iwanaga, Kyota Ishihara, Naoki Nishimura, and Ikki Tanaka, *Pythonではじめる数理最適化 ―ケーススタディでモデリングのスキルを身につけよう―*(in Japanese), Ohmsha, 2021.](https://www.ohmsha.co.jp/book/9784274231759/)
  - [Chapter 7: 商品推薦のための興味のスコアリング(in Japanese)](https://github.com/ohmsha/PyOptBook/tree/main/7.recommendation)

- [Jiro Iwanaga, Naoki Nishimura, Noriyoshi Sukegawa, and Yuichi Takano, “Improving collaborative filtering recommendations by estimating user preferences from clickstream data,” Electronic Commerce Research and Applications, Volume 37, Article 100877, 2019.](https://www.sciencedirect.com/science/article/abs/pii/S1567422319300547)


## Citation

<details>
<summary>Show citation & BibTeX</summary>

If you use `rfscorer` in academic work, you can cite it as follows in the body of your paper:

> We used `rfscorer` (Iwanaga et al., 2016), a Python library for Recency-Frequency-based recommendation scoring.¹
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

If you also use the probability matrix as input to a collaborative filtering model or as ML features, please also cite:

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

</details>

## License

MIT License


-----
# RFscorer (日本語README)

[English README](#RFscorer)

`rfscorer` は、最新度（Recency）と頻度（Frequency）に基づいて、ユーザーが過去に接触した商品の推薦スコア（商品選択確率）を推定する Python パッケージです。

> 本パッケージでは **RF** は **Recency-Frequency（最新度-頻度）** を表します

ユーザーの行動履歴から、各商品の **選択されやすさ** を表す推薦スコア（商品選択確率）を推定します。スコアは、次の2つの行動シグナルに基づいて計算されます。

- **最新度（recency）**：最後に接触してからの経過時間。最近接触した商品ほど関心が高い傾向。
- **頻度（frequency）**：商品への接触回数。何度も接触した商品ほど関心が高い傾向。

予測対象は、再閲覧、購買、コンバージョンなど用途に応じて自由に設定できます。

商品推薦は、過去の行動履歴を用いて、ユーザーが過去に接触した商品の中でどれを優先して推薦するかを決める問題と捉えられます。
たとえば、次のような比較です。

- **【頻度で判断】** 1回閲覧した商品A　🆚　**2回**閲覧した商品B　▶　たくさん閲覧した **商品B** を推薦
- **【最新度で判断】** **1日前**に閲覧した商品A　🆚　2日前に閲覧した商品C　▶　最近閲覧した **商品A** を推薦
- **【トレードオフ】** **1日前**に1回閲覧した商品A　🆚　2日前に**2回**閲覧した商品D　▶　**直感では判断できない**

`rfscorer` は、このような非自明な比較に対して、数理最適化により、最新度と頻度の交互作用を考慮しつつ、両者の自然な単調性を満たす推薦スコアを推定します。
これにより、ユーザーが過去に接触した商品に対して、データに基づく自然な推薦順位を与えることができます。

> 📄 **Based on the paper:** Jiro Iwanaga, Naoki Nishimura, Noriyoshi Sukegawa, and Yuichi Takano, “Estimating product-choice probabilities from recency and frequency of page views,” *Knowledge-Based Systems*, Vol. 99, 2016, pp. 157–167. [[論文]](https://www.sciencedirect.com/science/article/abs/pii/S0950705116000848)

また、`rfscorer` が出力する商品選択確率は、下流のモデルの入力（協調フィルタリングの評価値行列・機械学習モデルの特徴量）としても有効です。
最新度と頻度の交互作用が反映された有用な特徴量によるモデル構築ができます。

<p align="center">
  <img src="https://raw.githubusercontent.com/jiro-iwanaga/rfscorer/main/img/recommendation_system_architecture_using_rfscoring.png" width="720"/>
</p>
<p align="center"><i>RFスコアリングを用いた推薦システム構成</i></p>

> 📄 **Based on the paper:** Jiro Iwanaga, Naoki Nishimura, Noriyoshi Sukegawa, and Yuichi Takano, “Improving collaborative filtering recommendations by estimating user preferences from clickstream data,” *Electronic Commerce Research and Applications*, Vol. 37, Article 100877, 2019. [[論文]](https://www.sciencedirect.com/science/article/abs/pii/S1567422319300547)


## パッケージの特徴

| 特徴 | 説明 |
|------|------|
| **scikit&#8209;learn&nbsp;ラ&#8288;イ&#8288;ク** | `fit()` / `transform()` によるインターフェースを提供 |
| **最&#8288;小&#8288;限&#8288;の&#8288;デ&#8288;ー&#8288;タ&#8288;要&#8288;件** | 入力データは、`user`、`item`、`datetime` の３カラムをもつ行動履歴 |
| **説&#8288;明&#8288;可&#8288;能&#8288;性** | 数理最適化によりRF単調性を満たすスコアを推定するため、推薦理由を説明しやすい |
| **安&#8288;定&#8288;し&#8288;た&#8288;確&#8288;率&#8288;推&#8288;定** | 最新度と頻度から商品選択確率を直接推定するため、機械学習モデルの出力を確率スケールへ変換する際の不安定さを回避できる |
| **下&#8288;流&#8288;モ&#8288;デ&#8288;ル&#8288;へ&#8288;の&#8288;活&#8288;用** | 単独の推薦スコアとしてだけでなく、協調フィルタリングの評価値行列や機械学習モデルの特徴量としても利用可能 |
| **豊&#8288;富&#8288;な&#8288;診&#8288;断&#8288;と&#8288;可&#8288;視&#8288;化** | 各種統計量の出力や可視化機能が充実。実務家は業務で説明しやすく、研究者は分析結果を論文に記載しやすい |

## インストール

```bash
pip install rfscorer
```

## 使い方

以下は、行動履歴からモデル構築と推薦スコア（商品選択確率）算出までを行う最小限の例です（実行可能な完全版は[サンプル](#サンプル)を参照）。

### 最小限の例

```python
import pandas as pd
from rfscorer import RecencyFrequencyScorer, split_by_date

# 行動履歴の読み込み
df = ...  # カラム: user, item, datetime

# 基準日で観測データ・正解データに分割
target_date = "2026-07-07"
df_obs, df_gt = split_by_date(df, target_date, 7, 1) # 観測データ7日間・正解データ1日間

# モデル構築と最適化
scorer = RecencyFrequencyScorer()
scorer.fit(df_obs, df_gt)
scorer.optimize(kind="mono")

# 推薦スコアを算出(テストデータ)
df_test = ...  # テストデータ（カラム: user, item, datetime）
df_test_obs, _ = split_by_date(df_test, target_date, 7, 1)
df_scores = scorer.transform(df_test_obs, target_date, kind="mono")
```

| user   | item   | recency | frequency | probability | order |
|--------|--------|--------:|----------:|------------:|------:|
| u_001  | i_032  |       1 |         4 |      0.1167 |     1 |
| u_001  | i_017  |       2 |         3 |      0.0789 |     2 |
| u_001  | i_045  |       3 |         1 |      0.0248 |     3 |
| u_002  | i_011  |       1 |         2 |      0.0621 |     1 |
| u_002  | i_058  |       4 |         1 |      0.0182 |     2 |

各ユーザーに対して、商品選択確率（`probability` ）の高い順に商品を推薦します。推薦スコアが確率値であるため、期待値計算(例：推薦結果に対する期待収益の計算)が容易です。`order` 列を使えば、業務ルール(例：「各ユーザーに上位2個の商品を推薦する」)を簡単に実装できます。

### 可視化：最適化手法の比較
本パッケージは多くの最適化アプローチをサポートしています。ここでは代表的な3つの手法を可視化します。

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

各グラフは、**最新度（recency）**（ユーザーが商品に接触してからの経過時間）と **頻度（frequency）**（接触回数）に基づく商品選択確率の特徴をよく表しています：

- **Empirical（経験確率）**: 制約を課していない商品選択確率。ノイズにより単調性を満たさないため不自然な順序で商品を推薦する場合がある。
- **Monotonicity（単調性）**: RF単調性制約を課した商品選択確率。商品を自然な順序で推薦することを保証する。
- **Monotonicity-Convex-Concave（単調性＋凸凹）**: RF単調性制約と凹凸性制約を課した商品選択確率。最も滑らかなグラフを生成する

## サンプル

- [examples/tutorial_beginner_ja.ipynb](examples/tutorial_beginner_ja.ipynb) — 初級編では、最小限の利用方法を紹介します。データロード、モデル構築・最適化・可視化、推薦スコア算出、精度評価までのコードを紹介します。
- [examples/tutorial_practical_ja.ipynb](examples/tutorial_practical_ja.ipynb) — 実践編では、主要機能を紹介します。具体的には、時系列での訓練・テスト分割、各種モデル構築と精度比較、モデルの保存・ロードを紹介します。
- [examples/tutorial_advanced_fit_rolling_ja.ipynb](examples/tutorial_advanced_fit_rolling_ja.ipynb) — 応用編では、`fit_rolling()` を用いたローリング集計を扱います。複数の基準日にわたって集計することで経験的商品選択確率を安定させる方法を紹介します。

全チュートリアルの一覧は [examples/](examples/) を参照してください。

## 参考文献

- [Jiro Iwanaga, Naoki Nishimura, Noriyoshi Sukegawa, and Yuichi Takano, "Estimating product-choice probabilities from recency and frequency of page views," Knowledge-Based Systems, Volume 99, 2016, Pages 157–167.](https://www.sciencedirect.com/science/article/abs/pii/S0950705116000848)

- [岩永二郎・石原響太・西村直樹・田中一樹『Pythonではじめる数理最適化 ―ケーススタディでモデリングのスキルを身につけよう―』, オーム社, 2021.](https://www.ohmsha.co.jp/book/9784274231759/)
  - [第7章: 商品推薦のための興味のスコアリング](https://github.com/ohmsha/PyOptBook/tree/main/7.recommendation)

- [Jiro Iwanaga, Naoki Nishimura, Noriyoshi Sukegawa, and Yuichi Takano, "Improving collaborative filtering recommendations by estimating user preferences from clickstream data," Electronic Commerce Research and Applications, Volume 37, Article 100877, 2019.](https://www.sciencedirect.com/science/article/abs/pii/S1567422319300547)


## 引用について

<details>
<summary>引用方法と BibTeX を表示</summary>

学術論文等で `rfscorer` を利用する場合は、論文の引用と本Githubへのリンクを脚注を加え、本文中で以下のように引用できます：

> We used `rfscorer` (Iwanaga et al., 2016), a Python library for Recency-Frequency-based recommendation scoring.¹
>
> ¹ https://github.com/jiro-iwanaga/rfscorer

参考文献とBibTexは以下のとおりです：

- [Jiro Iwanaga, Naoki Nishimura, Noriyoshi Sukegawa, and Yuichi Takano, "Estimating product-choice probabilities from recency and frequency of page views," Knowledge-Based Systems, Volume 99, 2016, Pages 157–167.](https://www.sciencedirect.com/science/article/abs/pii/S0950705116000848)

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

さらに、商品選択確率行列を協調フィルタリングモデルの入力として利用する場合や機械学習の特徴量として利用する場合には、以下の文献も併せて引用してください：

- [Jiro Iwanaga, Naoki Nishimura, Noriyoshi Sukegawa, and Yuichi Takano, "Improving collaborative filtering recommendations by estimating user preferences from clickstream data," Electronic Commerce Research and Applications, Volume 37, Article 100877, 2019.](https://www.sciencedirect.com/science/article/abs/pii/S1567422319300547)


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

</details>

## ライセンス

MIT License
