# 機能設計書

## 概要

`rfscorer` は、ユーザー × 商品の閲覧履歴から商品の再閲覧確率を推定する。  
推定には2段階のアプローチをとる。

1. **経験的再閲覧確率の推定**: 観測期間に最新度 $r$、頻度 $f$の商品が、評価期間で再閲覧される割合とする
2. **最適化再閲覧確率の推定**: 経験的再閲覧確率を用いて、RF 制約と最小二乗誤差を目的関数にもつ凸2次計画問題を解いて推定する

## 数理モデル

### 用語定義

| 記号 | 説明 |
|------|------|
| $U$ | ユーザーのリスト。$u$はユーザーを表す。 |
| $I$ | 商品のリスト。$i$は商品を表す。 |
| $R$ | 観測期間の最新度のリスト(1以上の連続自然数の集合)。$r$は最新度を表す。 |
| $F$ | 観測期間の頻度のリスト(1以上の連続自然数の集合)。$f$は頻度を表す。 |
| $n_{r,f}$ | 観測期間で最新度 $r$、頻度 $f$ の商品が評価期間で再閲覧される頻度合計 |
| $N_{r,f}$ | 観測期間で最新度 $r$、頻度 $f$ の商品の頻度合計 |
| $p_{r,f}$ | 観測期間で最新度 $r$、頻度 $f$ の商品の評価期間における経験的再閲覧確率 |
| $x_{r,f}$ | 観測期間で最新度 $r$、頻度 $f$ の商品の評価期間における最適化再閲覧確率 |

### 経験的再閲覧確率の推定

「観測期間で最新度 $r$、頻度 $f$ の商品が評価期間で再閲覧される頻度合計」を
「観測期間で最新度 $r$、頻度 $f$ の商品の頻度合計」で割った値を経験的再閲覧確率とする。

$$p_{r,f} := \frac{n_{r,f}}{N_{r,f}}\ \ \  (r\in R, f\in F)$$

### 最適化再閲覧確率の推定

経験的確率 $p_{r,f}$ を基準として、以下の RF 単調性制約を満たす確率 $x_{r,f}$ を求める。

**RF制約**
- **Recency 制約（最新度の単調性）**  
最近閲覧した商品ほど再閲覧確率が高い。
$$r < r' \implies x_{r,f} \geq x_{r',f}\ \ \ (r, r'\in R) $$
- **Frequency 制約（頻度の単調性）**  
頻度が高い（閲覧数が多い）商品ほど再閲覧確率が高い。

$$f < f' \implies x_{r,f} \leq x_{r',f}\ \ \ (f, f'\in F) $$

**目的関数(最小二乗誤差)**
$$\sum_{r\in R, f\in F} N_{r,f} \cdot(p_{r,f} - x_{r,f})^2$$


## クラス仕様

### `RecencyFrequencyScorer`

#### コンストラクタ

```python
RecencyFrequencyScorer(df, user_col="user", item_col="item", datetime_col="datetime")
```

| パラメータ | 型 | デフォルト | 説明 |
|-----------|-----|-----------|------|
| `df` | `pd.DataFrame` | — | 閲覧履歴 |
| `user_col` | `str` | `"user"` | ユーザー識別子のカラム名 |
| `item_col` | `str` | `"item"` | 商品識別子のカラム名 |
| `datetime_col` | `str` | `"datetime"` | 閲覧日付のカラム名 |

内部では指定されたカラムを `user`・`item`・`datetime` に正規化して保持する（`interaction_log` 属性）。

#### メソッド

##### `fit(observation_period, evaluation_period, recency_limit=None, frequency_limit=None)`

観測期間・評価期間に基づき、$(r, f)$ 別の経験的再閲覧確率を推定する。

| パラメータ | 型 | デフォルト | 説明 |
|-----------|-----|-----------|------|
| `observation_period` | `tuple[str \| datetime, str \| datetime]` | — | 観測期間の開始日・終了日 |
| `evaluation_period` | `tuple[str \| datetime, str \| datetime]` | — | 評価期間の開始日・終了日 |
| `recency_limit` | `int \| None` | `None` | 最大最新度。`None` の場合、累積再閲覧数の分布から `RECENCY_LIMIT_RATE` に基づいて自動決定 |
| `frequency_limit` | `int \| None` | `None` | 最大頻度。`None` の場合、累積再閲覧数の分布から `FREQUENCY_LIMIT_RATE` に基づいて自動決定 |

戻り値: `self`

##### `predict(r, f, kind='empirical')`

指定した最新度 $r$・頻度 $f$ の再閲覧確率を返す。

| パラメータ | 型 | デフォルト | 説明 |
|-----------|-----|-----------|------|
| `r` | `int` | — | 最新度ランク（1以上） |
| `f` | `int` | — | 頻度（1以上） |
| `kind` | `str` | `'empirical'` | `'empirical'` または `'optimized'` |

戻り値: `float`

##### `transform(df, target_date, kind='empirical', user_col=None, item_col=None, datetime_col=None)`

入力 DataFrame の各 user×item ペアに最新度・頻度・再閲覧確率・順位を付与して返す。

| パラメータ | 型 | デフォルト | 説明 |
|-----------|-----|-----------|------|
| `df` | `pd.DataFrame` | — | スコアリング対象の閲覧履歴 |
| `target_date` | `str \| datetime` | — | 最新度・頻度の計算基準日 |
| `kind` | `str` | `'empirical'` | `'empirical'` または `'optimized'` |
| `user_col` | `str \| None` | `None` | ユーザーカラム名（省略時は `__init__` のデフォルト） |
| `item_col` | `str \| None` | `None` | 商品カラム名（省略時は `__init__` のデフォルト） |
| `datetime_col` | `str \| None` | `None` | 日付カラム名（省略時は `__init__` のデフォルト） |

戻り値: `pd.DataFrame`（カラム: `user`, `item`, `recency`, `frequency`, `probability`, `order`）

##### `evaluate(df_rec, UIrevisit, order=1, user_col=None, item_col=None)`

推薦結果と正解データを比較し、各順位カットオフでの評価指標を返す。
`df_rec` の user/item 列と `UIrevisit` の要素は内部で `str` にキャストして比較する。

| パラメータ | 型 | デフォルト | 説明 |
|-----------|-----|-----------|------|
| `df_rec` | `pd.DataFrame` | — | `transform()` の出力 |
| `UIrevisit` | `set` | — | 実際に再閲覧された `(user, item)` ペアの集合 |
| `order` | `int` | `1` | 評価する最大推薦順位 |

戻り値: `pd.DataFrame`（カラム: `order`, `n_recommended`, `n_hit`, `precision`, `recall`, `f1`, `recall_norm`, `f1_norm`）

##### `optimize()`

RF 単調性制約（Recency・Frequency）を満たす最適化再閲覧確率を推定する。`fit()` の後に呼び出す。
内部で `optimizer.py` の `RFOptimizer` を使用して凸2次計画問題を解く。

戻り値: `self`

##### `export_probability_csv(kind='empirical', path=None)`

再閲覧確率を CSV ファイルに書き出す。

| パラメータ | 型 | デフォルト | 説明 |
|-----------|-----|-----------|------|
| `kind` | `str` | `'empirical'` | `'empirical'`・`'optimized'`・`'all'` のいずれか。`'all'` は両者をマージして出力（カラム: `empirical_probability`, `optimized_probability`） |
| `path` | `str \| None` | `None` | 出力先。`None` の場合カレントディレクトリに `{kind}_probability.csv` を出力。ディレクトリを指定した場合はそのディレクトリにデフォルトファイル名で出力 |

戻り値: なし

##### `plot_probability_surface(kind='empirical', path=None)`

再閲覧確率を3次元ワイヤーフレームで可視化し PNG ファイルに保存する。

| パラメータ | 型 | デフォルト | 説明 |
|-----------|-----|-----------|------|
| `kind` | `str` | `'empirical'` | `'empirical'` または `'optimized'` |
| `path` | `str \| None` | `None` | 出力先。`None` の場合カレントディレクトリに `{kind}_probability_surface.png` を出力。ディレクトリを指定した場合はそのディレクトリにデフォルトファイル名で出力 |

戻り値: なし

##### `show()`

`fit()` 後の集計情報（レコード数・cv 数・期間・上限値）を標準出力に表示する。デバッグ・動作確認用。

戻り値: なし

#### 属性

| 属性 | 型 | 説明 | 利用可能なタイミング |
|------|-----|------|-----------------|
| `interaction_log` | `pd.DataFrame` | 正規化済み閲覧履歴（カラム: `user`, `item`, `datetime`） | `__init__()` 後 |
| `recency_limit` | `int` | 最新度の上限値 | `fit()` 後 |
| `frequency_limit` | `int` | 頻度の上限値 | `fit()` 後 |
| `R` | `list[int]` | 最新度のリスト（`range(1, recency_limit+1)`） | `fit()` 後 |
| `F` | `list[int]` | 頻度のリスト（`range(1, frequency_limit+1)`） | `fit()` 後 |
| `RF2N` | `dict` | `(r, f)` → サンプル数 $N_{r,f}$ のマッピング | `fit()` 後 |
| `RF2CV` | `dict` | `(r, f)` → cv 数 $n_{r,f}$ のマッピング | `fit()` 後 |
| `RF2Prob` | `dict` | `(r, f)` → 経験的再閲覧確率 $p_{r,f}$ のマッピング | `fit()` 後 |
| `empirical_probability_` | `pd.DataFrame` | 経験的再閲覧確率（カラム: `recency`, `frequency`, `N`, `cv`, `probability`） | `fit()` 後 |
| `empirical_probability_table_` | `pd.DataFrame` | 経験的再閲覧確率（横持ち。インデックス: `recency`、カラム: `frequency`） | `fit()` 後 |
| `empirical_probability_dict_` | `dict` | 経験的再閲覧確率（キー: `(r, f)`、値: `probability`） | `fit()` 後 |
| `optimized_probability_` | `pd.DataFrame` | 最適化再閲覧確率（カラム: `recency`, `frequency`, `probability`） | `optimize()` 後 |
| `optimized_probability_table_` | `pd.DataFrame` | 最適化再閲覧確率（横持ち。インデックス: `recency`、カラム: `frequency`） | `optimize()` 後 |
| `optimized_probability_dict_` | `dict` | 最適化再閲覧確率（キー: `(r, f)`、値: `probability`） | `optimize()` 後 |
| `record_num` | `int` | 全閲覧履歴のレコード数 | `__init__()` 後 |
| `record_num_obs` | `int` | 観測期間のレコード数 | `fit()` 後 |
| `record_num_eval` | `int` | 評価期間のレコード数 | `fit()` 後 |
| `record_num_target_org` | `int` | フィルタリング前の分析対象レコード数 | `fit()` 後 |
| `record_num_target` | `int` | フィルタリング後の分析対象レコード数 | `fit()` 後 |
| `total_cv_org` | `int` | フィルタリング前の cv 数 | `fit()` 後 |
| `total_cv` | `int` | フィルタリング後の cv 数 | `fit()` 後 |

## データフロー

```
入力 DataFrame (任意のカラム名)
        │
        ▼  __init__()
user / item / datetime に正規化 → interaction_log
        │
        ▼  fit(observation_period, evaluation_period)
観測期間・評価期間でフィルタ
r（最新度ランク）・f（頻度）を算出
(r, f) 別に n_{r,f}・N_{r,f} を集計
p_{r,f} = n_{r,f} / N_{r,f}
        ▼
empirical_probability_ / empirical_probability_table_ / empirical_probability_dict_
RF2N / RF2CV / RF2Prob
        │
        ├─  predict(r, f, kind)  ─→ 特定 (r, f) の再閲覧確率を返す
        │
        ├─  transform(df, target_date, kind)  ─→ user×item に r・f・確率・順位を付与
        │
        ├─  export_probability_csv(kind, path)  ─→ 確率テーブルを CSV に書き出す
        │
        ▼  optimize()  ← RFOptimizer (optimizer.py) に委譲
RF 制約付き凸2次計画問題を求解
        ▼
optimized_probability_ / optimized_probability_table_ / optimized_probability_dict_
        │
        └─  export_probability_csv(kind='all', path)  ─→ 経験・最適化を併記した CSV を書き出す
```

## 入出力例

```python
import pandas as pd
from rfscorer import RecencyFrequencyScorer

df = pd.read_csv("examples/access_log.csv")
# access_log.csv のカラム: user_id, item_id, date

scorer = RecencyFrequencyScorer(df, user_col="user_id", item_col="item_id", datetime_col="date")

scorer.fit(
    observation_period=("2015-07-02", "2015-07-06"),
    evaluation_period=("2015-07-07", "2015-07-08"),
)
scorer.empirical_probability_

df_rec = scorer.transform(df, target_date="2015-07-06")
prob = scorer.predict(r=1, f=3)
```

