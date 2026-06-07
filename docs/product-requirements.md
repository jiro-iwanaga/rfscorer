# プロダクト要求定義書

## 背景

EC サイトやコンテンツプラットフォームでは、ユーザーの閲覧履歴に基づいて商品推薦を行いたいというニーズがある。  一般的なブラックボックス型の推薦モデルは精度が高い一方、スコアの根拠を説明しにくく、運用・デバッグが困難な場合がある。

本パッケージは、Recency（最新度）と Frequency（頻度）という2つの行動シグナルのみを用いた、解釈可能な推薦スコアリング手法を提供する。手法の理論的基盤は以下の学術論文に基づく。

> Jiro Iwanaga, Naoki Nishimura, Noriyoshi Sukegawa, and Yuichi Takano,  
> "Estimating product-choice probabilities from recency and frequency of page views,"  
> *Knowledge-Based Systems*, Volume 99, 2016, Pages 157–167.

## 目的

RF スコアリング手法を Python パッケージとして PyPI に公開し、`pip install rfscorer` で誰でも利用できるようにする。

## ターゲットユーザー

- Python とデータ分析に習熟したデータサイエンティスト・機械学習エンジニア
- EC サイト・コンテンツプラットフォームの推薦システム担当者
- 解釈可能な推薦スコアを必要とするビジネスアナリスト

## 解決する課題

| 課題 | 説明 |
|------|------|
| 推薦スコアの算出 | ユーザー × 商品のインタラクション履歴から、各ユーザーが各商品を選択するスコアを推定したい（∝再閲覧確率） |
| 解釈可能性の確保 | スコアの根拠を説明できる手法が求められる |
| 実装コストの削減 | RF スコアリングを自前実装する手間をなくす |

## 機能要求

### 入力

- コンストラクタ
  - カラム名を `user_col`・`item_col`・`datetime_col` で指定する（デフォルト: `user`・`item`・`datetime`）
- `fit()` に渡す引数
  - `df`: ユーザー × 商品のインタラクション履歴 DataFrame。同一ユーザー × 商品の組み合わせが複数行存在することを想定（リピート閲覧）
  - `target_date`: 観測期間と評価期間の分割点となる基準日。観測期間は `target_date` まで（デフォルト: 28日遡る）、評価期間は `target_date` の翌日から（デフォルト: 7日分）
- 期間を明示的に指定したい場合は `fit_period()` を使用する
  - `observation_period`: 観測期間の開始日・終了日の tuple
  - `evaluation_period`: 評価期間の開始日・終了日の tuple

### 機能

| 機能 | 説明 |
|------|------|
| 経験的再閲覧確率の推定（`emp`） | 観測期間における最新度 $r$・頻度 $f$ 別に、評価期間での再閲覧比率を直接推定する |
| 周辺的経験的再閲覧確率の推定（`er` / `ef`） | `fit()` 時に自動計算。最新度方向（`er`）・頻度方向（`ef`）の周辺確率を RF グリッド全体にブロードキャストした確率面 |
| 1次元最適化再閲覧確率の推定（`mr` / `mf`） | 周辺確率を目標とした1次元の凸2次計画問題を解く。`mr` は Recency 単調性 + 凸性、`mf` は Frequency 単調性 + 凹性を制約として課す。結果を RF グリッド全体にブロードキャスト |
| 2次元最適化再閲覧確率の推定（`mono` / `mrc` / `mfc` / `mcc`） | RF 制約と最小二乗誤差を目的関数にもつ凸2次計画問題を解いて推定する。制約の組み合わせにより `mono`（単調性のみ）・`mrc`（+ Recency 凸性）・`mfc`（+ Frequency 凹性）・`mcc`（+ 両凹凸性）の4モデルを提供する |
| 狭義単調性（`eps` パラメータ） | `optimize(eps=ε)` に正の値を指定すると、隣接する最新度・頻度の確率値が必ず $\varepsilon$ 以上離れる狭義単調性制約を付与する。デフォルト（`eps=0.0`）は従来の弱単調性と等価 |

### 出力

| 属性 | 説明 |
|------|------|
| `empirical_probability_` | 経験的再閲覧確率（`emp`）。`pd.DataFrame`（カラム: `recency`, `frequency`, `N`, `cv`, `probability`）。`fit()` 後にアクセス可能 |
| `er_probability_` | 経験的 Recency 周辺確率（`er`）。`pd.DataFrame`（カラム: `recency`, `frequency`, `probability`）。`fit()` 後にアクセス可能 |
| `ef_probability_` | 経験的 Frequency 周辺確率（`ef`）。`pd.DataFrame`（カラム: `recency`, `frequency`, `probability`）。`fit()` 後にアクセス可能 |
| `mr_probability_` | mr モデル最適化再閲覧確率。`pd.DataFrame`（カラム: `recency`, `frequency`, `probability`）。`optimize(kind='mr')` 後にアクセス可能 |
| `mf_probability_` | mf モデル最適化再閲覧確率。`pd.DataFrame`（カラム: `recency`, `frequency`, `probability`）。`optimize(kind='mf')` 後にアクセス可能 |
| `mono_probability_` | mono モデル最適化再閲覧確率。`pd.DataFrame`（カラム: `recency`, `frequency`, `probability`）。`optimize(kind='mono')` 後にアクセス可能 |
| `mrc_probability_` | mrc モデル最適化再閲覧確率。`pd.DataFrame`（カラム: `recency`, `frequency`, `probability`）。`optimize(kind='mrc')` 後にアクセス可能 |
| `mfc_probability_` | mfc モデル最適化再閲覧確率。`pd.DataFrame`（カラム: `recency`, `frequency`, `probability`）。`optimize(kind='mfc')` 後にアクセス可能 |
| `mcc_probability_` | mcc モデル最適化再閲覧確率。`pd.DataFrame`（カラム: `recency`, `frequency`, `probability`）。`optimize(kind='mcc')` 後にアクセス可能 |

### kind エイリアス

短い正式名と長いエイリアス名の両方が使用可能。

| 正式名 | エイリアス |
|--------|-----------|
| `emp` | `empirical` |
| `er` | `empirical_recency` |
| `ef` | `empirical_frequency` |
| `mono` | `monotone` |
| `mr` | `monotone_recency` |
| `mf` | `monotone_frequency` |
| `mrc` | `monotone_recency_convex` |
| `mfc` | `monotone_frequency_concave` |
| `mcc` | `monotone_convex_concave` |

### API

```python
import pandas as pd
from rfscorer import RecencyFrequencyScorer

df = pd.read_csv("examples/access_log.csv")
scorer = RecencyFrequencyScorer(user_col="user_id", item_col="item_id", datetime_col="date")

scorer.fit(df, target_date="2015-07-06")
df_emp = scorer.empirical_probability_  # 経験的確率（2次元）
df_er  = scorer.er_probability_         # 経験的 Recency 周辺（fit 時に自動計算）
df_ef  = scorer.ef_probability_         # 経験的 Frequency 周辺（fit 時に自動計算）

scorer.optimize(kind="mr")             # 1次元: Recency 単調性 + 凸性（弱単調性）
df_mr = scorer.mr_probability_

scorer.optimize(kind="mf")             # 1次元: Frequency 単調性 + 凹性（弱単調性）
df_mf = scorer.mf_probability_

scorer.optimize(kind="mono")           # 2次元: 単調性のみ（弱単調性）
df_mono = scorer.mono_probability_

scorer.optimize(kind="mcc")            # 2次元: 単調性 + 両凹凸性
df_mcc = scorer.mcc_probability_

scorer.optimize(kind="mono", eps=1e-4) # 狭義単調性: 隣接値の差を 1e-4 以上に保証
df_mono_strict = scorer.mono_probability_
```

## 非機能要求

| 項目 | 要求 |
|------|------|
| 配布 | PyPI に公開し、`pip install rfscorer` でインストール可能 |
| API 設計 | scikit-learn スタイル（`fit` / `optimize`）に準拠し、既存ワークフローに組み込みやすくする |
| 解釈可能性 | スコアの算出根拠を説明できること。ブラックボックス化しない |
| ドキュメント | README および docstring で API・使用例を説明する |
| ライセンス | MIT License |

## 制約

- Python パッケージとして開発し、`uv` で依存関係を管理する
- ライブラリコードは `src/rfscorer/` に配置する
- 公開クラスは `from rfscorer import RecencyFrequencyScorer` でインポートできること
