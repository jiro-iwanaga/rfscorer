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

- ユーザー × 商品のインタラクション履歴を表す DataFrame（コンストラクタに渡す）
  - カラム名は `user_col`・`item_col`・`datetime_col` 引数で指定する（デフォルト: `user`・`item`・`datetime`）
  - 同一ユーザー × 商品の組み合わせが複数行存在することを想定（リピート閲覧）
- 観測期間・評価期間（`fit()` に渡す）
  - `observation_period`: 観測期間の開始日・終了日の tuple
  - `evaluation_period`: 評価期間の開始日・終了日の tuple

### 機能

| 機能 | 説明 |
|------|------|
| 経験的再閲覧確率の推定 | 観測期間における最新度 $r$・頻度 $f$ 別に、評価期間での再閲覧比率を直接推定する |
| 最適化再閲覧確率の推定 | RF 制約（Recency・Frequency に関する単調性制約）と最小二乗誤差を目的関数にもつ凸2次計画問題を解いて推定する |

### 出力

| 属性 | 説明 |
|------|------|
| `empirical_probability_` | 経験的再閲覧確率。`pd.Series`（インデックス: `(r, f)`）。`fit()` 後にアクセス可能 |
| `optimized_probability_` | 最適化再閲覧確率。`pd.Series`（インデックス: `(r, f)`）。`optimize()` 後にアクセス可能 |

### API

```python
import pandas as pd
from rfscorer import RecencyFrequencyScorer

df = pd.read_csv("examples/access_log.csv")
scorer = RecencyFrequencyScorer(df, user_col="user_id", item_col="item_id", datetime_col="date")

scorer.fit(
    observation_period=("2015-07-02", "2015-07-06"),
    evaluation_period=("2015-07-07", "2015-07-08"),
)
df_empirical = scorer.empirical_probability_.to_frame()

scorer.optimize()
df_optimized = scorer.optimized_probability_.to_frame()
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
