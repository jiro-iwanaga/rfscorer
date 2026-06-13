# プロダクト要求定義書

## 背景

EC サイトやコンテンツプラットフォームでは、ユーザーの閲覧履歴に基づいて商品推薦を行いたいというニーズがある。  一般的なブラックボックス型の推薦モデルは、スコアの根拠を説明しにくく、運用・デバッグが困難な場合がある。

本パッケージは、解釈可能な推薦スコアリング手法を提供する。Recency（最新度: ユーザーが商品と最後に接触してからの経過時間）と Frequency（頻度: ユーザーが商品と接触した回数）の2つの行動シグナルのみを用い、ユーザー × 商品ペアごとに商品選択確率を推定する。各ペアに付与される商品選択確率は選好スコアとして機能し、得られる行列は評価値行列（rating matrix）と類似した構造を持つため、協調フィルタリングや機械学習を用いた推薦モデルの入力（特徴量）としても活用できる。手法の理論的基盤は以下の学術論文に基づく。

> Jiro Iwanaga, Naoki Nishimura, Noriyoshi Sukegawa, and Yuichi Takano,  
> "Estimating product-choice probabilities from recency and frequency of page views,"  
> *Knowledge-Based Systems*, Volume 99, 2016, Pages 157–167.

また、本手法から得られたスコアを協調フィルタリングの入力として利用する応用は、以下の論文で実証されている。

> Jiro Iwanaga, Naoki Nishimura, Noriyoshi Sukegawa, and Yuichi Takano,  
> "Improving collaborative filtering recommendations by estimating user preferences from clickstream data,"  
> *Electronic Commerce Research and Applications*, Volume 37, Article 100877, 2019.

## 目的

RF（Recency-Frequency）スコアリング手法を Python パッケージとして PyPI に公開し、`pip install rfscorer` で誰でも利用できるようにする。

## ターゲットユーザー
- 実務家
  - Python とデータ分析に習熟したデータサイエンティスト・機械学習エンジニア
  - EC サイト・コンテンツプラットフォームの推薦システム担当者
  - 解釈可能な選好スコアを必要とするビジネスアナリスト
- 研究者
  - 推薦システム・情報検索
  - マーケティングサイエンス・消費者行動論
  - オペレーションズリサーチ・数理最適化
  - 認知心理学（記憶・単純接触効果）  
      
## 解決する課題

| 課題 | 説明 |
|------|------|
| 選好スコアの算出 | ユーザー × 商品の行動履歴から、各ユーザー × 商品ペアの選好スコアを推定する（∝対象イベント発生確率: 再閲覧・購買・CV など） |
| 解釈可能性の確保 | スコアの根拠を説明できる |
| ノイズの除去と自然な選好順序の獲得 | 観測データから直接得られる経験的商品選択確率は標本ノイズを含み、選好順序が不自然になる場合がある。最新度・頻度に対する単調性制約を課した最適化により、ノイズを抑えた滑らかな商品選択確率を再推定でき、自然な選好順序が得られる |
| 選好分布の傾きと形状の把握 | 最新度・頻度に対する選好の減衰・増加傾向（傾き）と凹凸性を定量的に把握できる |
| 後段モデルへの入力としての活用 | 推定した選好スコアを協調フィルタリングの評価値行列や機械学習の特徴量として直接利用できる |
| 実装コストの削減 | RF スコアリングを自前実装する手間を削減する |

## 機能要求

### 入力

- コンストラクタ
  - カラム名を `user_col`・`item_col`・`time_col` で指定する（デフォルト: `user`・`item`・`datetime`）
  - `time_col` には日付型（`datetime64`・文字列）・整数型いずれも指定可能
  - `unit`: 最新度の粒度を指定する正の整数（デフォルト: `1`）。`unit=7` で週単位、`unit=30` で月単位（近似）
- `fit()` に渡す引数（推奨）
  - `df_obs`: 観測期間の閲覧履歴 DataFrame。同一ユーザー × 商品の組み合わせが複数行存在することを想定（リピート閲覧）
  - `df_eval`: 評価期間のイベント履歴 DataFrame（閲覧・購買・CV など推定対象のイベント）。`df_obs` と同じカラム構成
  - `ref`: 最新度計算の基準値（日付または整数。省略時は `df_obs` の最大値を使用）
  - `recency_limit`: 最新度の上限値（省略時は累積 cv の 95% をカバーする値を自動設定）
  - `frequency_limit`: 頻度の上限値（省略時は累積 cv の 95% をカバーする値を自動設定）
- データ準備の補助
  - `from rfscorer import split_by_date` で利用可能
  - `split_by_date(df, target_date, observation_days=28, evaluation_days=7, time_col="datetime")` で単一の閲覧履歴 DataFrame を観測ログ・評価ログに分割可能（戻り値は `(df_obs, df_eval)` のタプル）
  - 任意の期間制御は標準 pandas フィルタ（例: `df[mask]`）で `df_obs` / `df_eval` を構築して `fit()` に渡す

### 機能

| 機能 | 説明 |
|------|------|
| 経験的商品選択確率の推定（`emp`） | 観測期間における最新度 $r$・頻度 $f$ 別に、評価期間での商品選択確率を直接推定する |
| 1次元経験的商品選択確率の推定（`er` / `ef`） | `fit()` 時に自動計算。最新度方向（`er`）・頻度方向（`ef`）の1次元経験的商品選択確率を dict / DataFrame として保持する |
| 1次元最適化商品選択確率の推定（`mr` / `mf`） | 1次元経験的商品選択確率を目標とした1次元の凸2次計画問題を解く。`mr` は Recency 単調性 + 凸性、`mf` は Frequency 単調性 + 凹性を制約として課す。結果は1次元 dict に格納され、2次元グリッドへのブロードキャストは行わない |
| 2次元最適化商品選択確率の推定（`mono` / `mrc` / `mfc` / `mcc`） | RF 制約と最小二乗誤差を目的関数にもつ凸2次計画問題を解いて推定する。制約の組み合わせにより `mono`（単調性のみ）・`mrc`（+ Recency 凸性）・`mfc`（+ Frequency 凹性）・`mcc`（+ 両凹凸性）の4モデルを提供する |
| 狭義単調性（`eps` パラメータ） | `optimize(eps=ε)` に正の値を指定すると、隣接する最新度・頻度の確率値が必ず $\varepsilon$ 以上離れる狭義単調性制約を付与する。デフォルト（`eps=0.0`）は従来の広義単調性と等価 |
| 推薦精度の評価（`evaluate`） | 推薦結果と評価期間のイベント履歴を比較し、各順位カットオフでの precision・recall・f1 を返す |
| 推薦スコアリング（`transform`） | 観測期間の閲覧履歴 DataFrame の各 user×item ペアに最新度・頻度・商品選択確率・推薦順位を付与して返す |
| 個別確率取得（`predict`） | 指定した最新度 $r$・頻度 $f$ に対応する商品選択確率を1件返す |
| 確率面の3次元可視化（`plot_probability_surface`） | 商品選択確率を recency × frequency の3次元ワイヤーフレームで可視化し `Figure` を返す |
| 1次元確率の折れ線可視化（`plot_marginal_probability`） | 最新度または頻度の一方向の商品選択確率を折れ線グラフで可視化し `Figure` を返す。経験的確率と最適化確率を重ねて表示可能 |
| 確率テーブルの CSV 出力（`export_probability_csv`） | 任意のモデルの商品選択確率テーブルを CSV ファイルに書き出す。`kind="all"` で全モデルを1ファイルにまとめて出力 |

### 出力

| 属性 | 説明 |
|------|------|
| `emp_probability_` | 経験的商品選択確率（`emp`）。`pd.DataFrame`（カラム: `recency`, `frequency`, `N`, `cv`, `probability`）。`fit()` 後にアクセス可能 |
| `er_probability_` | 1次元経験的商品選択確率（`er`）。`pd.DataFrame`（カラム: `recency`, `probability`）。`fit()` 後にアクセス可能 |
| `ef_probability_` | 1次元経験的商品選択確率（`ef`）。`pd.DataFrame`（カラム: `frequency`, `probability`）。`fit()` 後にアクセス可能 |
| `mr_probability_` | mr モデル最適化商品選択確率。`pd.DataFrame`（カラム: `recency`, `probability`）。`optimize(kind='mr')` 後にアクセス可能 |
| `mf_probability_` | mf モデル最適化商品選択確率。`pd.DataFrame`（カラム: `frequency`, `probability`）。`optimize(kind='mf')` 後にアクセス可能 |
| `mono_probability_` | mono モデル最適化商品選択確率。`pd.DataFrame`（カラム: `recency`, `frequency`, `probability`）。`optimize(kind='mono')` 後にアクセス可能 |
| `mrc_probability_` | mrc モデル最適化商品選択確率。`pd.DataFrame`（カラム: `recency`, `frequency`, `probability`）。`optimize(kind='mrc')` 後にアクセス可能 |
| `mfc_probability_` | mfc モデル最適化商品選択確率。`pd.DataFrame`（カラム: `recency`, `frequency`, `probability`）。`optimize(kind='mfc')` 後にアクセス可能 |
| `mcc_probability_` | mcc モデル最適化商品選択確率。`pd.DataFrame`（カラム: `recency`, `frequency`, `probability`）。`optimize(kind='mcc')` 後にアクセス可能 |

### kind エイリアス

短い正式名と長いエイリアス名の両方が使用可能。

| 正式名 | エイリアス |
|--------|-----------|
| `emp` | `empirical` |
| `er` | `empirical_recency` |
| `ef` | `empirical_frequency` |
| `mono` | `monotonic` |
| `mr` | `monotonic_recency` |
| `mf` | `monotonic_frequency` |
| `mrc` | `monotonic_recency_convex` |
| `mfc` | `monotonic_frequency_concave` |
| `mcc` | `monotonic_convex_concave` |

### API

```python
import pandas as pd
from rfscorer import RecencyFrequencyScorer, split_by_date

# サンプルデータ: ohmsha/PyOptBook (MIT License)
url = "https://raw.githubusercontent.com/ohmsha/PyOptBook/main/7.recommendation/access_log.csv"
df = pd.read_csv(url)

scorer = RecencyFrequencyScorer(user_col="user_id", item_col="item_id", time_col="date")

# split_by_date() で観測ログと評価ログを自動分割（推奨）
target_date = "2015-07-06"
df_obs, df_eval = split_by_date(df, target_date=target_date)
scorer.fit(df_obs, df_eval)
df_emp = scorer.emp_probability_  # 経験的商品選択確率（2次元）
df_er  = scorer.er_probability_         # 1次元経験的商品選択確率・Recency 方向（fit 時に自動計算）
df_ef  = scorer.ef_probability_         # 1次元経験的商品選択確率・Frequency 方向（fit 時に自動計算）

df_rec = scorer.transform(df_obs, ref=target_date)      # 推薦スコアリング
scorer.evaluate(df_rec, df_eval)                        # 評価

scorer.optimize(kind="mr")             # 1次元: Recency 単調性 + 凸性（広義単調性）
df_mr = scorer.mr_probability_

scorer.optimize(kind="mf")             # 1次元: Frequency 単調性 + 凹性（広義単調性）
df_mf = scorer.mf_probability_

scorer.optimize(kind="mono")           # 2次元: 単調性のみ（広義単調性）
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
| API 設計 | scikit-learn スタイル（`fit` / `transform` / `optimize`）に準拠し、既存ワークフローに組み込みやすくする |
| 解釈可能性 | スコアの算出根拠を説明できること。ブラックボックス化しない |
| ドキュメント | README および docstring で API・使用例を説明する |
| ライセンス | MIT License |

## 制約

- Python パッケージとして開発し、`uv` で依存関係を管理する
- ライブラリコードは `src/rfscorer/` に配置する
- 公開クラスは `from rfscorer import RecencyFrequencyScorer` でインポートできること
