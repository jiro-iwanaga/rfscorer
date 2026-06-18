# 用語集

本プロジェクトで使用するドメイン用語の定義。コード・ドキュメントで一貫して使用する。

## プロジェクト

| 用語 | 英語 / 記号 | 定義 |
|------|------------|------|
| RFscorer | RFscorer | プロジェクトの総称。PyPI で公開される Python パッケージの公式名称 |
| rfscorer | rfscorer | Python パッケージの実装名。`pip install rfscorer` でインストール時、および `from rfscorer import ...` でインポート時に使用する |

## 基本概念

| 用語 | 英語 / 記号 | 定義 |
|------|------------|------|
| ユーザー | user | 行動の主体。`user` カラムで識別する |
| 商品 | item | 閲覧対象。`item` カラムで識別する |
| 対象イベント | target event | 推定対象のイベント。再閲覧・購買・CV など。`fit()` / `fit_rolling()` 時に `df_gt` で指定 |
| 行動履歴 | behavior history | ユーザーが商品を閲覧した記録の時系列データ。`fit()` に DataFrame として渡す |

## 期間とデータ分割

| 用語 | 英語 / 記号 | 定義 |
|------|------------|------|
| 観測期間 | observation period | 最新度・頻度を計算する期間。`split_by_date()` / `fit_rolling()` で `observation_days` パラメータで指定 |
| 正解期間 | ground truth period | 対象イベントを観測する期間。観測期間の直後に設定。`split_by_date()` / `fit_rolling()` で `gt_days` パラメータで指定 |
| 観測データ | observation data | 観測期間に該当する行動履歴 DataFrame。`fit()` / `fit_rolling()` に `df_obs` として渡す |
| 正解データ | ground truth data | 正解期間に該当するイベント履歴 DataFrame。`fit()` / `fit_rolling()` に `df_gt` として渡す |

## アルゴリズム

### コア概念

| 用語 | 英語 / 記号 | 定義 |
|------|------------|------|
| RF | Recency-Frequency | 最新度と頻度の2つの行動シグナル。Random Forest ではない |
| RF スコアリング | RF scoring | 最新度と頻度から商品選択確率を推定する解釈可能な推薦スコアリング手法 |
| ローリング集計 | rolling aggregation | 分割点（基準日）を1日ずつ過去にずらしながら複数基準日で $n_{r,f}$・$N_{r,f}$ を積み増す集計。サンプル増による経験的確率の安定化と基準日バイアスの平滑化を目的とする。`fit_rolling()` が実装 |
| 最新度 | recency / $r$ | 最後の閲覧からの経過時間を整数化したもの。1 が最も直近。$r \in R$ |
| 頻度 | frequency / $f$ | 観測期間における閲覧回数。$f \in F$ |
| 実効サンプルサイズ（延べ） | effective / pooled sample size | 経験的確率の分母となる延べ件数。`fit_rolling()` では重なるロールで物理行が複数回計数される。属性: `record_num*`・`total_cv*` |
| データセット規模（物理ユニーク） | physical / unique dataset size | 和集合区間で重複なく数えた実件数。論文記載用のデータ数。属性: `n_obs_rows_`・`n_gt_events_`・`n_users_`・`n_items_` |

### 商品選択確率

| 用語 | 英語 / 記号 | 定義 |
|------|------------|------|
| 商品選択確率 | product-choice probability | 観測期間の (r,f) が正解期間で対象イベントを発生させる確率 |
| 推薦スコア | recommendation score | 商品選択確率の別称。推薦の根拠となるスコア |
| 経験的商品選択確率 | empirical product-choice probability / $p_{r,f}$ | $p_{r,f} = n_{r,f} / N_{r,f}$。観測データから直接算出。ノイズを含む |
| 最適化商品選択確率 | optimized product-choice probability / $x_{r,f}$ | RF 制約を満たすよう最適化した商品選択確率 |
| $n_{r,f}$ | — | 観測期間で (r,f) であった商品が正解期間で対象イベントを発生させた回数の合計 |
| $N_{r,f}$ | — | 観測期間で (r,f) であった (user, item) ペアの数 |

### 最適化と制約

| 用語 | 英語 / 記号 | 定義 |
|------|------------|------|
| 凸2次計画問題 | convex quadratic programming | 目的関数が凸で制約が線形な最適化問題。本パッケージは cvxpy で求解 |
| RF 制約 | RF constraints | 最適化に課す単調性制約の総称。Recency 制約と Frequency 制約からなる |
| Recency 制約 | recency constraint | $x_{r,f} \geq x_{r+1,f} + \varepsilon$。より直近 ($r$ が小さい) の商品ほど確率が高い |
| Frequency 制約 | frequency constraint | $x_{r,f} + \varepsilon \leq x_{r,f+1}$。より高頻度 ($f$ が大きい) の商品ほど確率が高い |
| 広義単調性 | weak monotonicity | $\varepsilon = 0$ のときの単調性。隣接する確率値が同値になることを許す |
| 狭義単調性 | strict monotonicity | $\varepsilon > 0$ のときの単調性。隣接する値が必ず $\varepsilon$ 以上離れることを保証 |
| $\varepsilon$（eps） | eps | 単調性制約における隣接値の最小差。`optimize(eps=ε)` で指定。デフォルト `0.0` |

### モデルの種類

| 用語 | 英語 / 記号 | 定義 |
|------|------------|------|
| 1次元最適化モデル | 1D optimization model | 最新度または頻度のみを変数とする1次元最適化。`mr`（Recency）と `mf`（Frequency） |
| 2次元最適化モデル | 2D optimization model | 最新度と頻度の全ペア $(r, f)$ を変数とする2次元最適化。`mono`・`mrc`・`mfc`・`mcc` |

> **表記ルール — "monotonic" と "monotonicity"**
>
> 本プロジェクトでは品詞の違いにより両語を使い分ける。
>
> - **`monotonic`**（形容詞）: `optimize(kind=...)` の引数値など、コード中のモード名・エイリアス（識別子）として使用する。例: `"monotonic"`, `"monotonic_recency"`
> - **`monotonicity`**（名詞）: docstring・コメント・テストコードで単調性という数学的性質を説明する際に使用する。例: `monotonicity constraints`, `weak monotonicity`, `test_recency_monotonicity`
>
> 識別子には `monotonic`、説明文には `monotonicity` を使用する。この共存は表記ゆれではなく意図的な使い分けである。

## API（簡潔版）

主要メソッドの簡潔説明。詳細は機能仕様書（`functional-design.md`）を参照。

| 用語 | 説明 |
|------|------|
| `RecencyFrequencyScorer` | RF スコアリングの主クラス。コンストラクタで列名・粒度 `unit` を指定 |
| `fit(df_obs, df_gt, ref, recency_limit, frequency_limit)` | 観測データと正解データから経験的商品選択確率を推定する |
| `fit_rolling(df_obs, df_gt, observation_days, gt_days, roll_days, end_date)` | 基準日を1日ずつ過去にずらしながら複数基準日で集計を積み増し、経験的商品選択確率を推定する（ローリング集計） |
| `transform(df, ref, kind)` | 行動履歴 DataFrame に最新度・頻度・確率・順位を付与する |
| `optimize(kind, eps, verbose)` | RF 制約付き凸2次計画問題を解いて最適化確率を推定する |
| `predict(r, f, kind)` | 指定した最新度・頻度の商品選択確率を返す |
| `evaluate(df_rec, df_gt, order)` | 推薦結果と正解データを比較し precision・recall・F1 を返す |
| `show()` | `fit()` 後の状態をデータ統計・相関係数・確率テーブルで表示 |
| `export_probability_csv(kind, path)` | 商品選択確率テーブルを CSV (`probability_{kind}.csv`) に出力 |
| `plot_probability_surface(kind, ...)` | 商品選択確率を3次元ワイヤーフレームで可視化 |
| `plot_marginal_probability(kind, ...)` | 最新度または頻度の一方向の確率を折れ線グラフで可視化 |
| `save(path)` / `load(path)` | モデルの pickle 形式での保存・ロード |
| `save_zip(path)` / `load_zip(path)` | モデルを zip アーカイブ（pickle + CSV + PNG）で保存・ロード |
| `split_by_date(df, target_date, observation_days, gt_days)` | 観測データと正解データに自動分割するユーティリティ関数 |
