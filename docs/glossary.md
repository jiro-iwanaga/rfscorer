# ユビキタス言語定義

本プロジェクトで使用するドメイン用語の定義。コード・ドキュメントで一貫して使用する。

## プロジェクト

| 用語 | 英語 / 記号 | 定義 |
|------|------------|------|
| RFscorer | RFscorer | プロジェクトの総称。PyPI で公開される Python パッケージの公式名称 |
| rfscorer | rfscorer | Python パッケージの実装名。`pip install rfscorer` でインストール時、および `from rfscorer import ...` でインポート時に使用する |

## データ

| 用語 | 英語 / 記号 | 定義 |
|------|------------|------|
| 閲覧ログ | interaction log | ユーザーが商品を閲覧した記録。`fit()` に DataFrame として渡す。カラム名はコンストラクタ引数で指定し、内部で `user`・`item`・`datetime` に正規化される |
| ユーザー | user | 閲覧ログの主体。`user` カラムで識別する |
| 商品 | item | 閲覧対象。`item` カラムで識別する |
| 観測期間 | observation period | 最新度・頻度を算出するために使用する期間。`fit()` に `df_obs` として渡す。`split_by_date()` を使えば単一の DataFrame から `target_date` を基準に自動分割できる |
| 正解期間 | ground truth period | 対象イベント（閲覧・購買・CV など）の発生を観測するために使用する期間。観測期間の直後に設定する。`fit()` に `df_eval` として渡す |

## アルゴリズム

| 用語 | 英語 / 記号 | 定義 |
|------|------------|------|
| RF | Recency-Frequency | 最新度と頻度の2つの行動シグナルを指す。Random Forest ではない |
| RF スコアリング | RF scoring | 最新度（Recency）と頻度（Frequency）を行動シグナルとして用い、商品選択確率を推定する解釈可能な推薦スコアリング手法。本パッケージが提供する中核機能 |
| 最新度 | recency / $r$ | 観測期間における、ユーザーが最後に閲覧した時点からの経過時間を整数で表した値。1 が最も直近。$r \in R$（$R$ は観測されたすべての最新度の集合） |
| 頻度 | frequency / $f$ | 観測期間における、ユーザーによる商品の閲覧回数。$f \in F$（$F$ は観測されたすべての頻度の集合） |
| 商品選択確率 | product-choice probability | 観測期間で最新度 $r$・頻度 $f$ であった商品が、正解期間に対象イベント（再閲覧・購買・CV など）を発生させる確率 |
| 経験的商品選択確率 | empirical product-choice probability / $p_{r,f}$ | $p_{r,f} = n_{r,f} / N_{r,f}$。観測データから直接算出した商品選択確率 |
| 最適化商品選択確率 | optimized product-choice probability / $x_{r,f}$ | RF 制約（単調性制約）と最小二乗誤差を目的関数とする凸2次計画問題を解いて得られる商品選択確率 |
| $n_{r,f}$ | — | 観測期間で最新度 $r$・頻度 $f$ であった商品が、正解期間に対象イベントを発生させた回数の合計 |
| $N_{r,f}$ | — | 観測期間で最新度 $r$・頻度 $f$ であった (user, item) ペアの数（=サンプル数） |
| RF 制約 | RF constraints | 最適化商品選択確率に課す単調性制約の総称。Recency 制約と Frequency 制約からなる |
| Recency 制約 | recency constraint | $x_{r,f} \geq x_{r+1,f} + \varepsilon\ \ \ (r, r+1 \in R,\ f \in F)$。最新度が小さい（より直近に閲覧した）商品ほど商品選択確率が高い |
| Frequency 制約 | frequency constraint | $x_{r,f} + \varepsilon \leq x_{r,f+1}\ \ \ (r \in R,\ f, f+1 \in F)$。頻度が高い商品ほど商品選択確率が高い |
| 広義単調性 | weak monotonicity | $\varepsilon = 0$ のときの単調性制約。隣接する確率値が同値になることを許す（$\geq$ または $\leq$） |
| 狭義単調性 | strict monotonicity | $\varepsilon > 0$ のときの単調性制約。隣接する最新度・頻度の確率値が必ず $\varepsilon$ 以上離れることを保証する |
| $\varepsilon$（eps） | eps | `optimize(eps=ε)` で指定する単調性制約の最小ギャップ。デフォルト `0.0`（広義単調性）。2次元モデル（mono/mrc/mfc/mcc）の上限は $\min(\max(p_{r,f}) / (\lvert R\rvert - 1),\ \max(p_{r,f}) / (\lvert F\rvert - 1))$、`mr` の上限は $\max(p_r) / (\lvert R\rvert - 1)$、`mf` の上限は $\max(p_f) / (\lvert F\rvert - 1)$ で自動計算される |
| 1次元最適化モデル | 1D optimization model | `optimize(kind='mr')` または `optimize(kind='mf')` で構築する1次元の最適化モデル。`mr` は最新度のみを変数とするモデル（$R2Prob$ を目標）、`mf` は頻度のみを変数とするモデル（$F2Prob$ を目標）。結果は1次元 dict として保存される |
| 2次元最適化モデル | 2D optimization model | `optimize(kind='mono'/'mrc'/'mfc'/'mcc')` で構築する2次元の最適化モデル。最新度と頻度の全ペア $(r, f) \in R \times F$ を変数とするモデル（$RF2Prob$ を目標）。結果は `RF2X`（dict）に格納される |

## API

| 用語 | 定義 |
|------|------|
| `RecencyFrequencyScorer` | RF スコアリングの主クラス。コンストラクタでカラム名・`unit` を受け取る |
| `unit` | 最新度の粒度を指定する正の整数（デフォルト `1`）。最新度は `(ref - 値) // unit + 1` で算出される。`unit=1` で日単位、`unit=7` で週単位、`unit=30` で月単位（近似）になる |
| `fit(df_obs, df_eval, ref=None, recency_limit=None, frequency_limit=None)` | 観測ログ `df_obs` と正解ログ `df_eval` を直接受け取り、経験的商品選択確率を推定するメソッド。scikit-learn スタイルの主要 fit メソッド。`ref` は最新度計算の基準値（日付または整数。`None` の場合は `df_obs` の最大値を使用） |
| `split_by_date(df, target_date, observation_days=28, evaluation_days=7, time_col='datetime')` | 単一の閲覧ログ DataFrame を `target_date` を基準に観測ログと正解ログに分割するユーティリティ関数。`from rfscorer import split_by_date` でインポート可能。戻り値は `(df_obs, df_eval)` のタプル。`target_date` は日付・整数いずれも可 |
| `predict(r, f, kind='emp')` | 指定した最新度 `r`・頻度 `f` の商品選択確率を返すメソッド。`r` は1が最も直近（数値が大きいほど古い）、`f` は観測期間の閲覧回数。1次元モデルでは片方の引数のみ参照される（`kind='mr'/'er'` は `r` のみ、`kind='mf'/'ef'` は `f` のみ）。`r` が `recency_limit` を超える場合・`f` が `frequency_limit` を超える場合は上限にクランプされる。`fit()` 後に利用可能 |
| `transform(df, ref=None, kind='emp', ...)` | 入力 DataFrame の各 user×item ペアに最新度・頻度・商品選択確率・順位を付与して返すメソッド。`ref` は最新度計算の基準値（`None` の場合は `df` の最大値を使用）。`recency_limit`・`frequency_limit` を超える値は確率参照時に上限にクランプされる（出力カラム `recency`・`frequency` は元値を保持）。`user_col`・`item_col`・`time_col` は省略すると `__init__` の設定値を使用する。`fit()` 後に利用可能 |
| `evaluate(df_rec, df_eval, order=1, ...)` | 推薦結果と正解期間のイベント履歴 `df_eval` を比較し評価指標を返すメソッド。戻り値カラムは `order`, `n_recommended`, `n_hit`, `precision`, `recall`, `f1`, `recall_norm`（df_rec 内で達成可能な最大ヒット数で正規化した recall）, `f1_norm`（recall_norm を用いた f1）。`user_col`・`item_col` は省略すると `__init__` の設定値を使用する |
| `plot_probability_surface(kind='emp', title=None, figsize=(6, 5), fontsize=12, recency_label='recency', frequency_label='frequency', probability_label='probability')` | 商品選択確率を3次元ワイヤーフレームで可視化し `matplotlib.figure.Figure` を返すメソッド。軸ラベル・タイトル・図サイズ・フォントサイズを指定可能。日本語ラベルには `rfscorer[ja]` が必要。`fit()` 後（`kind='mono'/'mrc'/'mfc'/'mcc'` の場合は `optimize()` 後）に利用可能。`kind='mr'`・`'mf'`・`'er'`・`'ef'` は1次元モデルのため `ValueError` を送出する（折れ線表示には `plot_marginal_probability()` を使用する） |
| `plot_marginal_probability(axis='recency', kind='emp', title=None, figsize=(5, 4), fontsize=12, recency_label='recency', frequency_label='frequency', probability_label='probability')` | 最新度または頻度の一方向の商品選択確率を折れ線グラフで可視化し `matplotlib.figure.Figure` を返すメソッド。`kind='emp'/'er'/'ef'` で1次元経験的確率のみ、`kind='mr'/'mf'` で1次元最適化確率のみ、`kind='all'` で両者を重ねて表示する。単調性確認や最適化前後の比較に使用する。日本語ラベルには `rfscorer[ja]` が必要。`fit()` 後に利用可能（`kind='mr'/'mf'/'all'` の場合は `optimize()` 後も必要） |
| `optimize(kind='mono', eps=0.0)` | `fit()` の結果を用いて、RF 制約付きの最適化商品選択確率を推定するメソッド。`kind='mono'`（単調性のみ）・`'mrc'`（単調性 + Recency 凸性）・`'mfc'`（単調性 + Frequency 凹性）・`'mcc'`（単調性 + 両凹凸性）・`'mr'`（1次元 Recency）・`'mf'`（1次元 Frequency）を指定する（長名エイリアスも使用可）。`eps > 0` で狭義単調性を適用する |
| `show()` | `fit()` 後の集計情報（レコード数・cv 数・期間・上限値）を標準出力に表示するデバッグ用メソッド |
| `R` | `fit()` 後に参照できる最新度のリスト |
| `F` | `fit()` 後に参照できる頻度のリスト |
| `recency_limit` | `fit()` 後に参照できる最新度の上限値。これを超える最新度は `recency_limit` にクランプされてスコアリングされる。`None` の場合は累積対象イベント発生数の 95% をカバーする最新度に自動設定される |
| `frequency_limit` | `fit()` 後に参照できる頻度の上限値。これを超える頻度は `frequency_limit` にクランプされてスコアリングされる。`None` の場合は累積対象イベント発生数の 95% をカバーする頻度に自動設定される |
| `emp_probability_` | `fit()` 後に参照できる経験的商品選択確率。`pd.DataFrame`（カラム: `recency`, `frequency`, `N`, `cv`, `probability`） |
| `emp_probability_table_` | `fit()` 後に参照できる経験的商品選択確率（横持ち）。`pd.DataFrame`（インデックス: `recency`、カラム: `frequency`） |
| `emp_probability_dict_` | `fit()` 後に参照できる経験的商品選択確率。`dict`（キー: `(r, f)`） |
| `er_probability_` | `fit()` 後に参照できる1次元経験的商品選択確率（最新度のみ）。`pd.DataFrame`（カラム: `recency`, `probability`） |
| `er_probability_dict_` | `fit()` 後に参照できる1次元経験的商品選択確率。`dict`（キー: `r`（int）） |
| `ef_probability_` | `fit()` 後に参照できる1次元経験的商品選択確率（頻度のみ）。`pd.DataFrame`（カラム: `frequency`, `probability`） |
| `ef_probability_dict_` | `fit()` 後に参照できる1次元経験的商品選択確率。`dict`（キー: `f`（int）） |
| `mr_probability_` | `optimize(kind='mr')` 後に参照できる1次元最適化商品選択確率（最新度のみ）。`pd.DataFrame`（カラム: `recency`, `probability`） |
| `mr_probability_dict_` | `optimize(kind='mr')` 後に参照できる1次元最適化商品選択確率。`dict`（キー: `r`（int）） |
| `mf_probability_` | `optimize(kind='mf')` 後に参照できる1次元最適化商品選択確率（頻度のみ）。`pd.DataFrame`（カラム: `frequency`, `probability`） |
| `mf_probability_dict_` | `optimize(kind='mf')` 後に参照できる1次元最適化商品選択確率。`dict`（キー: `f`（int）） |
| `mono_probability_` | `optimize(kind='mono')` 後に参照できる最適化商品選択確率。`pd.DataFrame`（カラム: `recency`, `frequency`, `probability`） |
| `mono_probability_table_` | `optimize(kind='mono')` 後に参照できる最適化商品選択確率（横持ち）。`pd.DataFrame`（インデックス: `recency`、カラム: `frequency`） |
| `mono_probability_dict_` | `optimize(kind='mono')` 後に参照できる最適化商品選択確率。`dict`（キー: `(r, f)`） |
| `mrc_probability_` | `optimize(kind='mrc')` 後に参照できる最適化商品選択確率。`pd.DataFrame`（カラム: `recency`, `frequency`, `probability`） |
| `mrc_probability_table_` | `optimize(kind='mrc')` 後に参照できる最適化商品選択確率（横持ち）。`pd.DataFrame`（インデックス: `recency`、カラム: `frequency`） |
| `mrc_probability_dict_` | `optimize(kind='mrc')` 後に参照できる最適化商品選択確率。`dict`（キー: `(r, f)`） |
| `mfc_probability_` | `optimize(kind='mfc')` 後に参照できる最適化商品選択確率。`pd.DataFrame`（カラム: `recency`, `frequency`, `probability`） |
| `mfc_probability_table_` | `optimize(kind='mfc')` 後に参照できる最適化商品選択確率（横持ち）。`pd.DataFrame`（インデックス: `recency`、カラム: `frequency`） |
| `mfc_probability_dict_` | `optimize(kind='mfc')` 後に参照できる最適化商品選択確率。`dict`（キー: `(r, f)`） |
| `mcc_probability_` | `optimize(kind='mcc')` 後に参照できる最適化商品選択確率。`pd.DataFrame`（カラム: `recency`, `frequency`, `probability`） |
| `mcc_probability_table_` | `optimize(kind='mcc')` 後に参照できる最適化商品選択確率（横持ち）。`pd.DataFrame`（インデックス: `recency`、カラム: `frequency`） |
| `mcc_probability_dict_` | `optimize(kind='mcc')` 後に参照できる最適化商品選択確率。`dict`（キー: `(r, f)`） |
