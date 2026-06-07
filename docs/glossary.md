# ユビキタス言語定義

本プロジェクトで使用するドメイン用語の定義。コード・ドキュメント・会話で一貫して使用する。

## データ

| 用語 | 英語 / 記号 | 定義 |
|------|------------|------|
| 閲覧履歴 | interaction history | ユーザーが商品を閲覧した記録。`fit()` または `fit_period()` に DataFrame として渡す。カラム名はコンストラクタ引数で指定し、内部で `user`・`item`・`datetime` に正規化される |
| ユーザー | user | 閲覧履歴の主体。`user` カラムで識別する |
| 商品 | item | 閲覧対象。`item` カラムで識別する |
| 観測期間 | observation period / `observation_period` | 最新度・頻度を算出するために使用する期間。`fit()` では `target_date` から自動導出される。`fit_period()` に開始日・終了日の tuple で明示指定することもできる |
| 評価期間 | evaluation period / `evaluation_period` | 再閲覧の有無を観測するために使用する期間。観測期間の直後に設定する。`fit()` では `target_date` の翌日から自動導出される。`fit_period()` に開始日・終了日の tuple で明示指定することもできる |

## アルゴリズム

| 用語 | 英語 / 記号 | 定義 |
|------|------------|------|
| RF | Recency-Frequency | 最新度と頻度の2つの行動シグナルを指す。Random Forest ではない |
| 最新度 | recency / $r$ | 観測期間における、ユーザーが最後に閲覧した日付の新しさをランクで表した値。1 が最も直近。$r \in R$（$R$ は観測されたすべての最新度ランクの集合） |
| 頻度 | frequency / $f$ | 観測期間における、ユーザーによる商品の閲覧回数。$f \in F$（$F$ は観測されたすべての頻度の集合） |
| 再閲覧確率 | revisit probability | 観測期間で最新度 $r$・頻度 $f$ であった商品が、評価期間に再閲覧される確率 |
| 経験的再閲覧確率 | empirical revisit probability / $p_{r,f}$ | $p_{r,f} = n_{r,f} / N_{r,f}$。観測データから直接算出した再閲覧確率 |
| 最適化再閲覧確率 | optimized revisit probability / $x_{r,f}$ | RF 制約（単調性制約）と最小二乗誤差を目的関数とする凸2次計画問題を解いて得られる再閲覧確率 |
| $n_{r,f}$ | — | 観測期間で最新度 $r$・頻度 $f$ であった商品が、評価期間に再閲覧された回数の合計 |
| $N_{r,f}$ | — | 観測期間で最新度 $r$・頻度 $f$ であった商品の閲覧回数の合計 |
| RF 制約 | RF constraints | 最適化再閲覧確率に課す単調性制約の総称。Recency 制約と Frequency 制約からなる |
| Recency 制約 | recency constraint | $r < r' \Rightarrow x_{r,f} \geq x_{r',f}$。最新度ランクが小さい（より直近に閲覧した）商品ほど再閲覧確率が高い |
| Frequency 制約 | frequency constraint | $f < f' \Rightarrow x_{r,f} \leq x_{r',f}$。頻度が高い商品ほど再閲覧確率が高い |

## API

| 用語 | 定義 |
|------|------|
| `RecencyFrequencyScorer` | RF スコアリングの主クラス。コンストラクタでカラム名を受け取る |
| `fit(df, target_date, observation_days=28, evaluation_days=7, recency_limit=None, frequency_limit=None)` | 閲覧履歴 DataFrame と基準日 `target_date` を受け取り、観測・評価ウィンドウを自動導出して経験的再閲覧確率を推定するメソッド。`observation_days`・`evaluation_days` でウィンドウ幅を調整できる（`None` でデータ全範囲） |
| `fit_period(df, observation_period, evaluation_period, recency_limit=None, frequency_limit=None)` | 観測期間・評価期間を tuple で明示指定して経験的再閲覧確率を推定するメソッド。`fit()` より細かい期間制御が必要な場合に使用する |
| `predict(r, f, kind='empirical')` | 指定した最新度 `r`・頻度 `f` の再閲覧確率を返すメソッド。`r` は1が最も直近（数値が大きいほど古い）、`f` は観測期間の閲覧回数。`fit()` または `fit_period()` 後に利用可能 |
| `transform(df, target_date, kind='empirical', ...)` | 入力 DataFrame の各 user×item ペアに最新度・頻度・再閲覧確率・順位を付与して返すメソッド。`user_col`・`item_col`・`datetime_col` は省略すると `__init__` の設定値を使用する。`fit()` または `fit_period()` 後に利用可能 |
| `evaluate(df_rec, UIrevisit, order=1, ...)` | 推薦結果と正解データを比較し precision・recall・f1 等の評価指標を返すメソッド。`user_col`・`item_col` は省略すると `__init__` の設定値を使用する |
| `plot_probability_surface(kind='empirical')` | 再閲覧確率を3次元ワイヤーフレームで可視化し `matplotlib.figure.Figure` を返すメソッド。Jupyter Lab / Colab ではインライン描画される。ファイル保存は `fig.savefig()` で行う。`fit()` または `fit_period()` 後（`kind='mono'` または `'mcc'` の場合は `optimize()` 後）に利用可能 |
| `plot_marginal_probability(axis='recency')` | 最新度または頻度の一方向に集約した周辺的経験的再閲覧確率を折れ線グラフで可視化し `matplotlib.figure.Figure` を返すメソッド。`optimize()` 前の単調性確認に使用する。`fit()` または `fit_period()` 後に利用可能 |
| `optimize(kind='mono')` | `fit()` または `fit_period()` の結果を用いて、RF 制約付きの最適化再閲覧確率を推定するメソッド。`kind='mono'`（単調性制約のみ）または `'mcc'`（単調性 + 凹凸性制約）を指定する |
| `show()` | `fit()` または `fit_period()` 後の集計情報（レコード数・cv 数・期間・上限値）を標準出力に表示するデバッグ用メソッド |
| `R` | `fit()` または `fit_period()` 後に参照できる最新度のリスト |
| `F` | `fit()` または `fit_period()` 後に参照できる頻度のリスト |
| `empirical_probability_` | `fit()` または `fit_period()` 後に参照できる経験的再閲覧確率。`pd.DataFrame`（カラム: `recency`, `frequency`, `N`, `cv`, `probability`） |
| `empirical_probability_table_` | `fit()` または `fit_period()` 後に参照できる経験的再閲覧確率（横持ち）。`pd.DataFrame`（インデックス: `recency`、カラム: `frequency`） |
| `empirical_probability_dict_` | `fit()` または `fit_period()` 後に参照できる経験的再閲覧確率。`dict`（キー: `(r, f)`） |
| `mono_probability_` | `optimize(kind='mono')` 後に参照できる最適化再閲覧確率。`pd.DataFrame`（カラム: `recency`, `frequency`, `probability`） |
| `mono_probability_table_` | `optimize(kind='mono')` 後に参照できる最適化再閲覧確率（横持ち）。`pd.DataFrame`（インデックス: `recency`、カラム: `frequency`） |
| `mono_probability_dict_` | `optimize(kind='mono')` 後に参照できる最適化再閲覧確率。`dict`（キー: `(r, f)`） |
| `mcc_probability_` | `optimize(kind='mcc')` 後に参照できる最適化再閲覧確率。`pd.DataFrame`（カラム: `recency`, `frequency`, `probability`） |
| `mcc_probability_table_` | `optimize(kind='mcc')` 後に参照できる最適化再閲覧確率（横持ち）。`pd.DataFrame`（インデックス: `recency`、カラム: `frequency`） |
| `mcc_probability_dict_` | `optimize(kind='mcc')` 後に参照できる最適化再閲覧確率。`dict`（キー: `(r, f)`） |
