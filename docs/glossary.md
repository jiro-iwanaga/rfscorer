# ユビキタス言語定義

本プロジェクトで使用するドメイン用語の定義。コード・ドキュメント・会話で一貫して使用する。

## データ

| 用語 | 英語 / 記号 | 定義 |
|------|------------|------|
| 閲覧履歴 | interaction history | ユーザーが商品を閲覧した記録。コンストラクタにDataFrameとして渡す。カラム名は引数で指定し、内部では `user`・`item`・`datetime` に正規化される |
| ユーザー | user | 閲覧履歴の主体。`user` カラムで識別する |
| 商品 | item | 閲覧対象。`item` カラムで識別する |
| 観測期間 | observation period / `observation_period` | 最新度・頻度を算出するために使用する期間。`fit()` に開始日・終了日の tuple で渡す |
| 評価期間 | evaluation period / `evaluation_period` | 再閲覧の有無を観測するために使用する期間。観測期間の直後に設定する。`fit()` に開始日・終了日の tuple で渡す |

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
| `RecencyFrequencyScorer` | RF スコアリングの主クラス。コンストラクタで閲覧履歴 DataFrame とカラム名を受け取る |
| `fit(observation_period, evaluation_period, recency_limit=None, frequency_limit=None)` | 観測期間・評価期間を tuple で受け取り、経験的再閲覧確率を推定するメソッド。`recency_limit`・`frequency_limit` は省略時に累積再閲覧数から自動決定 |
| `predict(r, f, kind='empirical')` | 指定した最新度 `r`・頻度 `f` の再閲覧確率を返すメソッド。`fit()` 後に利用可能 |
| `transform(df, target_date, kind='empirical', ...)` | 入力 DataFrame の各 user×item ペアに最新度・頻度・再閲覧確率・順位を付与して返すメソッド。`fit()` 後に利用可能。`kind='optimized'` を指定すると `optimize()` の結果を使用 |
| `evaluate(df_rec, UIrevisit, order=1, ...)` | 推薦結果と正解データを比較し precision・recall・f1 等の評価指標を返すメソッド |
| `plot_probability_surface(kind='empirical', path=None)` | 再閲覧確率を3次元ワイヤーフレームで可視化し PNG ファイルに保存するメソッド。`fit()` 後（`kind='optimized'` の場合は `optimize()` 後）に利用可能 |
| `optimize()` | `fit()` の結果を用いて、RF 制約付きの最適化再閲覧確率を推定するメソッド |
| `show()` | `fit()` 後の集計情報（レコード数・cv 数・期間・上限値）を標準出力に表示するデバッグ用メソッド |
| `interaction_log` | コンストラクタで正規化した閲覧履歴。カラムは `user`・`item`・`datetime` |
| `R` | `fit()` 後に参照できる最新度のリスト |
| `F` | `fit()` 後に参照できる頻度のリスト |
| `empirical_probability_` | `fit()` 後に参照できる経験的再閲覧確率。`pd.DataFrame`（カラム: `recency`, `frequency`, `N`, `cv`, `probability`） |
| `empirical_probability_table_` | `fit()` 後に参照できる経験的再閲覧確率（横持ち）。`pd.DataFrame`（インデックス: `recency`、カラム: `frequency`） |
| `empirical_probability_dict_` | `fit()` 後に参照できる経験的再閲覧確率。`dict`（キー: `(r, f)`） |
| `optimized_probability_` | `optimize()` 後に参照できる最適化再閲覧確率。`pd.DataFrame`（カラム: `recency`, `frequency`, `probability`） |
| `optimized_probability_table_` | `optimize()` 後に参照できる最適化再閲覧確率（横持ち）。`pd.DataFrame`（インデックス: `recency`、カラム: `frequency`） |
| `optimized_probability_dict_` | `optimize()` 後に参照できる最適化再閲覧確率。`dict`（キー: `(r, f)`） |
