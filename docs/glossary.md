# 用語集

本プロジェクトで使用するドメイン用語の定義。コード・ドキュメントで一貫して使用する。

## プロジェクト

| 用語 | 英語 / 記号 | 定義 |
|------|------------|------|
| RFscorer | RFscorer | プロジェクトの総称。PyPI で公開される Python パッケージの公式名称 |
| rfscorer | rfscorer | Python パッケージの実装名。`pip install rfscorer` でインストール時、および `from rfscorer import ...` でインポート時に使用する |

## データ

| 用語 | 英語 / 記号 | 定義 |
|------|------------|------|
| 行動履歴 | behavior history | ユーザーが商品を閲覧した記録。`fit()` に DataFrame として渡す。カラム名はコンストラクタ引数で指定し、内部で `user`・`item`・`datetime` に正規化される |
| ユーザー | user | 行動履歴の主体。`user` カラムで識別する |
| 商品 | item | 閲覧対象。`item` カラムで識別する |
| 観測期間 | observation period | 最新度・頻度を算出するために使用する期間。`fit()` に `df_obs` として渡す。`split_by_date()` を使えば単一の DataFrame から `target_date` を基準に自動分割できる |
| 正解期間 | ground truth period | 対象イベント（閲覧・購買・CV など）の発生を観測するために使用する期間。観測期間の直後に設定する。`fit()` に `df_gt` として渡す |
| 観測データ | observation data | 観測期間に該当する行動履歴 DataFrame。`fit(df_obs, ...)` に `df_obs` として渡す。最新度・頻度を計算する基となる閲覧履歴 |
| 正解データ | ground truth data | 正解期間に該当するイベント履歴 DataFrame。`fit(..., df_gt)` に `df_gt` として渡す。対象イベント（再閲覧・購買・CV など）の発生を記録 |

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
| 1次元最適化モデル | 1D optimization model | `optimize(kind='mr')` または `optimize(kind='mf')` で構築する1次元の最適化モデル。`mr` は最新度のみを変数とするモデル（最新度別経験的確率 $p_r$ を目標）、`mf` は頻度のみを変数とするモデル（頻度別経験的確率 $p_f$ を目標）。結果は1次元 dict として保存される |
| 2次元最適化モデル | 2D optimization model | `optimize(kind='mono'/'mrc'/'mfc'/'mcc')` で構築する2次元の最適化モデル。最新度と頻度の全ペア $(r, f) \in R \times F$ を変数とするモデル（経験的商品選択確率 $p_{r,f}$ を目標）。結果は `{kind}_probability_dict_`（dict）に格納される |

> **表記ルール — "monotonic" と "monotonicity"**
>
> 本プロジェクトでは品詞の違いにより両語を使い分ける。
>
> - **`monotonic`**（形容詞）: `optimize(kind=...)` の引数値など、コード中のモード名・エイリアス（識別子）として使用する。例: `"monotonic"`, `"monotonic_recency"`
> - **`monotonicity`**（名詞）: docstring・コメント・テストコードで単調性という数学的性質を説明する際に使用する。例: `monotonicity constraints`, `weak monotonicity`, `test_recency_monotonicity`
>
> 識別子には `monotonic`、説明文には `monotonicity` を使用する。この共存は表記ゆれではなく意図的な使い分けである。
