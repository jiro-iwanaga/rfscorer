# 機能設計書

## 概要

`rfscorer` は、ユーザー × 商品の閲覧履歴から商品の商品選択確率を推定する。  
推定には2段階のアプローチをとる。

1. **経験的商品選択確率の推定**: 観測期間に最新度 $r$、頻度 $f$ の商品が、評価期間で対象イベント（再閲覧・購買・CV など）を発生させる割合とする
2. **最適化商品選択確率の推定**: 経験的商品選択確率を用いて、RF 制約と最小二乗誤差を目的関数にもつ凸2次計画問題を解いて推定する

## 数理モデル

### 用語定義

| 記号 | 説明 |
|------|------|
| $U$ | ユーザーのリスト。$u$はユーザーを表す。 |
| $I$ | 商品のリスト。$i$は商品を表す。 |
| $R$ | 観測期間の最新度のリスト(1以上の連続自然数の集合)。$r$は最新度を表す。 |
| $F$ | 観測期間の頻度のリスト(1以上の連続自然数の集合)。$f$は頻度を表す。 |
| $n_{r,f}$ | 観測期間で最新度 $r$、頻度 $f$ の商品が評価期間で対象イベントを発生させる頻度合計 |
| $N_{r,f}$ | 観測期間で最新度 $r$、頻度 $f$ の商品の頻度合計 |
| $p_{r,f}$ | 観測期間で最新度 $r$、頻度 $f$ の商品の評価期間における経験的商品選択確率（2次元） |
| $p_r$ | 最新度 $r$ の1次元経験的商品選択確率（$f$ 方向に集約） |
| $p_f$ | 頻度 $f$ の1次元経験的商品選択確率（$r$ 方向に集約） |
| $x_{r,f}$ | 観測期間で最新度 $r$、頻度 $f$ の商品の評価期間における最適化商品選択確率 |

### 経験的商品選択確率の推定（`emp`）

「観測期間で最新度 $r$、頻度 $f$ の商品が評価期間で対象イベントを発生させる頻度合計」を
「観測期間で最新度 $r$、頻度 $f$ の商品の頻度合計」で割った値を経験的商品選択確率とする。

$$p_{r,f} := \frac{n_{r,f}}{N_{r,f}}\ \ \  (r\in R, f\in F)$$

### 1次元経験的商品選択確率の推定（`er` / `ef`）

$p_{r,f}$ を一方の次元で集約した1次元確率を、`fit()` 呼び出し時に自動計算する。結果は1次元 dict / DataFrame として保持され、2次元グリッドへのブロードキャストは行わない。

- **`er`**（Empirical Recency）: 最新度 $r$ の1次元確率。
$$x_r = p_r\ \ \ (r\in R)$$
- **`ef`**（Empirical Frequency）: 頻度 $f$ の1次元確率。
$$x_f = p_f\ \ \ (f\in F)$$

### 最適化商品選択確率の推定

経験的商品選択確率を基準として、RF 制約を満たす最適化商品選択確率を求める。

#### 2次元最適化モデル（`mono` / `mrc` / `mfc` / `mcc`）

経験的商品選択確率 $p_{r,f}$ を目標として、(r, f) グリッド全体で最適化する。

**共通制約（単調性）**
すべてのモデルに適用する。`eps=0.0`（デフォルト）の場合は広義単調性、`eps > 0` の場合は狭義単調性となり隣接値の差が $\varepsilon$ 以上になる。

- **Recency 制約**: 最近閲覧した商品ほど商品選択確率が高い。
$$x_{r,f} \geq x_{r+1,f} + \varepsilon\ \ \ (r, r+1 \in R,\ f \in F)$$
- **Frequency 制約**: 商品の閲覧が多いほど商品選択確率が高い。
$$x_{r,f} + \varepsilon \leq x_{r,f+1}\ \ \ (r \in R,\ f, f+1 \in F)$$

$\varepsilon = 0$ のとき広義単調性（$\geq$）、$\varepsilon > 0$ のとき狭義単調性。$\varepsilon$ の上限は $\max(p_{r,f}) / (\lvert R\rvert - 1)$ および $\max(p_{r,f}) / (\lvert F\rvert - 1)$ の小さい方の値とする。

**追加制約（凹凸性）**

| モデル | Recency 凸性 | Frequency 凹性 |
|--------|:-----------:|:-------------:|
| `mono` | — | — |
| `mrc`  | ✓ | — |
| `mfc`  | — | ✓ |
| `mcc`  | ✓ | ✓ |

- **Recency 凸性**（`mrc`・`mcc`）: 最近閲覧した商品ほど商品選択確率の落ち幅が大きい。
$$x_{r,f} - 2x_{r+1,f} + x_{r+2,f} \geq 0\ \ \ (r, r+1, r+2 \in R)$$
- **Frequency 凹性**（`mfc`・`mcc`）: 商品の閲覧が多いほど商品選択確率の上昇幅が小さい。
$$x_{r,f} - 2x_{r,f+1} + x_{r,f+2} \leq 0\ \ \ (f, f+1, f+2 \in F)$$

**目的関数（共通）**
$$\sum_{r\in R, f\in F} N_{r,f} \cdot(p_{r,f} - x_{r,f})^2$$

#### 1次元最適化モデル（`mr` / `mf`）

1次元経験的確率 $p_r$・$p_f$ を目標として1次元で最適化する。結果は1次元 dict に格納される。

- **`mr`**（Monotonic Recency）: $r$ 方向の単調性と凸性を同時に制約。
  - 変数: $x_r\ (r \in R)$
  - 単調性: $x_r \geq x_{r+1} + \varepsilon\ \ \ (r, r+1 \in R)$
  - 凸性: $x_r - 2x_{r+1} + x_{r+2} \geq 0$
  - 目的関数: $\sum_{r \in R} N_r \cdot (p_r - x_r)^2$
  - $\varepsilon$ の上限: $\max(p_r) / (\lvert R\rvert - 1)$

- **`mf`**（Monotonic Frequency）: $f$ 方向の単調性と凹性を同時に制約。
  - 変数: $x_f\ (f \in F)$
  - 単調性: $x_f + \varepsilon \leq x_{f+1}\ \ \ (f, f+1 \in F)$
  - 凹性: $x_f - 2x_{f+1} + x_{f+2} \leq 0$
  - 目的関数: $\sum_{f \in F} N_f \cdot (p_f - x_f)^2$
  - $\varepsilon$ の上限: $\max(p_f) / (\lvert F\rvert - 1)$


## クラス仕様

### `RecencyFrequencyScorer`

#### コンストラクタ

```python
RecencyFrequencyScorer(user_col="user", item_col="item", time_col="datetime", unit=1)
```

| パラメータ | 型 | デフォルト | 説明 |
|-----------|-----|-----------|------|
| `user_col` | `str` | `"user"` | ユーザー識別子のカラム名 |
| `item_col` | `str` | `"item"` | 商品識別子のカラム名 |
| `time_col` | `str` | `"datetime"` | 時点カラムのカラム名（`datetime64`・文字列・整数いずれも可） |
| `unit` | `int` | `1` | 最新度の粒度（正の整数）。`unit=7` で週単位、`unit=30` で月単位（近似） |

#### メソッド

##### `fit(df_obs, df_eval, ref=None, recency_limit=None, frequency_limit=None)`

観測ログ DataFrame と評価ログ DataFrame を直接受け取り、$(r, f)$ 別の経験的商品選択確率を推定する。scikit-learn スタイルの主要 fit メソッド。

| パラメータ | 型 | デフォルト | 説明 |
|-----------|-----|-----------|------|
| `df_obs` | `pd.DataFrame` | — | 観測期間の閲覧履歴 |
| `df_eval` | `pd.DataFrame` | — | 評価期間のイベント履歴（閲覧・購買・CV など推定対象のイベント） |
| `ref` | `str \| datetime \| int \| None` | `None` | 最新度計算の基準値（日付または整数）。`None` の場合は `df_obs[time_col].max()` を使用 |
| `recency_limit` | `int \| None` | `None` | 最大最新度。`None` の場合、累積対象イベント数の分布から `RECENCY_LIMIT_RATE` に基づいて自動決定 |
| `frequency_limit` | `int \| None` | `None` | 最大頻度。`None` の場合、累積対象イベント数の分布から `FREQUENCY_LIMIT_RATE` に基づいて自動決定 |

戻り値: `self`

##### `predict(r, f, kind="emp")`

指定した最新度 $r$・頻度 $f$ の商品選択確率を返す。

| パラメータ | 型 | デフォルト | 説明 |
|-----------|-----|-----------|------|
| `r` | `int` | — | 最新度（1が最も直近、数値が大きいほど古い。1以上） |
| `f` | `int` | — | 頻度（観測期間の閲覧回数。1以上） |
| `kind` | `str` | `"emp"` | `"emp"`・`"er"`・`"ef"`・`"mono"`・`"mr"`・`"mf"`・`"mrc"`・`"mfc"`・`"mcc"` のいずれか（長名エイリアスも使用可） |

戻り値: `float`

##### `transform(df, ref=None, kind="emp", user_col=None, item_col=None, time_col=None)`

入力 DataFrame の各 user×item ペアに最新度・頻度・商品選択確率・順位を付与して返す。scikit-learn スタイルの主要 transform メソッド。

| パラメータ | 型 | デフォルト | 説明 |
|-----------|-----|-----------|------|
| `df` | `pd.DataFrame` | — | スコアリング対象の閲覧履歴（観測期間でフィルタ済みを想定） |
| `ref` | `str \| datetime \| int \| None` | `None` | 最新度・頻度の計算基準値（日付または整数）。`None` の場合は `df[time_col].max()` を使用 |
| `kind` | `str` | `"emp"` | `"emp"`・`"er"`・`"ef"`・`"mono"`・`"mr"`・`"mf"`・`"mrc"`・`"mfc"`・`"mcc"` のいずれか（長名エイリアスも使用可） |
| `user_col` | `str \| None` | `None` | ユーザーカラム名。省略時は `__init__` で設定した値を使用 |
| `item_col` | `str \| None` | `None` | 商品カラム名。省略時は `__init__` で設定した値を使用 |
| `time_col` | `str \| None` | `None` | 時点カラム名。省略時は `__init__` で設定した値を使用 |

戻り値: `pd.DataFrame`。ユーザー・商品カラム名は `__init__`（または引数の上書き）で設定した名前になる。その他のカラム: `recency`, `frequency`, `probability`, `order`

##### `evaluate(df_rec, df_eval, order=1, user_col=None, item_col=None)`

推薦結果と評価期間のイベント履歴を比較し、各順位カットオフでの評価指標を返す。
`df_rec` の user/item 列と `df_eval` の user/item 列は内部で `str` にキャストして比較する。

| パラメータ | 型 | デフォルト | 説明 |
|-----------|-----|-----------|------|
| `df_rec` | `pd.DataFrame` | — | `transform()` の出力 |
| `df_eval` | `pd.DataFrame` | — | 評価期間のイベント履歴（閲覧・購買・CV など）。`fit()` に渡したものと同じ DataFrame を渡すことを想定 |
| `order` | `int` | `1` | 評価する最大推薦順位 |
| `user_col` | `str \| None` | `None` | ユーザーカラム名。省略時は `__init__` で設定した値を使用 |
| `item_col` | `str \| None` | `None` | 商品カラム名。省略時は `__init__` で設定した値を使用 |

戻り値: `pd.DataFrame`（カラム: `order`, `n_recommended`, `n_hit`, `precision`, `recall`, `f1`, `recall_norm`, `f1_norm`）

##### `optimize(kind="mono", eps=0.0)`

RF 制約を満たし、経験的商品選択確率との誤差を最小化する最適化商品選択確率を推定する。`fit()` 後に呼び出す。
内部で `optimizer.py` の `RecencyFrequencyOptimizer` を使用して凸2次計画問題を解く。
結果は `kind` に対応する属性（例: `mr_probability_*`、`mono_probability_*`）に格納されるため、複数モデルの結果を同時に保持できる。

| パラメータ | 型 | デフォルト | 説明 |
|-----------|-----|-----------|------|
| `kind` | `str` | `"mono"` | `"mr"`（Recency 1次元・単調性 + 凸性）・`"mf"`（Frequency 1次元・単調性 + 凹性）・`"mono"`（2次元・単調性のみ）・`"mrc"`（2次元・単調性 + Recency 凸性）・`"mfc"`（2次元・単調性 + Frequency 凹性）・`"mcc"`（2次元・単調性 + Recency 凸性 + Frequency 凹性）のいずれか（長名エイリアスも使用可） |
| `eps` | `float` | `0.0` | 単調性制約における隣接値の最小差 $\varepsilon$。`0.0`（デフォルト）のとき広義単調性。正の値を指定すると狭義単調性となり、同一 recency または frequency で確率値が一致しなくなる。上限はデータから自動計算され、超過すると `ValueError` |

戻り値: `self`

##### `export_probability_csv(kind="emp", path=None)`

商品選択確率を CSV ファイルに書き出す。

| パラメータ | 型 | デフォルト | 説明 |
|-----------|-----|-----------|------|
| `kind` | `str` | `"emp"` | `"emp"`・`"er"`・`"ef"`・`"mono"`・`"mr"`・`"mf"`・`"mrc"`・`"mfc"`・`"mcc"`・`"all"` のいずれか（長名エイリアスも使用可）。`"all"` は9モデルをマージして出力（カラム: `emp_probability`, `er_probability`, `ef_probability`, `mono_probability`, `mr_probability`, `mf_probability`, `mrc_probability`, `mfc_probability`, `mcc_probability`） |
| `path` | `str \| None` | `None` | 出力先。`None` の場合カレントディレクトリに `{kind}_probability.csv` を出力。ディレクトリを指定した場合はそのディレクトリにデフォルトファイル名で出力 |

戻り値: なし

##### `plot_probability_surface(kind="emp", title=None, figsize=(6, 5), fontsize=12, recency_label="recency", frequency_label="frequency", probability_label="probability")`

商品選択確率を3次元ワイヤーフレームで可視化し、`matplotlib.figure.Figure` を返す。

Jupyter Lab / Colab では返り値がそのままインライン描画される。
ファイルに保存する場合は `fig.savefig("output.png")` を呼ぶ。
日本語軸ラベルを使用する場合は `pip install rfscorer[ja]` で `japanize-matplotlib` を導入する。

| パラメータ | 型 | デフォルト | 説明 |
|-----------|-----|-----------|------|
| `kind` | `str` | `"emp"` | `"emp"`・`"er"`・`"ef"`・`"mono"`・`"mrc"`・`"mfc"`・`"mcc"` のいずれか（長名エイリアスも使用可）。`"mr"`・`"mf"` は1次元モデルのため `ValueError` を送出する（`plot_marginal_probability()` を使用する） |
| `title` | `str \| None` | `None` | 図のタイトル。`None` の場合は表示しない |
| `figsize` | `tuple[float, float]` | `(6, 5)` | 図のサイズ（インチ）。論文用途では最終印刷サイズに合わせる |
| `fontsize` | `int` | `12` | 軸ラベル・目盛りのフォントサイズ。論文用途では対象ジャーナルの本文サイズ（通常 8〜10 pt）に合わせる |
| `recency_label` | `str` | `"recency"` | x 軸（最新度）のラベル |
| `frequency_label` | `str` | `"frequency"` | y 軸（頻度）のラベル |
| `probability_label` | `str` | `"probability"` | z 軸（確率）のラベル |

戻り値: `matplotlib.figure.Figure`

##### `plot_marginal_probability(axis="recency", kind="emp", title=None, figsize=(5, 4), fontsize=12, recency_label="recency", frequency_label="frequency", probability_label="probability")`

最新度または頻度の一方向の商品選択確率を折れ線グラフで可視化し、`matplotlib.figure.Figure` を返す。
経験的商品選択確率（`emp`）と1次元最適化商品選択確率（`mr`・`mf`）を重ねて表示できる。

Jupyter Lab / Colab では返り値がそのままインライン描画される。
ファイルに保存する場合は `fig.savefig("output.png")` を呼ぶ。
日本語軸ラベルを使用する場合は `pip install rfscorer[ja]` で `japanize-matplotlib` を導入する。

| パラメータ | 型 | デフォルト | 説明 |
|-----------|-----|-----------|------|
| `axis` | `str` | `"recency"` | `"recency"`（最新度方向）または `"frequency"`（頻度方向） |
| `kind` | `str` | `"emp"` | `"emp"`（1次元経験的確率のみ）・`"mr"`（mr 最適化のみ、`axis="recency"` 限定）・`"mf"`（mf 最適化のみ、`axis="frequency"` 限定）・`"all"`（経験的 + 最適化を重ねて表示）のいずれか（長名エイリアスも使用可） |
| `title` | `str \| None` | `None` | 図のタイトル。`None` の場合は表示しない |
| `figsize` | `tuple[float, float]` | `(5, 4)` | 図のサイズ（インチ）。論文用途では最終印刷サイズに合わせる |
| `fontsize` | `int` | `12` | 軸ラベル・目盛りのフォントサイズ。論文用途では対象ジャーナルの本文サイズ（通常 8〜10 pt）に合わせる |
| `recency_label` | `str` | `"recency"` | x 軸のラベル（`axis="recency"` 時に使用） |
| `frequency_label` | `str` | `"frequency"` | x 軸のラベル（`axis="frequency"` 時に使用） |
| `probability_label` | `str` | `"probability"` | y 軸（確率）のラベル |

線スタイル: `kind="all"` のとき emp が実線・最適化が破線。単独表示のときは実線。すべて黒色。

戻り値: `matplotlib.figure.Figure`

##### `show()`

`fit()` 後の集計情報（レコード数・cv 数・期間・上限値）を標準出力に表示する。デバッグ・動作確認用。

戻り値: なし

#### 属性

| 属性 | 型 | 説明 | 利用可能なタイミング |
|------|-----|------|-----------------|
| `recency_limit` | `int` | 最新度の上限値 | `fit()` 後 |
| `frequency_limit` | `int` | 頻度の上限値 | `fit()` 後 |
| `R` | `list[int]` | 最新度のリスト（`range(1, recency_limit+1)`） | `fit()` 後 |
| `F` | `list[int]` | 頻度のリスト（`range(1, frequency_limit+1)`） | `fit()` 後 |
| `RF2N` | `dict` | `(r, f)` → サンプル数 $N_{r,f}$ のマッピング | `fit()` 後 |
| `RF2CV` | `dict` | `(r, f)` → cv 数 $n_{r,f}$ のマッピング | `fit()` 後 |
| `RF2Prob` | `dict` | `(r, f)` → 経験的商品選択確率 $p_{r,f}$ のマッピング | `fit()` 後 |
| `R2N` | `dict` | `r` → 最新度別サンプル数のマッピング（`RF2N` の $f$ 方向集約） | `fit()` 後 |
| `R2CV` | `dict` | `r` → 最新度別 cv 数のマッピング | `fit()` 後 |
| `R2Prob` | `dict` | `r` → 最新度別経験的商品選択確率のマッピング | `fit()` 後 |
| `F2N` | `dict` | `f` → 頻度別サンプル数のマッピング（`RF2N` の $r$ 方向集約） | `fit()` 後 |
| `F2CV` | `dict` | `f` → 頻度別 cv 数のマッピング | `fit()` 後 |
| `F2Prob` | `dict` | `f` → 頻度別経験的商品選択確率のマッピング | `fit()` 後 |
| `emp_probability_` | `pd.DataFrame` | 経験的商品選択確率（カラム: `recency`, `frequency`, `N`, `cv`, `probability`） | `fit()` 後 |
| `recency_probability_` | `pd.DataFrame` | 最新度別経験的商品選択確率（カラム: `recency`, `N`, `cv`, `probability`） | `fit()` 後 |
| `frequency_probability_` | `pd.DataFrame` | 頻度別経験的商品選択確率（カラム: `frequency`, `N`, `cv`, `probability`） | `fit()` 後 |
| `emp_probability_table_` | `pd.DataFrame` | 経験的商品選択確率（横持ち。インデックス: `recency`、カラム: `frequency`） | `fit()` 後 |
| `emp_probability_dict_` | `dict` | 経験的商品選択確率（キー: `(r, f)`、値: `probability`） | `fit()` 後 |
| `er_probability_` | `pd.DataFrame` | er モデル1次元経験的商品選択確率（最新度のみ。ブロードキャストなし）（カラム: `recency`, `probability`） | `fit()` 後 |
| `er_probability_dict_` | `dict` | er モデル1次元経験的商品選択確率（キー: `r`（int）、値: `probability`） | `fit()` 後 |
| `ef_probability_` | `pd.DataFrame` | ef モデル1次元経験的商品選択確率（頻度のみ。ブロードキャストなし）（カラム: `frequency`, `probability`） | `fit()` 後 |
| `ef_probability_dict_` | `dict` | ef モデル1次元経験的商品選択確率（キー: `f`（int）、値: `probability`） | `fit()` 後 |
| `mr_probability_` | `pd.DataFrame` | mr モデル1次元最適化商品選択確率（カラム: `recency`, `probability`） | `optimize(kind="mr")` 後 |
| `mr_probability_dict_` | `dict` | mr モデル最適化商品選択確率（キー: `r`（int）、値: `probability`） | `optimize(kind="mr")` 後 |
| `mf_probability_` | `pd.DataFrame` | mf モデル1次元最適化商品選択確率（カラム: `frequency`, `probability`） | `optimize(kind="mf")` 後 |
| `mf_probability_dict_` | `dict` | mf モデル最適化商品選択確率（キー: `f`（int）、値: `probability`） | `optimize(kind="mf")` 後 |
| `mono_probability_` | `pd.DataFrame` | mono モデル最適化商品選択確率（カラム: `recency`, `frequency`, `probability`） | `optimize(kind="mono")` 後 |
| `mono_probability_table_` | `pd.DataFrame` | mono モデル最適化商品選択確率（横持ち） | `optimize(kind="mono")` 後 |
| `mono_probability_dict_` | `dict` | mono モデル最適化商品選択確率（キー: `(r, f)`、値: `probability`） | `optimize(kind="mono")` 後 |
| `mrc_probability_` | `pd.DataFrame` | mrc モデル最適化商品選択確率（カラム: `recency`, `frequency`, `probability`） | `optimize(kind="mrc")` 後 |
| `mrc_probability_table_` | `pd.DataFrame` | mrc モデル最適化商品選択確率（横持ち） | `optimize(kind="mrc")` 後 |
| `mrc_probability_dict_` | `dict` | mrc モデル最適化商品選択確率（キー: `(r, f)`、値: `probability`） | `optimize(kind="mrc")` 後 |
| `mfc_probability_` | `pd.DataFrame` | mfc モデル最適化商品選択確率（カラム: `recency`, `frequency`, `probability`） | `optimize(kind="mfc")` 後 |
| `mfc_probability_table_` | `pd.DataFrame` | mfc モデル最適化商品選択確率（横持ち） | `optimize(kind="mfc")` 後 |
| `mfc_probability_dict_` | `dict` | mfc モデル最適化商品選択確率（キー: `(r, f)`、値: `probability`） | `optimize(kind="mfc")` 後 |
| `mcc_probability_` | `pd.DataFrame` | mcc モデル最適化商品選択確率（カラム: `recency`, `frequency`, `probability`） | `optimize(kind="mcc")` 後 |
| `mcc_probability_table_` | `pd.DataFrame` | mcc モデル最適化商品選択確率（横持ち） | `optimize(kind="mcc")` 後 |
| `mcc_probability_dict_` | `dict` | mcc モデル最適化商品選択確率（キー: `(r, f)`、値: `probability`） | `optimize(kind="mcc")` 後 |
| `record_num` | `int` | 全閲覧履歴のレコード数 | `fit()` 後 |
| `record_num_obs` | `int` | 観測期間のレコード数 | `fit()` 後 |
| `record_num_eval` | `int` | 評価期間のレコード数 | `fit()` 後 |
| `record_num_target_org` | `int` | フィルタリング前の分析対象レコード数 | `fit()` 後 |
| `record_num_target` | `int` | フィルタリング後の分析対象レコード数 | `fit()` 後 |
| `total_cv_org` | `int` | フィルタリング前の cv 数 | `fit()` 後 |
| `total_cv` | `int` | フィルタリング後の cv 数 | `fit()` 後 |

## ユーティリティ

### `split_by_date(df, target_date, observation_days=28, evaluation_days=7, time_col="datetime")`

`from rfscorer import split_by_date` で利用可能。`target_date` を基準に単一の DataFrame を観測ログ・評価ログに分割するスタンドアロン関数。`RecencyFrequencyScorer` に依存せず、ローリング workflow など複数 `target_date` を渡す研究的用途にも利用できる。

- 観測期間: `max(df の time_col 最小値, target_date - observation_days 時間単位)` 〜 `target_date`（含む）
- 評価期間: `target_date の翌時点` 〜 `min(df の time_col 最大値, target_date + evaluation_days 時間単位)`

| パラメータ | 型 | デフォルト | 説明 |
|-----------|-----|-----------|------|
| `df` | `pd.DataFrame` | — | 分割対象の DataFrame |
| `target_date` | `str \| datetime \| int` | — | 観測期間と評価期間の分割点（日付または整数） |
| `observation_days` | `int \| None` | `28` | `target_date` から遡る最大時間単位数。`None` の場合は df の先頭まで |
| `evaluation_days` | `int \| None` | `7` | `target_date` から進む最大時間単位数。`None` の場合は df の末尾まで |
| `time_col` | `str` | `"datetime"` | 時点カラム名 |

戻り値: `tuple[pd.DataFrame, pd.DataFrame]`（`(df_obs, df_eval)`）。元の df の構造を保持したサブセット。

## データフロー

```
観測ログ (df_obs)    評価ログ (df_eval)
        │                    │
        └──────────┬──────────┘
                   ▼
        fit(df_obs, df_eval)
        （df_obs, df_eval = split_by_date(df, target_date) でユーティリティ分割も可能）
                   │
user / item / time_col に正規化（datetime64・文字列は ordinal 整数に変換）
観測期間・評価期間でフィルタ
r（最新度）・f（頻度）を算出
(r, f) 別に n_{r,f}・N_{r,f} を集計
p_{r,f} = n_{r,f} / N_{r,f}（2次元）、p_r・p_f も同時に計算
        ▼
emp_probability_ / _table_ / _dict_   ← 2次元経験的商品選択確率（emp）
er_probability_ / _dict_                    ← R2Prob を1次元 dict / DataFrame として保持（er）
ef_probability_ / _dict_                    ← F2Prob を1次元 dict / DataFrame として保持（ef）
RF2N / RF2CV / RF2Prob / R2N / R2CV / R2Prob / F2N / F2CV / F2Prob
        │
        ├─  predict(r, f, kind)  ─→ 特定 (r, f) の商品選択確率を返す
        │
        ├─  transform(df, ref, kind)  ─→ user×item に r・f・確率・順位を付与
        │   （df の事前フィルタはユーザー側で行う）
        │
        ├─  plot_probability_surface(kind)  ─→ 3次元ワイヤーフレームで可視化
        │
        ├─  plot_marginal_probability(axis, kind)  ─→ 1次元折れ線グラフで可視化
        │
        ├─  export_probability_csv(kind, path)  ─→ 確率テーブルを CSV に書き出す
        │
        ├─  optimize(kind='mr'|'mf')  ← RecencyFrequencyOptimizer (optimizer.py) に委譲（1次元最適化）
        │   1次元経験的確率を目標とした1次元凸2次計画問題を求解し、結果を1次元 dict に格納（ブロードキャストなし）
        │       ▼
        │   {mr|mf}_probability_ / _dict_
        │
        └─  optimize(kind='mono'|'mrc'|'mfc'|'mcc')  ← RecencyFrequencyOptimizer に委譲（2次元最適化）
            RF 制約付き凸2次計画問題を求解（kind に応じた追加制約を適用）
                ▼
            {mono|mrc|mfc|mcc}_probability_ / _table_ / _dict_
                │
                └─  export_probability_csv(kind='all', path)
                    ─→ emp + er + ef + mr + mf + mono + mrc + mfc + mcc を併記した CSV を書き出す
```

## 入出力例

```python
import pandas as pd
from rfscorer import RecencyFrequencyScorer, split_by_date

# サンプルデータ: ohmsha/PyOptBook (MIT License) — カラム: user_id, item_id, date
url = "https://raw.githubusercontent.com/ohmsha/PyOptBook/main/7.recommendation/access_log.csv"
df = pd.read_csv(url)

scorer = RecencyFrequencyScorer(user_col="user_id", item_col="item_id", time_col="date")

# split_by_date() で観測ログと評価ログを自動分割（推奨）
target_date = "2015-07-06"
df_obs, df_eval = split_by_date(df, target_date=target_date)
scorer.fit(df_obs, df_eval)
scorer.emp_probability_

df_rec = scorer.transform(df_obs, ref=target_date)
prob = scorer.predict(r=1, f=3)
scorer.evaluate(df_rec, df_eval)

# 最適化商品選択確率の推定
scorer.optimize(kind="mono")           # 2次元: 単調性のみ
scorer.optimize(kind="mcc")            # 2次元: 単調性 + Recency 凸性 + Frequency 凹性
scorer.optimize(kind="mr")             # 1次元: Recency 単調性 + 凸性
df_mono = scorer.mono_probability_
df_mcc  = scorer.mcc_probability_

# すべてのモデルを CSV に書き出す
scorer.optimize(kind="mf")
scorer.optimize(kind="mrc")
scorer.optimize(kind="mfc")
scorer.export_probability_csv(kind="all", path="output/probabilities.csv")

# split_by_date() で期間をカスタマイズする場合
df_obs, df_eval = split_by_date(df, target_date="2015-07-06", observation_days=4, evaluation_days=2)
scorer.fit(df_obs, df_eval)

# 期間を明示的に指定する場合（標準 pandas フィルタ）
mask_obs = (df["date"] >= "2015-07-02") & (df["date"] <= "2015-07-06")
mask_eval = (df["date"] >= "2015-07-07") & (df["date"] <= "2015-07-08")
scorer.fit(df[mask_obs], df[mask_eval])
```

